import copy
import timeit
import os
import numpy as np
import torch
from transformers import StoppingCriteriaList
from transformers.generation.stopping_criteria import (
    EosTokenCriteria,
    MaxLengthCriteria,
)
import modelopt.torch.quantization as mtq
from modelopt.torch.utils.dataset_utils import (
    create_forward_loop,
    get_dataset_dataloader,
)
from modelopt.torch.quantization.utils import export_torch_mode
import huggingface_hub
from huggingface_hub import snapshot_download
from safetensors import safe_open
def export_llm(model, inputs, min_seq_len=1, max_seq_len=16):
    """
    Exports the LLM model into an ExportedProgram with dynamic shapes.
    In the case of guard failures due to some PyTorch kernel implements, we also
    try to re-export the graph by expressing them as runtime assert nodes
    """
    with torch.no_grad():
        with export_torch_mode():
            # max=1024 has contraint violation error. https://github.com/pytorch/pytorch/issues/125604
            seq_len = torch.export.Dim("seq_len", min=min_seq_len, max=max_seq_len)
            position_ids = torch.arange(inputs.shape[1]).unsqueeze(0).to(inputs.device)
            try:
                print("Trying to export the model using torch.export.export()..")
                # strict=False only enables aotautograd tracing and excludes dynamo.
                ep = torch.export.export(
                    model,
                    args=(inputs,),
                    kwargs={"position_ids": position_ids},
                    dynamic_shapes=({1: seq_len}, {1: seq_len}),
                    strict=False,
                )
            except:
                print(
                    "Trying torch.export._trace._export to trace the graph since torch.export.export() failed"
                )
                # This API is used to express the constraint violation guards as asserts in the graph.
                ep = torch.export._trace._export(
                    model,
                    args=(inputs,),
                    kwargs={"position_ids": position_ids},
                    dynamic_shapes=({1: seq_len}, {1: seq_len}),
                    strict=False,
                    allow_complex_guards_as_runtime_asserts=True,
                )

    return ep


def get_zeroed_static_cache_inputs(model: torch.fx.GraphModule):
    """
    Extracts and returns zeroed static KV cache tensors from a torch.fx.GraphModule. This should only be used for static cache_v1 and static cache_v2.

    This function identifies placeholder nodes in the graph that represent KV cache tensors,
    and creates zeroed tensors with the same shape, dtype, and device as the original placeholders.

    Args:
        model (torch.fx.GraphModule): The exported model graph containing KV cache placeholders

    Returns:
        tuple: A tuple of zeroed tensors corresponding to the KV cache placeholders in the graph
    """
    # placeholder nodes are expected to be in the following order:
    # input_ids, kv_cache_key, kv_cache_value, start_idx, end_idx
    placeholder_nodes = [node for node in model.graph.nodes if node.op == "placeholder"]
    # The first two inputs are input_ids, position_ids. The last two inputs are start_idx, end_idx. In between are the KV cache tensors.
    kv_cache_inputs = placeholder_nodes[2:-2]
    zeroed_kv_cache_inputs = []
    for input in kv_cache_inputs:
        zeroed_kv_cache_inputs.append(
            torch.zeros(
                input.meta["val"].shape,
                dtype=input.meta["val"].dtype,
                device=torch.device("cuda:0"),
            )
        )

    return tuple(zeroed_kv_cache_inputs)


def get_zeroed_dynamic_cache_inputs(model: torch.fx.GraphModule):
    """
    Extracts and returns zeroed KV cache tensors from a torch.fx.GraphModule. This should only be used for dynamic cache.

    This function identifies placeholder nodes in the graph that represent KV cache tensors,
    and creates zeroed tensors with the same shape, dtype, and device as the original placeholders.

    Args:
        model (torch.fx.GraphModule): The exported model graph containing KV cache placeholders

    Returns:
        tuple: A tuple of zeroed tensors corresponding to the KV cache placeholders in the graph
    """
    # placeholder nodes are expected to be in the following order:
    # input_ids, kv_cache_key, kv_cache_value, start_idx, end_idx
    placeholder_nodes = [node for node in model.graph.nodes if node.op == "placeholder"]
    # The first two inputs are input_ids, position_ids. The last input is is_generate. In between are the KV cache tensors.
    kv_cache_inputs = placeholder_nodes[2:-1]
    zeroed_kv_cache_inputs = []
    for input in kv_cache_inputs:
        zeroed_kv_cache_inputs.append(
            torch.zeros(
                input.meta["val"].shape,
                dtype=input.meta["val"].dtype,
                device=torch.device("cuda:0"),
            )
        )

    return tuple(zeroed_kv_cache_inputs)


def generate(model, input_seq, max_output_seq_length, eos_token_id, benchmark=True):
    """
    Greedy decoding of the model. This generates up to max_tokens.
    """
    stopping_criteria = StoppingCriteriaList(
        [
            MaxLengthCriteria(max_length=max_output_seq_length),
            EosTokenCriteria(eos_token_id=eos_token_id),
        ]
    )
    isl = input_seq.shape[1]
    osl = max_output_seq_length - isl

    num_tokens_generated = 0
    while num_tokens_generated < osl:
        position_ids = torch.arange(input_seq.shape[1]).unsqueeze(0).cuda()
        outputs = model(input_seq, position_ids=position_ids)
        logits = outputs.logits
        next_token_logits = logits[:, -1, :]
        next_tokens = torch.argmax(next_token_logits, dim=-1)
        input_seq = torch.cat([input_seq, next_tokens[:, None]], dim=-1)
        num_tokens_generated += 1
        # TODO: Handle batch in this check
        if not benchmark and stopping_criteria(input_seq, logits).item():
            break

    return input_seq


def generate_with_static_cache(model, input_seq, max_output_seq_length, eos_token_id):
    """
    Greedy decoding of the model with static KV cache.
    """
    start_idx = 0
    end_idx = input_seq.shape[1]
    position_ids = torch.arange(input_seq.shape[1]).unsqueeze(0).cuda()
    output_seq = input_seq.clone()
    # TODO: Confirm this: When end_idx = max_output_seq_length-1, number of tokens generated = OSL
    num_tokens_generated = 0
    kv_cache = get_zeroed_static_cache_inputs(model)
    while end_idx < max_output_seq_length:
        position_ids = (
            torch.tensor([[start_idx]], dtype=torch.int64).cuda()
            if input_seq.shape[1] == 1
            else position_ids
        )
        input_signature = (input_seq, position_ids, *kv_cache, start_idx, end_idx)
        logits_keys_values = model(*input_signature)
        num_tokens_generated += 1
        logits = logits_keys_values[0]
        kv_cache = logits_keys_values[1:]
        next_token_logits = logits[:, -1, :]
        next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        output_seq = torch.cat([output_seq, next_tokens], dim=-1)
        input_seq = next_tokens
        start_idx = end_idx
        end_idx = start_idx + 1
    return output_seq


def generate_with_dynamic_cache(model, input_seq, max_output_seq_length, eos_token_id):
    """
    Greedy decoding of the model with dynamic KV cache.
    """
    position_ids = torch.arange(input_seq.shape[1]).unsqueeze(0).cuda()
    output_seq = input_seq.clone()
    num_output_tokens = max_output_seq_length - input_seq.shape[1]
    num_tokens_generated = 0
    kv_cache = get_zeroed_dynamic_cache_inputs(model)
    last_position_id = position_ids[-1, -1].item()
    breakpoint()
    while num_tokens_generated < num_output_tokens:
        is_generate = False if input_seq.shape[1] > 1 else True
        position_ids = (
            torch.tensor([[last_position_id + 1]], dtype=torch.int64).cuda()
            if input_seq.shape[1] == 1
            else position_ids
        )
        input_signature = (input_seq, position_ids, *kv_cache, is_generate)
        logits_keys_values = model(*input_signature)
        num_tokens_generated += 1
        logits = logits_keys_values[0]
        kv_cache = logits_keys_values[1:]
        next_token_logits = logits[:, -1, :]
        next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        output_seq = torch.cat([output_seq, next_tokens], dim=-1)
        input_seq = next_tokens
        last_position_id += 1
    return output_seq


def time_generate(
    generate_fn, model, inputs, output_seq_length, eos_token_id, iterations=10
):
    """
    Measure the time for generating a sentence over certain number of iterations
    """
    timings = []
    for _ in range(iterations):
        start_time = timeit.default_timer()
        _ = generate_fn(model, inputs, output_seq_length, eos_token_id)
        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        timings.append(end_time - start_time)

    return timings


def record_stats(backend, timings, precision, batch_size=1, compile_time_s=None):
    """
    Records different timing stats and adds it to the result
    """
    times = np.array(timings)
    speeds = batch_size / times
    time_mean = np.mean(times).item()
    time_med = np.median(times).item()
    time_99th = np.percentile(times, 99).item()
    time_std = np.std(times, ddof=0).item()
    speed_mean = np.mean(speeds).item()
    speed_med = np.median(speeds).item()

    stats = {
        "Backend": backend,
        "Precision": precision,
        "Batch size": batch_size,
        "Median(FPS)": speed_med,
        "Mean(FPS)": speed_mean,
        "Median-Latency(ms)": time_med * 1000,
        "Mean-Latency(ms)": time_mean * 1000,
        "Latency-StdDev(ms)": time_std * 1000,
        "Compile Time(s)": compile_time_s,
    }
    return stats

def quantize_model(model, tokenizer):
    calib_dataloader = get_dataset_dataloader(
        tokenizer=tokenizer,
        batch_size=32,
        num_samples=512,
        device=torch.device("cuda:0"),
    )

    quant_cfg = mtq.FP8_DEFAULT_CFG
    calibrate_loop = create_forward_loop(dataloader=calib_dataloader)

    model = mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)
    print(f"Quantization done.")

    return model

quantize_op = torch.ops.tensorrt.quantize_op

class TensorRTQuantizedLinear(torch.nn.Module):
    """TensorRT quantized linear layer."""
    
    def __init__(self, original_linear: torch.nn.Linear, input_amax, weight_amax):
        super().__init__()
        self.original_linear = original_linear
        
        # Copy the original weights and bias
        #self.weight = nn.Parameter(original_linear.weight.clone()).cuda()
        if original_linear.bias is not None:
            self.bias = torch.nn.Parameter(original_linear.bias.clone()).cuda()
        else:
            self.bias = None
        
        # Quantization parameters
        self.input_amax = torch.nn.Parameter(input_amax).cuda()
        self.weight_amax = torch.nn.Parameter(weight_amax).cuda()
    
    def forward(self, x):
        # Quantized forward pass
        x_quantized = quantize_op(
            x, self.input_amax, num_bits=8, exponent_bits=4, unsigned=False, narrow_range=False
        )
        #weight = (self.weight * self.weight_scale).to(torch.float32)
        quantized_weight = quantize_op(
            self.original_linear.weight, self.weight_amax, num_bits=8, exponent_bits=4, unsigned=False, narrow_range=False
        )

        # Perform quantized linear operation
        return torch.nn.functional.linear(x_quantized, quantized_weight, self.bias)
        
    
def download_hf_model(model: str):
    ignore_patterns = ["original/**/*"]
    
    hf_folder = snapshot_download(
        model,
        local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
        ignore_patterns=ignore_patterns,
        revision=None)
    return hf_folder

def convert_linear_to_tensorrt_quantized(model, model_name):
    if os.path.isdir(model_name):
        hf_folder = model_name
    else:
        hf_folder = download_hf_model(model_name)
    tensors = {}
    for file in os.listdir(hf_folder):
        if file.endswith(".safetensors"):
            with safe_open(os.path.join(hf_folder, file), framework="pt", device="cpu") as f:
                # Get all tensor names
                tensor_names = f.keys()
                print(f"Available tensors: {tensor_names}")
                for name in tensor_names:
                    tensors[name] = f.get_tensor(name)

    for name, module in model.named_modules():
        target = torch.nn.modules.linear.Linear
        if isinstance(module, target):
            #print(name)
            weight_scale_name = name + ".weight_scale"
            input_scale_name = name + ".input_scale"
            if weight_scale_name not in tensors:
                print(f"Weight scale tensor {weight_scale_name} not found")
                continue

            if input_scale_name not in tensors:
                print(f"Input scale tensor {input_scale_name} not found")
                continue
            
            #weight_data = tensors[name+".weight"].to(torch.float32)
            
            weight_amax = tensors[weight_scale_name] * 448.0
            input_amax = tensors[input_scale_name] * 448.0
            #dequantized_weight_data = weight_data * weight_scale
            module.weight.data = module.weight.to(torch.float32) * tensors[weight_scale_name]
            quantized_module = TensorRTQuantizedLinear(module, input_amax, weight_amax)

            # Replace in parent module
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            
            # update state_dict
            #quantized_module.weight.data = dequantized_weight_data
            if parent_name:
                parent_module = model.get_submodule(parent_name)
                setattr(parent_module, child_name, quantized_module)
            else:
                # Root module
                setattr(model, child_name, quantized_module)
    
    return model