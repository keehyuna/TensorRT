"""
.. _run_llm:

Running LLM inference with Torch-TensorRT
==========================================================

This script illustrates Torch-TensorRT workflow with dynamo backend on popular LLM models.
"""

import argparse
import copy
import os
import timeit
from contextlib import nullcontext
import json
# %%
# Imports and Model Definition
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
import torch
import torch_tensorrt
import torch.nn as nn
from torchtrt_ext import register_sdpa
from torchtrt_ext import register_dequant
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import (
    export_llm,
    generate,
    generate_with_static_cache,
    record_stats,
    time_generate,
    quantize_model,
)
import huggingface_hub
from huggingface_hub import snapshot_download

import modelopt.torch.quantization as mtq
from safetensors import safe_open

DEVICE = torch.device("cuda:0")

quantize_op = torch.ops.tensorrt.quantize_op
dequantize_op = torch.ops.torchtrt_ex.dequantize_op
class TensorRTQuantizedLinear(torch.nn.Module):
    """TensorRT quantized linear layer."""
    
    def __init__(self, original_linear: nn.Linear, input_amax, weight_scale):
        super().__init__()
        self.original_linear = original_linear
        
        # Copy the original weights and bias
        #self.weight = nn.Parameter(original_linear.weight.clone()).cuda()
        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.clone()).cuda()
        else:
            self.bias = None
        
        # Quantization parameters
        self.input_amax = nn.Parameter(input_amax).cuda()
        self.weight_scale = nn.Parameter(weight_scale).cuda()
    
    def forward(self, x):
        # Quantized forward pass
        x_quantized = quantize_op(
            x, self.input_amax, num_bits=8, exponent_bits=4, unsigned=False, narrow_range=False
        )
        #weight = (self.weight * self.weight_scale).to(torch.float32)
        weight = dequantize_op(self.original_linear.weight, self.weight_scale)
        # Perform quantized linear operation
        output = torch.nn.functional.linear(x_quantized, weight, self.bias)
        #output = torch.ops.aten.linear.default(
        #    x_quantized,
        #    self.weight,
        #    self.bias
        #)

        return output
    
def download_hf_model(model: str):
    ignore_patterns = ["original/**/*"]
    
    hf_folder = snapshot_download(
        model,
        local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
        ignore_patterns=ignore_patterns,
        revision=None)
    return hf_folder

def apply_quantized_linear(model, model_name):
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
            
            weight_scale = tensors[weight_scale_name]
            input_amax = tensors[input_scale_name] * 448.0
            #dequantized_weight_data = weight_data * weight_scale
            if module.weight.dtype != torch.float8_e4m3fn:
                raise RuntimeError("Expected module weight dtype to be float8_e4m3fn, but got {module.weight.dtype}")
            quantized_module = TensorRTQuantizedLinear(module, input_amax, weight_scale)

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
def get_model(args):
    """
    Load and configure the language model for inference.

    This function loads a pre-trained causal language model using the specified
    model name and configures it with the appropriate precision and settings
    for inference.

    Args:
        args: Parsed command line arguments containing:
            - model (str): Name or path of the model to load
            - precision (str): Precision to use ("FP16", "BF16", or "FP32")

    Returns:
        torch.nn.Module: The loaded and configured model ready for inference,
            moved to CUDA device with the specified precision
    """

    with torch.no_grad():
        model = (
            AutoModelForCausalLM.from_pretrained(
                args.model,
                use_cache=False,
                attn_implementation="sdpa",
            )
            .eval()
            .cuda()
        )
    # Do not convert model parameter dtypes when using pre-quantized models
    if args.pre_quantized:
        model = apply_quantized_linear(model, args.model)
        assert args.precision == "None", "Precision should be None when using pre-quantized models"
    else:
        if args.precision == "FP16":
            model = model.to(torch.float16)
        elif args.precision == "BF16":
            model = model.to(torch.bfloat16)
        else:
            model = model.to(torch.float32)

    return model

def compile_torchtrt(model, input_ids, args):
    """
    Compile a PyTorch model to TensorRT using torch_tensorrt.dynamo.compile.

    This function exports the given model to a TorchScript representation and then
    compiles it to TensorRT for optimized inference. The compilation process includes
    precision-specific optimizations and various performance tuning parameters.

    Args:
        model (torch.nn.Module): The PyTorch model to compile
        input_ids (torch.Tensor): Input token IDs tensor used for model export
        args: Parsed command line arguments containing:
            - num_tokens (int): Number of tokens to generate (used for max sequence length)
            - precision (str): Precision to use ("FP16", "BF16", or "FP32")
            - debug (bool): Whether to enable debug logging
            - min_block_size (int): Minimum block size for TensorRT compilation

    Returns:
        torch_tensorrt.dynamo.TorchTensorRTModule: The compiled TensorRT model ready
            for optimized inference
    """
    max_seq_len = input_ids.shape[1] + args.num_tokens
    ep = export_llm(model, input_ids, max_seq_len=max_seq_len)
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0).to(DEVICE)
    # Set precision specific flags
    use_fp32_acc = False
    use_explicit_typing = False
    if args.precision == "FP16":
        enabled_precisions = {torch.float32}
        use_fp32_acc = True
        use_explicit_typing = True
    elif args.precision == "BF16":
        enabled_precisions = {torch.bfloat16}
        use_fp32_acc = False
    else:
        enabled_precisions = {torch.float32}
    
    qformat = "_q_"+ args.qformat if args.qformat else ""
    
    logging_dir = f"./{args.model}_{args.precision}{qformat}"
    #with torch_tensorrt.logging.debug() if args.debug else nullcontext():
    with torch_tensorrt.dynamo.Debugger(
        "debug",
        logging_dir=logging_dir,
        #capture_fx_graph_after=["constant_fold"],
        #save_engine_profile=True,
        #profile_format="trex",
        engine_builder_monitor=False,
        #save_layer_info=True,
    ) if args.debug else nullcontext():
        trt_model = torch_tensorrt.dynamo.compile(
            ep,
            inputs=[input_ids, position_ids],
            enabled_precisions=enabled_precisions,
            # truncate_double=True,
            use_explicit_typing=use_explicit_typing,
            use_fp32_acc=use_fp32_acc,
            device=DEVICE,
            disable_tf32=True,
            use_python_runtime=True,
            debug=args.debug,
            offload_module_to_cpu=True,
            min_block_size=args.min_block_size,
        )

    return trt_model


def print_outputs(backend_name, gen_tokens, tokenizer):
    """
    Print the generated tokens from the model.
    """
    print(f"========= {backend_name} =========")
    print(
        f"{backend_name} model generated text: ",
        tokenizer.decode(gen_tokens[0], skip_special_tokens=True),
    )
    print("===================================")


def measure_perf(trt_model, input_signature, backend_name):
    """
    Measure the performance of a TensorRT model by running it multiple times and
    calculating the average time per iteration.
    """
    total_time = 0
    iterations = 10

    print("Running warmup iteration...")
    # Warmup run
    _ = trt_model(*input_signature)
    torch.cuda.synchronize()

    print(f"Measuring performance over {iterations} iterations...")
    for i in range(iterations):
        start_time = timeit.default_timer()
        _ = trt_model(*input_signature)
        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        iter_time = end_time - start_time
        total_time += iter_time

    avg_time = total_time / iterations
    print(
        f"Backend: {backend_name} Average time per iteration: {avg_time*1000:.4f} milliseconds"
    )
    print(
        f"Backend: {backend_name} Average throughput: {1.0/avg_time:.2f} iterations/second"
    )


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Run inference on a model with random input values"
    )
    arg_parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Name of LLM model",
    )
    arg_parser.add_argument(
        "--tokenizer",
        type=str,
        default="",
        help="Name of LLM model tokenizer",
    )
    arg_parser.add_argument(
        "--prompt", type=str, default="What is parallel programming ?", help="Prompt"
    )
    arg_parser.add_argument(
        "--precision",
        type=str,
        default="FP16",
        help="Precision to use in the model. Options: FP16, BF16, FP32",
    )
    arg_parser.add_argument(
        "--iterations", type=int, default=5, help="no. of iterations to run"
    )
    arg_parser.add_argument(
        "--min_block_size", type=int, default=1, help="no. of iterations to run"
    )
    arg_parser.add_argument(
        "--num_tokens",
        type=int,
        default=128,
        help="no. of output tokens to be generated",
    )
    arg_parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size used for benchmarking"
    )
    arg_parser.add_argument(
        "--isl",
        type=int,
        default=2048,
        help="Input sequence length used for benchmarking",
    )
    arg_parser.add_argument(
        "--enable_pytorch_run",
        action="store_true",
        help="Enable pytorch run (default: False)",
    )
    arg_parser.add_argument(
        "--cache",
        type=str,
        default="",
        help="Type of KV cache to use. Options: static_v1, static_v2",
    )
    arg_parser.add_argument(
        "--cudagraph", action="store_true", help="Enable cudagraphs (default: False)"
    )
    arg_parser.add_argument(
        "--debug", action="store_true", help="Enable debug (default: False)"
    )
    arg_parser.add_argument(
        "--benchmark", action="store_true", help="Enable benchmark (default: False)"
    )
    arg_parser.add_argument(
        "--qformat",
        help=(
            "Quantization format"
        ),
        default=None,
    )
    arg_parser.add_argument(
        "--pre_quantized",
        action="store_true",
        help="Use pre-quantized model weights (default: False)",
    )
    args = arg_parser.parse_args()
    with torch.inference_mode():
        model = get_model(args)

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
        # Set pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Prepare input for benchmarking or evaluation
        if args.benchmark:
            input_ids = torch.randint(
                1, 10000, (args.batch_size, args.isl), dtype=torch.int64
            ).to(model.device)
            position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0).to(DEVICE)
        else:
            model_inputs = tokenizer(args.prompt, return_tensors="pt")
            input_ids = model_inputs["input_ids"].to(DEVICE)
            position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0).to(DEVICE)

        MAX_OUTPUT_SEQ_LENGTH = input_ids.shape[1] + args.num_tokens
        # Pyt
        pyt_gen_tokens = None
        pyt_timings = None
        pyt_stats = None
        if args.qformat == "fp8":
            model = quantize_model(model, tokenizer)
        if args.enable_pytorch_run:
            pyt_gen_tokens = generate(
                model, input_ids.clone(), MAX_OUTPUT_SEQ_LENGTH, tokenizer.eos_token_id
            )
            if args.benchmark:
                pyt_timings = time_generate(
                    generate,
                    model,
                    input_ids.clone(),
                    MAX_OUTPUT_SEQ_LENGTH,
                    tokenizer.eos_token_id,
                    iterations=args.iterations,
                )
                pyt_stats = record_stats(
                    "PyTorch",
                    pyt_timings,
                    args.precision,
                    batch_size=args.batch_size,
                    compile_time_s=None,
                )

        if args.cache == "static_v1":
            # This import is required to register static v1 KV cache transformations as lowering passes
            import static_cache_v1
        if args.cache == "static_v2":
            # This import is required to register static v2 KV cache transformations as lowering passes
            import static_cache_v2

        # Compile the model with Torch-TensorRT
        trt_model = compile_torchtrt(model, input_ids, args)

        if args.cache == "static_v1" or args.cache == "static_v2":
            if args.cudagraph:
                # Run a decoding loop with prefill and generate phases so that the CUDAGraph is recorded for both of these phases.
                # trt_input_signature = (input_ids.clone(),) + get_zeroed_kv_cache_inputs(trt_model)
                torch_tensorrt.runtime.set_cudagraphs_mode(True)

            trt_gen_tokens = generate_with_static_cache(
                trt_model,
                input_ids.clone(),
                MAX_OUTPUT_SEQ_LENGTH,
                tokenizer.eos_token_id,
            )

            if args.benchmark:
                trt_timings = time_generate(
                    generate_with_static_cache,
                    trt_model,
                    input_ids.clone(),
                    MAX_OUTPUT_SEQ_LENGTH,
                    tokenizer.eos_token_id,
                    iterations=args.iterations,
                )
        else:
            trt_gen_tokens = generate(
                trt_model,
                input_ids.clone(),
                MAX_OUTPUT_SEQ_LENGTH,
                tokenizer.eos_token_id,
            )
            if args.benchmark:
                trt_timings = time_generate(
                    generate,
                    trt_model,
                    input_ids.clone(),
                    MAX_OUTPUT_SEQ_LENGTH,
                    tokenizer.eos_token_id,
                    iterations=args.iterations,
                )

        if args.benchmark:
            trt_stats = record_stats(
                "TensorRT",
                trt_timings,
                args.precision,
                batch_size=args.batch_size,
                compile_time_s=None,
            )
        match_result = "N/A"
        model_name = args.model.replace("/", "_")
        qformat = args.qformat if args.qformat else "no_quant"

        if not args.benchmark:
            if args.enable_pytorch_run:
                print_outputs("PyTorch", pyt_gen_tokens, tokenizer)

            print_outputs("TensorRT", trt_gen_tokens, tokenizer)

            if args.enable_pytorch_run:
                print(
                    f"PyTorch and TensorRT outputs match: {torch.equal(pyt_gen_tokens, trt_gen_tokens)}"
                )
                match_result = str(torch.equal(pyt_gen_tokens, trt_gen_tokens))
                out_json_file = f"{model_name}_{qformat}_match.json"
                result = {}
                result["match"] = match_result
                with open(os.path.join("result", out_json_file), "w") as f:
                    json.dump(result, f, indent=4)
                    print(f"Results saved to {out_json_file}")
        if args.benchmark:
            result = {}
            args_dict = vars(args)
            
            result["args"] = args_dict
            result["pyt_stats"] = pyt_stats if args.enable_pytorch_run else None
            result["trt_stats"] = trt_stats if args.benchmark else None
            out_json_file = f"{model_name}_{qformat}_benchmark.json"            
            with open(os.path.join("result0731", out_json_file), "w") as f:
                json.dump(result, f, indent=4)
                print(f"Results saved to {out_json_file}")
            if args.enable_pytorch_run:
                print("=========PyTorch PERFORMANCE============ \n")
                print(pyt_stats)
            print("===================== \n")
            print("=========TensorRT PERFORMANCE============ \n")
            print(trt_stats)
