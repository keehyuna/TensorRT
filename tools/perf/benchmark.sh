#!/bin/bash

MODELS_DIR="models"

# Download the Torchscript models
#python hub.py

#large_model_batch_sizes=(1 2 4 8 16 32 64)
#backends_no_torchscript=("torch" "dynamo" "torch_compile" "inductor" "tensorrt")
#weight_streaming_percents=("auto" "20%" "40%" "80%" "100%")

large_model_batch_sizes=(1 2 4)
backends_no_torchscript=("torch" "dynamo")
weight_streaming_percents=("disabled" "20%" "40%" "80%" "100%" "auto")

# Benchmark Stable Diffusion UNet model
echo "Benchmarking SD UNet model"
for bs in ${large_model_batch_sizes[@]}
do
  for backend in ${backends_no_torchscript[@]}
  do
    for weight_streaming_percent in ${weight_streaming_percents[@]}
    do
    python perf_run.py --model_torch sd_unet \
                       --precision fp16 --inputs="(${bs}, 4, 64, 64);(${bs});(${bs}, 1, 768)" \
                       --batch_size ${bs} \
                       --truncate \
                       --backends ${backend} \
                       --weight_streaming_percent ${weight_streaming_percent} \
                       --report "sd_unet_perf_bs${bs}_backend_${backend}_ws_${weight_streaming_percent}.csv"
    done
  done
done

# Collect and concatenate all results
echo "Concatenating all results"
python accumulate_results.py


