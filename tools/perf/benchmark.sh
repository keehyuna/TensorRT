#!/bin/bash

MODELS_DIR="models"

# Download the Torchscript models
#python hub.py

#batch_sizes=(1 2 4 8 16 32 64 128 256)
batch_sizes=(1 2 4 8 16)
#backends=("torch" "ts_trt" "dynamo" "torch_compile" "inductor" "tensorrt")
backends=("torch" "dynamo")
weight_streaming_percents=("auto" "20%" "40%" "80%" "100%")


# Benchmark Resnet50 model
echo "Benchmarking Resnet50 model"
for bs in ${batch_sizes[@]}
do
  for backend in ${backends[@]}
  do
    for weight_streaming_percent in ${weight_streaming_percents[@]}
    do
        python perf_run.py --model ${MODELS_DIR}/resnet50_scripted.jit.pt \
                           --model_torch resnet50 \
                           --inputs="(${bs}, 3, 224, 224)" \
                           --batch_size ${bs} \
                           --truncate \
                           --backends ${backend} \
                           --weight_streaming_percent ${weight_streaming_percent} \
                           --report "resnet50_perf_bs${bs}_backend_${backend}_ws_${weight_streaming_percent}.csv"
    done
  done
done

# Collect and concatenate all results
echo "Concatenating all results"
python accumulate_results.py
