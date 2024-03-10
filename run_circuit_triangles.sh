#!/bin/bash

DATA=$1
NODE=$2
EDGE=$3
AGG=$4

python circuit_triangles.py \
    --model EleutherAI/pythia-70m-deduped \
    --num_examples 100 \
    --batches 10 \
    --dataset $DATA \
	--node_threshold $NODE \
	--edge_threshold $EDGE \
	--aggregation $AGG \
    --example_length 8 \
    --dict_id id

# --submodules model.gpt_neox.layers.{}.attention.dense, model.gpt_neox.layers.{}.mlp.dense_4h_to_h,model.gpt_neox.layers.{}