#!/bin/bash

# EleutherAI/pythia-70m-deduped

MODEL=$1
NUM_EXAMPLE=$2
DATA=$3
NODE=$4
EDGE=$5
AGG=$6
LENGTH=$7
DICT_ID=$8
DATA_TYPE=$9



python circuit.py \
    --model $MODEL \
    --num_examples $NUM_EXAMPLE \
    --batch_size 10 \
    --prune \
    --dataset $DATA \
	--node_threshold $NODE \
	--edge_threshold $EDGE \
	--aggregation $AGG \
    --example_length $LENGTH \
    --dict_id $DICT_ID \
    --data_type $DATA_TYPE
    