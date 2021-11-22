#!/bin/bash

TPU_VISIBLE_CHIPS=0 TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 python3 t5_generation.py --model-path=/home/qj213/INT/data/hf_model \
    --eval-path=/home/qj213/hf_seq2seq/int_reproduction/int_data/part_1_test.json \
    --dump-path=/home/qj213/hf_seq2seq/int_reproduction/int_data &
TPU_VISIBLE_CHIPS=1 TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 python3 t5_generation.py --model-path=/home/qj213/INT/data/hf_model \
    --eval-path=/home/qj213/hf_seq2seq/int_reproduction/int_data/part_2_test.json \
    --dump-path=/home/qj213/hf_seq2seq/int_reproduction/int_data &
TPU_VISIBLE_CHIPS=2 TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 python3 t5_generation.py --model-path=/home/qj213/INT/data/hf_model \
    --eval-path=/home/qj213/hf_seq2seq/int_reproduction/int_data/part_3_test.json \
    --dump-path=/home/qj213/hf_seq2seq/int_reproduction/int_data &
TPU_VISIBLE_CHIPS=3 TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 python3 t5_generation.py --model-path=/home/qj213/INT/data/hf_model \
    --eval-path=/home/qj213/hf_seq2seq/int_reproduction/int_data/part_4_test.json \
    --dump-path=/home/qj213/hf_seq2seq/int_reproduction/int_data &
TPU_VISIBLE_CHIPS=4 TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 python3 t5_generation.py --model-path=/home/qj213/INT/data/hf_model \
    --eval-path=/home/qj213/hf_seq2seq/int_reproduction/int_data/part_5_test.json \
    --dump-path=/home/qj213/hf_seq2seq/int_reproduction/int_data &
TPU_VISIBLE_CHIPS=5 TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 python3 t5_generation.py --model-path=/home/qj213/INT/data/hf_model \
    --eval-path=/home/qj213/hf_seq2seq/int_reproduction/int_data/part_6_test.json \
    --dump-path=/home/qj213/hf_seq2seq/int_reproduction/int_data &
TPU_VISIBLE_CHIPS=6 TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 python3 t5_generation.py --model-path=/home/qj213/INT/data/hf_model \
    --eval-path=/home/qj213/hf_seq2seq/int_reproduction/int_data/part_7_test.json \
    --dump-path=/home/qj213/hf_seq2seq/int_reproduction/int_data &
TPU_VISIBLE_CHIPS=7 TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 python3 t5_generation.py --model-path=/home/qj213/INT/data/hf_model \
    --eval-path=/home/qj213/hf_seq2seq/int_reproduction/int_data/part_8_test.json \
    --dump-path=/home/qj213/hf_seq2seq/int_reproduction/int_data &
