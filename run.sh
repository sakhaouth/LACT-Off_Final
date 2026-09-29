#!/bin/bash

# Suggested by the OOM error itself -- reduces fragmentation, must be set
# before the python process starts.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

llm_model_name=TINYBERT     # fastest option: 2 layers, 128 hidden, ~4.4M params
train_epochs=50
llm_learning_rate=1e-3
lstm_learning_rate=1e-3

# FIX: TinyBERT only has 2 pretrained transformer blocks (DistilGPT2/DistilBERT
# have 6, GPT2/BERT have 12, LLaMA has 32). New_Time_LLM.py clips to the right
# max per model regardless, but set it correctly here so you're not surprised.
llama_layers=2
# This is now the *effective* batch size (used for LSTM directly, and as the
# gradient-accumulation target for the LLM branch).
batch_size=64

# NEW: actual per-step batch size fed into the GPT2 forward pass. This is what
# controls GPU memory for the LLM branch -- keep it small, batch_size handles
# the rest via gradient accumulation.
llm_micro_batch_size=8

d_model=32
d_ff=128
comment='time-series-predictor'

python3 -m load_predictor.claude_train_predictor \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./CSV \
  --data_path ./CSV/local_load_raw-4.csv \
  --model_id S_LOAD_16_1 \
  --llm_model $llm_model_name \
  --data S_LOAD \
  --features M \
  --seq_len 16 \
  --label_len 8 \
  --pred_len 1 \
  --factor 3 \
  --enc_in 12 \
  --dec_in 12 \
  --c_out 12 \
  --des 'Exp' \
  --itr 1 \
  --llm 1 \
  --lstm 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --freq 15min \
  --seasonal_patterns "every 15 minute" \
  --content "This dataset contains server load information recorded at 15-minute intervals, with each row representing four metrics—total CPU cycle load, total memory load, total number of tasks, and time-to-live (TTL) load—for three priority queues: low, medium, and high, arranged sequentially by priority" \
  --batch_size $batch_size \
  --llm_micro_batch_size $llm_micro_batch_size \
  --llm_learning_rate $llm_learning_rate \
  --lstm_learning_rate $lstm_learning_rate \
  --llm_layers $llama_layers \
  --gradient_checkpointing 1 \
  --use_amp 1 \
  --train_epochs $train_epochs \
  --model_comment $comment
