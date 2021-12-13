# Test
USE_TORCH=false python3 run_summarization_flax.py \
    --output_dir /tmp/tst-summarization \
    --tpu_num_cores 8 \
	--model_name_or_path t5-small \
    --source_prefix "summarize: " \
	--dataset_name=xsum \
    --num_train_epochs 10 \
    --do_train --do_eval --do_predict --predict_with_generate \
    --learning_rate 5e-5 --warmup_steps 300 \
    --per_device_train_batch_size 8 \
	--per_device_eval_batch_size 8 \
    --overwrite_output_dir

# T5-small on PISA
USE_TORCH=false python3 run_summarization_flax.py \
    --output_dir models \
    --tpu_num_cores 8 \
	--model_name_or_path t5-small \
    --source_prefix "summarize: " \
	--train_file /home/qj213/seq2seq_all/with_state/train.json \
    --validation_file /home/qj213/seq2seq_all/with_state/val.json \
    --test_file /home/qj213/seq2seq_all/with_state/test.json \
    --text_column source --summary_column target \
    --num_train_epochs 100 \
    --do_train --do_eval --do_predict --predict_with_generate \
    --learning_rate 5e-4 --warmup_steps 500 \
    --per_device_train_batch_size 64 \
	--per_device_eval_batch_size 64 \
    --max_source_length 512 --max_target_length 256 \
    --overwrite_output_dir \
    --wandb_run_name benchmarking-models/t5-small

# T5-base on PISA
USE_TORCH=false python3 run_summarization_flax.py \
    --output_dir models \
    --tpu_num_cores 8 \
	--model_name_or_path t5-base \
    --source_prefix "summarize: " \
	--train_file /home/qj213/seq2seq_all/with_state/train.json \
    --validation_file /home/qj213/seq2seq_all/with_state/val.json \
    --test_file /home/qj213/seq2seq_all/with_state/test.json \
    --text_column source --summary_column target \
    --num_train_epochs 20 \
    --do_train --do_eval --do_predict --predict_with_generate \
    --learning_rate 5e-4 --warmup_steps 500 \
    --per_device_train_batch_size 32 \
	--per_device_eval_batch_size 32 \
    --max_source_length 512 --max_target_length 256 \
    --overwrite_output_dir \
    --wandb_run_name benchmarking-models/t5-base

# T5-base from scratch on PISA
USE_TORCH=false python3 run_summarization_flax.py \
    --output_dir models \
    --tpu_num_cores 8 \
	--model_type t5 --tokenizer_name /home/qj213/customised_tokenizer_t5_vocab_size1000 --config_name /home/qj213/t5-base \
    --source_prefix "summarize: " \
	--train_file /home/qj213/seq2seq_all/with_state/train.json \
    --validation_file /home/qj213/seq2seq_all/with_state/val.json \
    --test_file /home/qj213/seq2seq_all/with_state/test.json \
    --text_column source --summary_column target \
    --num_train_epochs 20 \
    --do_train --do_eval --do_predict --predict_with_generate \
    --learning_rate 5e-4 --warmup_steps 500 \
    --per_device_train_batch_size 32 \
	--per_device_eval_batch_size 32 \
    --max_source_length 512 --max_target_length 256 \
    --overwrite_output_dir \
    --wandb_run_name benchmarking-models/t5-base-customised-token

# T5-v1.1-large on PISA
USE_TORCH=false python3 run_summarization_flax.py \
    --output_dir models \
    --tpu_num_cores 8 \
    --model_name_or_path google/t5-v1_1-large \
    --source_prefix "summarize: " \
    --train_file /home/qj213/seq2seq_all/with_state/train.json \
    --validation_file /home/qj213/seq2seq_all/with_state/val.json \
    --test_file /home/qj213/seq2seq_all/with_state/test.json \
    --text_column source --summary_column target \
    --num_train_epochs 20 \
    --do_train --do_eval --do_predict --predict_with_generate \
    --learning_rate 5e-4 --warmup_steps 500 \
    --per_device_train_batch_size 4 --per_device_eval_batch_size 4 \
    --max_source_length 128 --max_target_length 128 \
    --overwrite_output_dir \
    --wandb_run_name benchmarking-models/google/t5-v1_1-large

# T5-v1.1-large on PISA with last 1 step
USE_TORCH=false python3 run_summarization_flax.py \
    --output_dir models \
    --tpu_num_cores 8 \
    --model_name_or_path google/t5-v1_1-large \
    --source_prefix "summarize: " \
    --train_file /home/qj213/seq2seq_all/with_proof_and_state/last_1_step_train.json \
    --validation_file /home/qj213/seq2seq_all/with_proof_and_state/last_1_step_val.json \
    --test_file /home/qj213/seq2seq_all/with_proof_and_state/last_1_step_test.json \
    --text_column source --summary_column target \
    --num_train_epochs 20 \
    --do_train --do_eval --do_predict --predict_with_generate \
    --learning_rate 5e-4 --warmup_steps 500 \
    --per_device_train_batch_size 8 --per_device_eval_batch_size 8 \
    --max_source_length 128 --max_target_length 64 \
    --overwrite_output_dir \
    --wandb_run_name benchmarking-models/google/t5-v1_1-large-last-1-step

# GPT-2 on PISA
USE_TORCH=false python3 run_clm_flax.py \
    --output_dir models \
    --tpu_num_cores 8 \
	--model_name_or_path gpt2 \
	--train_file /home/qj213/seq2seq_all/with_state/train.json \
    --validation_file /home/qj213/seq2seq_all/with_state/val.json \
    --test_file /home/qj213/seq2seq_all/with_state/test.json \
    --text_column source --summary_column target \
    --num_train_epochs 20 \
    --do_train --do_eval --do_predict --predict_with_generate \
    --learning_rate 5e-4 --warmup_steps 500 \
    --per_device_train_batch_size 64 \
	--per_device_eval_batch_size 64 \
    --max_source_length 512 --max_target_length 256 \
    --overwrite_output_dir \
    --wandb_run_name benchmarking-models/gpt2

USE_TORCH=false python3 run_summarization_flax.py \
    --output_dir models \
    --tpu_num_cores 8 \
	--model_name_or_path facebook/bart-base \
    --tokenizer_name facebook/bart-base \
	--train_file /home/qj213/seq2seq_all/with_state/train.json \
    --validation_file /home/qj213/seq2seq_all/with_state/val.json \
    --test_file /home/qj213/seq2seq_all/with_state/test.json \
    --text_column source --summary_column target \
    --num_train_epochs 20 \
    --do_train --do_eval --do_predict --predict_with_generate \
    --learning_rate 5e-5 --warmup_steps 0 \
    --per_device_train_batch_size 64 \
	--per_device_eval_batch_size 64 \
    --max_source_length 512 --max_target_length 256 \
    --overwrite_output_dir \
    --wandb_run_name benchmarking-models/facebook/bart-base

USE_TORCH=false python3 run_summarization_flax.py --output_dir /home/qj213/INT/data/hf_model --tpu_num_cores 8 --model_type t5 --tokenizer_name /home/qj213/t5-small --config_name /home/qj213/t5-small --source_prefix "summarize: " --train_file /home/qj213/INT/data/processed_int/train.json --validation_file /home/qj213/INT/data/processed_int/valid.json --test_file /home/qj213/INT/data/processed_int/test.json --text_column source --summary_column target --num_train_epochs 20 --do_train --do_eval --do_predict --predict_with_generate     --learning_rate 1e-4 --warmup_steps 300     --per_device_train_batch_size 8 --per_device_eval_batch_size 8     --max_source_length 256 --max_target_length 256 --overwrite_output_dir --wandb_run_name int/reproduction_scratch_t5_small

USE_TORCH=false python3 run_summarization_flax.py --output_dir /home/qj213/INT/data/hf_model --tpu_num_cores 8 --model_name_or_path t5-small --source_prefix "summarize: " --train_file /home/qj213/INT/data/processed_int/train.json --validation_file /home/qj213/INT/data/processed_int/valid.json --test_file /home/qj213/INT/data/processed_int/test.json --text_column source --summary_column target --num_train_epochs 40 --do_train --do_eval --do_predict --predict_with_generate     --learning_rate 1e-4 --warmup_steps 300     --per_device_train_batch_size 8 --per_device_eval_batch_size 8     --max_source_length 256 --max_target_length 256 --overwrite_output_dir