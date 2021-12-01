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
    --per_device_train_batch_size 1 \
	--per_device_eval_batch_size 1 \
    --max_source_length 512 --max_target_length 256 \
    --overwrite_output_dir \
    --wandb_run_name benchmarking-models/google/t5-v1_1-large


USE_TORCH=false python3 run_summarization_flax.py --output_dir /home/qj213/INT/data/hf_model --tpu_num_cores 8 --model_type t5 --tokenizer_name /home/qj213/t5-small --config_name /home/qj213/t5-small --source_prefix "summarize: " --train_file /home/qj213/INT/data/processed_int/train.json --validation_file /home/qj213/INT/data/processed_int/valid.json --test_file /home/qj213/INT/data/processed_int/test.json --text_column source --summary_column target --num_train_epochs 20 --do_train --do_eval --do_predict --predict_with_generate     --learning_rate 1e-4 --warmup_steps 300     --per_device_train_batch_size 8 --per_device_eval_batch_size 8     --max_source_length 256 --max_target_length 256 --overwrite_output_dir --wandb_run_name int/reproduction_scratch_t5_small

USE_TORCH=false python3 run_summarization_flax.py --output_dir /home/qj213/INT/data/hf_model --tpu_num_cores 8 --model_name_or_path t5-small --source_prefix "summarize: " --train_file /home/qj213/INT/data/processed_int/train.json --validation_file /home/qj213/INT/data/processed_int/valid.json --test_file /home/qj213/INT/data/processed_int/test.json --text_column source --summary_column target --num_train_epochs 40 --do_train --do_eval --do_predict --predict_with_generate     --learning_rate 1e-4 --warmup_steps 300     --per_device_train_batch_size 8 --per_device_eval_batch_size 8     --max_source_length 256 --max_target_length 256 --overwrite_output_dir