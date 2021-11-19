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
    --num_train_epochs 10 \
    --do_train --do_eval --do_predict --predict_with_generate \
    --learning_rate 5e-5 --warmup_steps 300 \
    --per_device_train_batch_size 8 \
	--per_device_eval_batch_size 8 \
    --max_source_length 256 --max_target_length 32 \
    --overwrite_output_dir


USE_TORCH=false python3 run_summarization_flax.py --output_dir /home/qj213/INT/data/hf_model --tpu_num_cores 8 --model_type encoder-decoder --source_prefix "summarize: " --train_file /home/qj213/INT/data/int_repro_processed/train.json --validation_file /home/qj213/INT/data/int_repro_processed/valid.json --text_column source --summary_column target --num_train_epochs 100 --do_train --do_eval --do_predict --predict_with_generate     --learning_rate 1e-4 --warmup_steps 300     --per_device_train_batch_size 8 --per_device_eval_batch_size 8     --max_source_length 256 --max_target_length 256