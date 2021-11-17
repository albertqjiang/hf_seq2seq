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

