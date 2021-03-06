from transformers import FlaxT5ForConditionalGeneration, T5TokenizerFast
from jax import jit

import jax.numpy as jnp
import numpy as np

config_path = "aqj213/t5-small-pisa-state-only-finetuned"

if __name__ == "__main__":
    model = FlaxT5ForConditionalGeneration.from_pretrained(config_path)
    tokenizer = T5TokenizerFast.from_pretrained(config_path)
    input_ids = tokenizer("summarize: proof (prove) goal: No subgoals!", return_tensors='np', padding="max_length", truncation=True).input_ids
    non_zero = np.count_nonzero(input_ids)
    attention_mask = np.zeros_like(input_ids)
    np.place(attention_mask, np.arange(attention_mask.shape[1])<non_zero, [1.])
    
    input_ids = np.repeat(input_ids, 8, axis=0)
    attention_mask = np.repeat(attention_mask, 8, axis=0)
    # print(input_ids.shape)
    # print(non_zero)
    # print(attention_mask.shape)
    # print(np.count_nonzero(attention_mask))

    def sample(input_ids, attention_mask):
        return model.generate(input_ids, attention_mask=attention_mask, do_sample=True)

    fast_gen = jit(sample)
    summary = fast_gen(input_ids, attention_mask)
    summary_ids = summary.sequences
    candidate_scores = summary.scores
    for summary_id, score in zip(summary_ids, candidate_scores):
        print(tokenizer.decode(summary_id, skip_special_tokens=True, clean_up_tokenization_spaces=False), f" | Score: {score[0]}")
