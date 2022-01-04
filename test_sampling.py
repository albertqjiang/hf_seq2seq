from transformers import FlaxT5ForConditionalGeneration, T5TokenizerFast
from jax import jit

import jax.numpy as jnp

config_path = "aqj213/t5-small-pisa-state-only-finetuned"

if __name__ == "__main__":
    model = FlaxT5ForConditionalGeneration.from_pretrained(config_path)
    tokenizer = T5TokenizerFast.from_pretrained(config_path)
    input_ids = tokenizer("summarize: proof (prove) goal: No subgoals!", return_tensors='jax', padding="max_length", truncation=True).input_ids
    non_zero = jnp.count_nonzero(input_ids)
    attention_mask = jnp.zeros_like(input_ids)
    attention_mask = jnp.place(attention_mask, attention_mask<non_zero, [1.])
    attention_mask = jnp.expand_dims(attention_mask, axis=0)
    
    input_ids = jnp.repeat(input_ids, 8, axis=0)
    attention_mask = jnp.repeat(attention_mask, 8, axis=0)
    print(input_ids.shape)
    print(attention_mask.shape)

    def sample(input_ids, attention_mask):
        return model.generate(input_ids, attention_mask=attention_mask, do_sample=True)

    fast_gen = jit(sample)
    summary_ids = fast_gen(input_ids, attention_mask).sequences
    for summary_id in summary_ids:
        print(tokenizer.decode(summary_id, skip_special_tokens=True, clean_up_tokenization_spaces=False))
