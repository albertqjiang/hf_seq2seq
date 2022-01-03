from transformers import FlaxT5ForConditionalGeneration, T5TokenizerFast
from jax import jit




if __name__ == "__main__":
    model = FlaxT5ForConditionalGeneration.from_pretrained("t5-small")
    tokenizer = T5TokenizerFast.from_pretrained("t5-small")
    input_ids = tokenizer("summarize: hello", return_tensors='jax', padding="max_length", truncation=True).input_ids

    def sample(input_ids):
        return model.generate(input_ids, do_sample=True)

    fast_gen = jit(sample)
    summary_ids = fast_gen(input_ids).sequences
    print(tokenizer.decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False))
