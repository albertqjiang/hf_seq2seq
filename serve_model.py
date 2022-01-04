import argparse
import json
import threading
import time
from queue import Queue, Empty

import jax
from jax import jit
import numpy as np

from transformers import FlaxT5ForConditionalGeneration, T5TokenizerFast

from flask import Flask, request, make_response, jsonify
app = Flask(__name__)

requests_queue = Queue()

"""
curl --header "Content-Type: application/json" \
  --request POST \
  --data '{"context":"eleutherai", "n": 8}' \
  http://localhost:5000/complete
"""


def _build_cors_prelight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add('Access-Control-Allow-Headers', "*")
    response.headers.add('Access-Control-Allow-Methods', "*")
    return response


def _corsify_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


@app.route('/complete', methods=['POST', 'OPTIONS'])
def complete():
    if request.method == "OPTIONS":  # CORS preflight
        return _build_cors_prelight_response()
    elif request.method == "POST":  # The actual request following the preflight
        content = request.json

        if requests_queue.qsize() > 100:
            return {"error": "queue full, try again later"}

        response_queue = Queue()

        requests_queue.put(({
                                "context": content["context"],
                                "n": int(content["n"])
                            }, response_queue))

        completions = [response_queue.get()]
        while not response_queue.empty():
            completions.append(response_queue.get())

        return _corsify_actual_response(jsonify({"completion": completions}))
    else:
        raise RuntimeError("Weird - don't know how to handle method {}".format(request.method))


def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, default=None, help="Config file location")
    parser.add_argument("--max-length", type=int, default=64, help="Maximum length of generation")
    parser.add_argument("--single-generation-batch", type=int, default=8, help="How many candidates to generate at one time")
    args = parser.parse_args()
    return args

def tokenize(tokenizer, context, n):
    input_ids = tokenizer("summarize: " + context, return_tensors='np', padding="max_length", truncation=True).input_ids
    non_zero = np.count_nonzero(input_ids)
    attention_mask = np.zeros_like(input_ids)
    np.place(attention_mask, np.arange(attention_mask.shape[1])<non_zero, [1.])
    input_ids = np.repeat(input_ids, n, axis=0)
    attention_mask = np.repeat(attention_mask, n, axis=0)
    return input_ids, attention_mask


if __name__ == "__main__":
    threading.Thread(target=app.run, kwargs={"port": 5000, "host": "0.0.0.0"}).start()

    args = parse_args()
    config_path = args.config_path
    max_length = args.max_length
    single_generation_batch = args.single_generation_batch
    model = FlaxT5ForConditionalGeneration.from_pretrained(config_path)
    tokenizer = T5TokenizerFast.from_pretrained(config_path)

    def sample(input_ids, attention_mask):
        return model.generate(input_ids, attention_mask=attention_mask, do_sample=True)
    fast_generate = jit(sample)
    
    tokenizer = T5TokenizerFast.from_pretrained(config_path)
    # Compile the funciton
    start = time.time()
    print("Compiling generation function")
    context = "Hello"
    input_ids, attention_mask = tokenize(tokenizer=tokenizer, context=context, n=single_generation_batch)
    fast_generate(input_ids, attention_mask)
    print(f"Generation compilation done, it took {time.time()-start:.06}s")

    start = time.time()
    print(f"jax devices: {jax.device_count()}")
    print(f"jax runtime initialized in {time.time() - start:.06}s")

    while True:
        all_ctx = []
        all_q = []
        try:
            o, q = requests_queue.get(block=False)
            n = o["n"]
            context = o["context"]

            all_q = n * [q]
        except Empty:
            if len(all_ctx):
                break
            else:
                time.sleep(0.1)

        if not all_q:
            continue

        start = time.time()
        sequences = []
        log_probs_for_sequences = []
        single_generation_batch = 8 if n > 8 else n
        input_ids, attention_mask = tokenize(tokenizer=tokenizer, context=context, n=single_generation_batch)
        for i in range(n // single_generation_batch):
            all_tokenized = []

            outputs = fast_generate(input_ids, attention_mask)
            output_ids = outputs.sequences
            output_scores = outputs.scores.squeeze().tolist()
            output_strings = tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            
            for o_string, o_score in zip(output_strings, output_scores):
                sequences.append(o_string)
                log_probs_for_sequences.append(o_score)

        for o, q, slp in zip(sequences, all_q, log_probs_for_sequences):
            q.put(("".join(o).strip(), slp))
            # q.put((tokenizer.decode(o), slp.tolist()))

        print(f"completion done in {time.time() - start:06}s")
