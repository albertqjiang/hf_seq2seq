import argparse
import json
import threading
import time
from queue import Queue, Empty
from copy import deepcopy

import jax
import numpy as np
import optax

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
    # parser.add_argument("--temp", type=float, default=1.0, help="The temperature to use for generation")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    threading.Thread(target=app.run, kwargs={"port": 5000, "host": "0.0.0.0"}).start()

    args = parse_args()
    config_path = args.config_path
    max_length = args.max_length
    # temp = args.temp
    model = FlaxT5ForConditionalGeneration.from_pretrained(config_path)
    tokenizer = T5TokenizerFast.from_pretrained(config_path)

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
        input_ids = tokenizer(["summarize: " + context] * single_generation_batch, return_tensors='np').input_ids
        for i in range(n // single_generation_batch):
            all_tokenized = []

            outputs = model.generate(input_ids, max_length=max_length, num_beams=single_generation_batch)
            output_ids = outputs.sequences
            output_scores = outputs.scores.tolist()
            output_strings = tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            
            for o_string, o_score in zip(output_strings, output_scores):
                sequences.append(tokenizer.convert_ids_to_tokens(o_string))
                log_probs_for_sequences.append(o_score)

        for o, q, slp in zip(sequences, all_q, log_probs_for_sequences):
            q.put((o, slp))
            # q.put((tokenizer.decode(o), slp.tolist()))

        print(f"completion done in {time.time() - start:06}s")
