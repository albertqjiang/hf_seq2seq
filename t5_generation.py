from transformers import FlaxT5ForConditionalGeneration, T5TokenizerFast
from tqdm import tqdm

import argparse
import json
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generation for T5 models")
    parser.add_argument('--model-path', type=str, help="path to the model as well as the tokenizer")
    parser.add_argument('--eval-path', type=str, help="path to the file to evaluate")
    parser.add_argument('--dump-path', type=str, help="path to dump the output")
    args = parser.parse_args()

    model = FlaxT5ForConditionalGeneration.pre_pretrained(args.model_path)
    tokenizer = T5TokenizerFast.from_pretrained(args.model_path)
    with open(os.path.join(args.dump_path, "test.src")) as dump_s, open(os.path.join(args.dump_path, "test.tgt")) as dump_t:
        for line in tqdm(open(args.eval_path).readlines()):
            line_json = json.loads(line.strip())
            source = line_json["source"]
            input = tokenizer(["summarize " + source], return_tensors='np').input_ids
            summary_ids = model.generate(input).sequences
            output = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
            dump_s.write(source)
            dump_s.write("\n")
            dump_t.write(output)
            dump_t.write("\n")
