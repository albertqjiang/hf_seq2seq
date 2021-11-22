from transformers import FlaxT5ForConditionalGeneration, T5TokenizerFast
from tqdm import tqdm

import argparse
import json
import os


def translate(model, tokenizer, source):
    input = tokenizer(["summarize " + source], return_tensors='np').input_ids
    summary_ids = model.generate(input).sequences
    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generation for T5 models")
    parser.add_argument('--model-path', type=str, help="path to the model as well as the tokenizer")
    parser.add_argument('--eval-path', type=str, help="path to the file to evaluate")
    parser.add_argument('--dump-path', type=str, help="path to dump the output")
    args = parser.parse_args()
    
    file_name = args.eval_path.split("/")[-1]
    model = FlaxT5ForConditionalGeneration.from_pretrained(args.model_path)
    tokenizer = T5TokenizerFast.from_pretrained(args.model_path)
    with open(os.path.join(args.dump_path, "generate_"+file_name+".src"), "w") as dump_s, \
            open(os.path.join(args.dump_path, "generate_"+file_name+".tgt"), "w") as dump_t:
        for line in tqdm(open(args.eval_path).readlines()):
            line_json = json.loads(line.strip())
            source = line_json["source"]
            target = translate(model, tokenizer, source)
        
            dump_s.write(source)
            dump_s.write("\n")
            dump_t.write(target)
            dump_t.write("\n")
