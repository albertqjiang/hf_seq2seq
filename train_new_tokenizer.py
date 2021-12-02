from transformers import AutoTokenizer
from datasets import load_dataset
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a custom tokenizer")
    parser.add_argument("--train-file", "-tf", type=str, help="The path to the training file")
    parser.add_argument("--valid-file", "-vf", type=str, help="The path to the validation file")
    parser.add_argument("--test-file", "-tef", type=str, help="The path to the test file")
    parser.add_argument("--vocab-size", "-vs", type=int, help="The vocabulary size to use.")
    parser.add_argument("--dump-path", "-dp", type=str, help="Where to save the customised tokenizer",
                        default="/home/qj213")
    data_args = parser.parse_args()

    data_files = {}
    data_files["train"] = data_args.train_file
    data_files["validation"] = data_args.valid_file
    data_files["test"] = data_args.test_file
    extension = data_args.train_file.split(".")[-1]
    dataset = load_dataset(extension, data_files=data_files)   

    batch_size = 1000
    def batch_iterator():
        for i in range(0, len(dataset["train"]), batch_size):
            batch = dataset["train"][i : i + batch_size]
            yield [source + target for source, target in zip(batch["source"], batch["target"])]

    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    new_tokenizer = tokenizer.train_new_from_iterator(batch_iterator(), vocab_size=data_args.vocab_size)
    new_tokenizer.save_pretrained(os.path.join(data_args.dump_path, f"customised_tokenizer_t5_vocab_size{data_args.vocab_size}"))
