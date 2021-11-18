import json
import random
random.seed(0)

if __name__ == "__main__":
    for partition in ["train", "val", "test"]:
        src_path = "seq2seq_all/with_state/{}.src".format(partition)
        tgt_path = "seq2seq_all/with_state/{}.tgt".format(partition)

        src_lines = open(src_path).readlines()
        tgt_lines = open(tgt_path).readlines()

        with open("seq2seq_all/with_state/{}.json".format(partition), "w") as fout:
            for src_line, tgt_line in zip(src_lines, tgt_lines):
                fout.write(json.dumps(
                    {
                        "source": src_line.strip().replace("State: ", ""),
                        "target": tgt_line.strip()
                    }
                ))
        