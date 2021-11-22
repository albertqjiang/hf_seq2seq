import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Split some files into eight parts for generation")
    parser.add_argument("--file-path", type=str, help="The path to the file")
    parser.add_argument("--output-path", type=str, help="Where to dump the resulting files")
    args = parser.parse_args()

    file_name = args.file_path.split("/")[-1]
    lines = open(args.file_path).readlines()
    number_of_lines = len(lines)
    one_eighth = int(number_of_lines/8)

    for i in range(8):
        part_lines = lines[i*one_eighth: (i+1)*one_eighth]
        with open(os.path.join(args.output_path, "part_{}_".format(i+1) + file_name), "w") as fout:
            fout.writelines(part_lines)
