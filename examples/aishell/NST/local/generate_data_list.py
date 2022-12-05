import argparse
import os


def get_args():
    parser = argparse.ArgumentParser(description='generate data.list file ')
    parser.add_argument('--tar_dir', help='path for tar file')
    parser.add_argument('--out_data_list', help='output path for data list')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    target_dir = args.tar_dir
    # target_dir = "data/train/wenet_bad_tar_10_7_nst3_r6"
    data_list = os.listdir(target_dir)
    output_file = args.out_data_list
    # output_file = "data/train/wenet_bad_tar_10_7_nst3_r6.list"
    with open(output_file, "w") as writer:
        for line in data_list:
            writer.write(target_dir + "/" + line + "\n")


if __name__ == '__main__':
    main()
