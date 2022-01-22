import os
import argparse


def parse_args():
    """
    args for converting.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str, help='training script name')
    parser.add_argument('--config', type=str, default='baseline', help='yaml configure file name')
    parser.add_argument('--save_dir', type=str, help='root directory to save checkpoints, logs, and tensorboard')
    parser.add_argument('--mode', type=str, choices=["single", "multiple"], default="multiple",
                        help="train on single gpu or multiple gpus")
    parser.add_argument('--nproc_per_node', type=int, help="number of GPUs per node")  # specify when mode is multiple
    parser.add_argument('--use_lmdb', type=int, choices=[0, 1], default=0)  # whether datasets are in lmdb format
    parser.add_argument('--script_prv', type=str, help='training script name')
    parser.add_argument('--config_prv', type=str, default='baseline', help='yaml configure file name')

    parser.add_argument('--out_dir', type=str, help='root directory to save converted model')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    train_cmd = "python lib/train/convert.py --script %s --config %s --save_dir %s --use_lmdb %d " \
                "--script_prv %s --config_prv %s --out_dir %s" \
                % (args.script, args.config, args.save_dir, args.use_lmdb, args.script_prv, args.config_prv,
                   args.out_dir)
    os.system(train_cmd)


if __name__ == "__main__":
    main()
