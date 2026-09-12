import argparse

from evm.magnifier.magnifier import Magnifier
from evm.utils.misc import load_config


def main(args):
    cfg = load_config(args.cfg)
    video_path = args.video_path
    output_path = args.output_path
    magnifier = Magnifier(cfg)
    magnifier.magnify(video_path, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", type=str, default="evm/config/evm_config.yaml")
    parser.add_argument("video_path", type=str, help="input_video_path")
    parser.add_argument("output_path", type=str, help="output_video_path")
    args = parser.parse_args()
    main(args)
