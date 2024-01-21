"""
Script for testing a trained SqueezeNet model
"""

import torch
import argparse
from src.squeezenet import SqueezeNet

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", '-i', help="Path to an input image")
    parser.add_argument("--model", '-m', nargs='?', type=str, default="./pretrained/model.pt", help="Path to a SqueezeNet model (*.pt)")

    args = parser.parse_args()

    if not args.image:
        print("ERROR: Please provide an input image with the -i option")
        raise SystemExit(1)

    if not args.model:
        print("ERROR: Please provide an input model file with the -m option")
        raise SystemExit(1)

    clf = SqueezeNet()

    clf.load_custom_model(args.model)
    clf.test(args.image)
