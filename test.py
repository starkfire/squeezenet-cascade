"""
Script for testing a trained SqueezeNet model
"""

import torch
from src.squeezenet import SqueezeNet

if __name__ == "__main__":
    clf = SqueezeNet()

    clf.load_custom_model("./pretrained/model.pt")
    
    clf.test("./cardinal.jpg")
    clf.test("./cockatiel.jpg")
