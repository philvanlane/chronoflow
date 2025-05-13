import numpy as np
import zuko
import torch

def loadModel(weights_file):
    """
    Load the model from the specified path.
    """
    model = zuko.flows.NSF(1, 3, transforms=3)
    weights = torch.load(weights_file)
    model.load_state_dict(weights)
    return model