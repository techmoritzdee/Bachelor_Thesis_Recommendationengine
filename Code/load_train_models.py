import torch
from train import MODEL_SAVE_PATH

torch.set_printoptions(sci_mode=False)

model = torch.load(MODEL_SAVE_PATH)
print("Model parameters for testing:")
print('Betas:', model.linear.weight.data)
print('Bias:', model.linear.bias.data)
