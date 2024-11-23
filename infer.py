from torch.jit import load
import torch
import torch
import torch.nn
import torch.optim as optim
import segmentation_models_pytorch as smp


learning_rate = 0.0001
pretrained_path = "model.pth"


model = smp.UNet()
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

checkpoint = torch.load(pretrained_path)
optimizer.load_state_dict(checkpoint['optimizer'])

from collections import OrderedDict
new_state_dict = OrderedDict()

for k, v in checkpoint['model'].items():
    name = k[7:]
    new_state_dict[name] = v

# load params
model.load_state_dict(new_state_dict)

