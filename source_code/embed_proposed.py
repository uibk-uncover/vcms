"""Simulates embedding with the trained proposed model at 0.4 bits per element into an example cover image.

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image
import torch

import _models
import _stego

#
device_name = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
device = torch.device(device_name)
_stego.initialize(device)


# load model
model_path = Path('../models/proposed')
args = json.load(open(model_path / 'config.json'))
net_a = _models.DepthwiseSeparableBlock(**args['net_a']).to(device)
checkpoint = torch.load(model_path / 'model' / 'model_step_24000.pt.tar', map_location=torch.device(device_name), weights_only=True)
net_a.load_state_dict(checkpoint['net_a']['state_dict'])
net_a.eval()
gan = _stego.StegoGAN27(net_a=net_a, pad=16)

# load cover image
cover_path = Path('../example_images/proposed/31145_cover296.png')
x0 = np.array(Image.open(cover_path))
x0_ = torch.from_numpy(x0.astype('float32').transpose(2, 0, 1) / 255.)[None].to(device)
print(f'Loaded cover image from "{cover_path}".')

# embed
res_a = gan(
    x0=x0_,
    alpha=.4,
    simulator_method='hard',
    stego_seed=12345,
    mode='pass',
)
x1 = torch.clamp(torch.round(res_a.x1 * 255.), 0, 255.)
x1 = x1.cpu().numpy().astype('uint8')[0].transpose(1, 2, 0)
print(f'Changed {(x0 != x1).mean()*100:.02f}% of elements.')

stego_path = './31145_stego296_0.4.png'  # replace with filename
Image.fromarray(x1).save(stego_path)
print(f'The stego object saved as "{stego_path}".')  #