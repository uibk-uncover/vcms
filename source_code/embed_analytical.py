"""Simulates HILL embedding at 0.4 bits per element into an example cover image.

Adapted from https://github.com/uibk-uncover/conseal

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import conseal as cl
import numpy as np
from pathlib import Path
from PIL import Image

# load cover
cover_path = Path('../example_images/HILL/31145_cover296.png')
x0 = np.array(Image.open(cover_path))

# embed HILL 0.4 bpe
rho_p1, rho_m1 = cl.color.map_channels(cl.hill.compute_cost_adjusted, x0, stack_axis=1)  # each [3 H W]
delta = cl.color.concatenate.simulate(
    rhos=(list(rho_p1), list(rho_m1)),  # one list of channels per change
    alpha=0.4,  # bpe, over all channels
    seed=12345,
    stack_axis=-1,  # [H W 3]
)
x1 = (x0.astype('int16') + delta).astype('uint8')  # adjusted costs keep x1 within [0, 255]

print(f'Changed {(x0 != x1).mean()*100:.02f}% of elements.')

stego_path = './31145_stego296_hill_0.4.png'  # replace with filename
Image.fromarray(x1).save(stego_path)
print(f'The stego object saved as "{stego_path}".')  #
