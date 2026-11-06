"""Cover dataset and data loader that samples an embedding rate (alpha) per image.

Author: Martin Benes
Affiliation: University of Innsbruck
"""


import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as transforms
from typing import Union, Dict, Any, Callable, Optional, Tuple
import warnings

pd.options.mode.chained_assignment = None  # suppress Pandas warning


class CoverDataset(torch.utils.data.Dataset):
    """Cover images listed in a split file; items are (image, alpha, name)."""
    def __init__(
        self,
        data_dir: Union[Path, str],
        config_name: str,
        cover_dirname: str = 'images',
        shuffle_seed: int = None,
        transform: Optional[Callable] = None,
    ):
        # parameters
        self.data_dir = Path(data_dir)
        self.cover_dir = self.data_dir / cover_dirname
        self.transform = transform

        # get selected dataset
        self.config_cover = pd.read_csv(self.data_dir / config_name, low_memory=False)
        assert len(self.config_cover) > 0, 'no such covers found'

        to_remove = []
        for i, row in self.config_cover.iterrows():
            if not (self.cover_dir / row['name']).exists():
                warnings.warn(f'{row['name']} not found, ignoring')
                to_remove.append(i)
        self.config_cover = self.config_cover.drop(to_remove)

        # for dataset reshuffling
        self._rng = np.random.default_rng(shuffle_seed)
        self.reshuffle()

        # check dataset
        assert len(self.config_cover) > 0, 'no such covers found, did you forget running prepare_dataset.py?'

    def __len__(self) -> int:
        """Dataset length."""
        return len(self.config_cover)

    def state_dict(self):
        return {'rng': self._rng_prev_state}
    def load_state_dict(self, state):
        self._rng.bit_generator.state = state['rng']
    def reshuffle(self):
        """Call after each epoch to reshuffle."""
        self._rng_prev_state = self._rng.bit_generator.state  # last rng state
        # shuffle cover
        shuffle_seed = self._rng.integers(2**32-1)
        self.config_cover = self.config_cover.sort_values('name')
        self._reset_indices(drop=True)
        self.config_cover = self.config_cover.sample(
            frac=1,
            random_state=shuffle_seed,
        )
        self._reset_indices(drop=True)

    def _reset_indices(self, drop=True):
        """Reset the index of the cover dataset."""
        self.config_cover = self.config_cover.reset_index(drop=drop)

    def __getitem__(self, idx: int):
        """Index dataset."""
        # find cover
        image_row = self.config_cover.iloc[idx, :]
        image_name = image_row['name']

        # load image
        image = np.array(Image.open(self.cover_dir / image_name))

        # transform
        if self.transform is not None:
            image = self.transform(image)

        # retrieve image+target pairs
        return (*image, image_name)


class UniformAlpha(torch.nn.Module):
    """Transform attaching an embedding rate alpha, fixed or drawn uniformly from [low, high]."""
    def __init__(self, value: float = None, low: float = 0, high: float = 1):
        super().__init__()
        self.value = value
        self.low = low
        self.high = high

    def forward(self, x) -> Tuple[torch.Tensor]:
        C, H, W = x.size()
        if self.value is not None:
            alpha = torch.ones((1,), device=x.device) * self.value
        else:
            alpha = torch.empty((1,), device=x.device).uniform_(self.low, self.high)
        return x, alpha

    def __repr__(self):
        if self.value is not None:
            return f'FixedAlpha({self.value})'
        else:
            return f'UniformAlpha({self.low}, {self.high})'


def get_data_loader(
    config_name: Path,
    args: Dict[str, Any],
    take_num_images: int = None,
    drop_last: bool = False,
    **kw,
):
    """Data loader over the covers of a split, with alpha attached to each image."""
    # Dataset transform
    transform = [transforms.ToTensor(),]
    alpha_range = args.get('alpha_range', {})
    transform += [UniformAlpha(value=args.get('alpha', None), **alpha_range)]
    transform = transforms.Compose(transform)

    print('Data transform')
    print(transform.transforms)

    # Dataset
    dataset = CoverDataset(
        # dataset
        data_dir=args['data_dir'],
        config_name=config_name,
        # training parameters
        cover_dirname=args.get('cover_dirname', 'images'),
        shuffle_seed=args.get('shuffle_seed', None),  # for reshuffling every epoch
        # other
        transform=transform,
    )

    # Create data loaders
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, range(take_num_images)) if take_num_images is not None else dataset,
        batch_size=args["batch_size"],
        shuffle=False,
        num_workers=args.get('num_workers', 0),
        pin_memory=True,
        drop_last=drop_last,
        **kw,
    )

    #
    return loader, dataset
