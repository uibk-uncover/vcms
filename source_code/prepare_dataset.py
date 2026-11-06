"""Downloads the ALASKA v2 color TIFF images listed in the splits and saves their top-left 296x296 crops as PNG.

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import io
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

#
input_url = 'https://alaska.utt.fr/DATASETS/ALASKA_v2_TIFF_512_COLOR'
out_dir = Path('../data/images')
out_dir.mkdir(exist_ok=True)

# retry transient failures (connection errors, 429, 5xx) with backoff
session = requests.Session()
session.mount('https://', HTTPAdapter(max_retries=Retry(
    total=5,
    backoff_factor=1.,
    status_forcelist=(429, 500, 502, 503, 504),
)))

#
df = pd.concat([
    pd.read_csv('../data/split_tr.csv'),
    pd.read_csv('../data/split_va.csv'),
    pd.read_csv('../data/split_te.csv'),
]).sort_values('name').reset_index(drop=True)

#
for fname in tqdm(df['name']):
    # skip images prepared in a previous (interrupted) run
    if (out_dir / fname).exists():
        continue

    res = session.get(f'{input_url}/{Path(fname).stem}.tif', timeout=60)
    res.raise_for_status()
    # decode TIFF
    x0 = np.array(Image.open(io.BytesIO(res.content), formats=['TIFF']))

    # top-left crop to 296x296
    x0 = x0[:296, :296]
    Image.fromarray(x0).save(out_dir / f'{fname}.tmp', format='PNG')
    (out_dir / f'{fname}.tmp').rename(out_dir / fname)  # no half-written files on interrupt
