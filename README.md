# Supplementary material

This supplementary material accompanies the NeurIPS 2026 paper 'Variational Cover Modification Steganography'.

The repository is structured as follows:

| Folder               | Contents                                                                     |
|----------------------|------------------------------------------------------------------------------|
| data                 | Training, validation, and test split.                                        |
| detector_predictions | Logits produced by our trained detectors.                                    |
| example_images       | Example images from all methods in the study.                                |
| models               | Weights, logs, and metadata from the training of the proposed model.         |
| source_code          | Source code for generating the dataset, training, embedding, and evaluation. |


The source code lets you reproduce the following steps:


| Script                                 | Behavior                                                                         |
|----------------------------------------|----------------------------------------------------------------------------------|
| detector_predictions/calculate_roc.py  | Recalculates the ROC AUC from all the detector logits in the folder.             |
| source_code/embed_analytical.py        | Simulates steganographic embedding into a cover image using the HILL method.     |
| source_code/embed_proposed.py          | Simulates steganographic embedding into a cover image using the proposed method. |
| source_code/prepare_dataset.py         | Downloads and prepares the ALASKA dataset.                                       |
| source_code/train_proposed.py          | Performs the adversarial training. (computationally expensive)                   |


## Setup

The code requires Python 3.12 or newer. Install the dependencies with

```bash
pip install -r requirements.txt
```

All scripts use paths relative to their own folder, so run each script from inside the folder that contains it, e.g.,

```bash
cd source_code
python embed_proposed.py
```

`embed_proposed.py` uses the checkpoint `models/proposed/model/model_step_24000.pt.tar`.

`calculate_roc.py` approximates the ROC AUC by trapezoidal integration over 100 quantile thresholds of the detector scores.


## Dataset

We carried out our experiments on the ALASKA dataset, which is available at https://alaska.utt.fr/.
The ALASKA dataset is released under the Creative Commons BY-NC-ND licence.
We have explicit permission from the dataset authors to redistribute the example images.


## End-to-end methods

For the end-to-end methods, we use the following public implementations:

| Method         | Repository                                       | License     | Note                           |
|----------------|--------------------------------------------------|-------------|--------------------------------|
| AdvSteg        | https://github.com/jhayes14/advsteg              |             |                                |
| DeepSteg       | https://github.com/Daniel-EST/deep-steganography |             | unofficial implementation      |
| HiDDeN         | https://github.com/ando-khachatryan/HiDDeN       |             | reimplementation for torch 2.0 |
| SteganoGAN     | https://github.com/DAI-Lab/SteganoGAN            | MIT license |                                |
| HiNet          | https://github.com/TomTomTommi/HiNet             |             |                                |
| CRoSS          | https://github.com/yujiwen/CRoSS                 |             |                                |
| DiffStega      | https://github.com/evtricks/diffstega            |             |                                |


We provide example images and the prediction vectors over the test set for these methods, as well as for the analytical methods HILL and WOW and for the proposed method.
The test set has 12,001 images.

For methods whose embedding is computationally expensive, we trained and tested the detectors on a subset of the images.
This does not affect the conclusions, since the detectors detect these methods almost perfectly (ROC AUC above 0.999).
The predictions for these methods cover only part of the test set:

| Method    | Test images covered |
|-----------|---------------------|
| CRoSS     | 768                 |
| DiffStega | 866                 |
| DeepSteg  | 5,956               |


## Publishing

This material is released under a CC-BY-4.0 licence. The images are released under the respective ALASKA license.


## Cite us!

If you are advancing our idea further or comparing to our method, please cite the paper.

```bibtex
@inproceedings{benes2026variational,
	title = {Variational cover modification steganography},
	author = {Bene{\v s}, Martin and B{\"o}hme, Rainer},
	booktitle = {Conference on Neural Information Processing Systems ({NeurIPS})},
	year = {2026}
}
```

