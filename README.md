# Thesis
### Adversarial de-biasing on skin cancer data using Lantent Adversarial De-biasing
The present work is a bachelor thesis with main focus mitigating bias in skin lesion classification images. The code is by no means optimized. To run the code you need to first download the data (which i will upload to kaggle soon) and insert them into the archive folder. Other than that you just open a terminal to the thesis folder, install the requirements.txt and run the `main.py` file.

## Methodology
The Latent Adversarial Debiasing pipeline is composed of ta Vector-Quantizing Variational AutoEncoder (VQ-VAE) and an intermediate classifier used for calculating the "easy to learn features of the images".
Adversarial Walk is used to find a unique images petrurbation such that the entropy of the simple classifier is high, so that the easy to learn features that are not relevant to the primary classification goal are seperated from the signal, making the models more bias tolerant and robust.

The aim is to mitigate bias in already biased datasets, and specifically skin cancer melanoma datasets where the skin tone can be a real issue in diagnosing and treating this form of cancer.
