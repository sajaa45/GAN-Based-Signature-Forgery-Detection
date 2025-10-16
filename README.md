```markdown
# Signature Forgery Detection Using GANs

This project implements an end-to-end system for **signature forgery detection** by combining **Generative Adversarial Networks (GANs)** for synthetic signature generation and a **Convolutional Neural Network (CNN)** for classification.

---

## Overview

Signature verification is a key task in biometric authentication and fraud prevention. Collecting large datasets of genuine and forged signatures is often challenging. This project addresses this by:

1. Training a GAN to learn the visual distribution of genuine signatures.
2. Generating synthetic (fake) signature samples to augment limited forged data.
3. Training a CNN classifier to distinguish between genuine and forged signatures.

The system demonstrates the potential of combining **generative** and **discriminative** models for enhanced forgery detection.

The system demonstrates the potential of combining **generative** and **discriminative** models for enhanced forgery detection. 

A full detailed report documenting the methodology, experiments, and results is included in this repository.

---

## Features

- **GAN Training:** Generates realistic synthetic signatures using a Wasserstein GAN with Gradient Penalty (WGAN-GP).
- **CNN Classification:** Trains a convolutional neural network to classify genuine vs forged signatures.
- **Evaluation on GAN Samples:** Assesses the realism of generated signatures using the trained CNN.

---

## Dataset

- 2,500 genuine signature images and 2,500 forged signature images.
- Split: 80% training / 20% validation.
- Preprocessed to grayscale, resized to `128x128` for GAN, `64x64` for CNN, and normalized appropriately.

---

## Project Structure

```

src/
├── app.py                     # Main script to run different stages
├── data_prep.py               # Preprocessing pipeline for the dataset
├── train_gan.py               # GAN training script
├── generate_fakes.py          # Generate synthetic signatures from trained GAN
├── train_signature_classifier.py  # CNN classifier training
└── evaluate_gan_with_cnn.py   # Evaluate generated images with trained CNN
data/
└── generated_fakes/           # Stores generated fake signatures

````

---

## Usage

Run the main pipeline using:

```bash
python src/app.py
````

In `app.py`, each step can be uncommented depending on which stage you want to execute:

* `run_data_prep()` – Preprocesses and organizes the dataset.
* `run_gan_training()` – Trains the GAN on genuine signatures.
* `run_generate_fakes()` – Generates synthetic forged signatures.
* `run_cnn_training()` – Trains the CNN classifier.
* `run_gan_evaluation()` – Evaluates GAN-generated images with the trained CNN.

---

## Results

* **CNN Accuracy:** ~75% on the test dataset.
* **GAN Evaluation:** ~55% of generated samples predicted as forged by the CNN, indicating the GAN produces partially realistic features.
* **Training Observations:** Generator gradually improves during training, producing visually coherent signatures, though some instability occurs in later epochs.



```

