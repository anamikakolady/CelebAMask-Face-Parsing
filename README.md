# MobileNet U-Net with CBAM for Face Parsing

This repository contains a lightweight face parsing model developed for the CelebAMask-HQ dataset as part of the AI6126 Advanced Computer Vision project.

## Overview

The proposed architecture combines:

* **MobileNet-style depthwise separable convolutions** for computational efficiency
* **U-Net** encoder-decoder structure for semantic segmentation
* **Convolutional Block Attention Module (CBAM)** for enhanced feature representation
* **Dilated convolutions** for improved contextual understanding
* **Bilinear upsampling** for smoother segmentation outputs

The model is designed to achieve accurate facial component segmentation while maintaining a low parameter count (~1.46M trainable parameters).

## Performance

* Validation Accuracy: **91.55%**
* Validation F-Measure: **0.7868**
* Test F-Measure: **0.81608**

## Repository Structure

* `dev-public/` – Training images and masks
* `test-public/` – Test images
* `model/` – Trained model checkpoint
* `scoring_result/` – Predicted segmentation masks
* `mobilenet_unet_cbam.py` – Model architecture
* `main.py` – Training pipeline
* `run.py` – Inference and mask generation
* `requirements.txt` – Project dependencies

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python main.py
```

Generate predictions:

```bash
python run.py
```

## Author

Anamika Martin Kolady
M.Sc. Artificial Intelligence, Nanyang Technological University
