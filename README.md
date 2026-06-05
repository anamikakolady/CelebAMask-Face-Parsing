# MobileNet U-Net with CBAM for Face Parsing

A lightweight face parsing model developed for the AI6126 Advanced Computer Vision Face Parsing Challenge using the CelebAMask-HQ dataset.

## Sample Results
<p align="center">
  <img src="assets/sample_input.jpg" width="250" alt="Input Image">
  <img src="assets/sample_prediction.png" width="250" alt="Predicted Mask">
</p>

<p align="center">
  <em>Input image (left) and predicted segmentation mask (right).</em>
</p>

## Overview

This project proposes a MobileNet U-Net architecture enhanced with a Convolutional Block Attention Module (CBAM) for accurate and computationally efficient face parsing. The model combines depthwise separable convolutions, attention mechanisms, and dilated convolutions to achieve strong segmentation performance while maintaining a low parameter count.

### Key Features

* MobileNet-style depthwise separable convolutions for efficient computation
* U-Net encoder-decoder architecture for semantic segmentation
* CBAM attention module for improved feature representation
* Dilated convolutions for enhanced contextual understanding
* Bilinear upsampling to reduce checkerboard artifacts
* Lightweight design with approximately 1.46 million trainable parameters

## Results

| Metric               | Score   |
| -------------------- | ------- |
| Validation Accuracy  | 91.55%  |
| Validation F-Measure | 0.7868  |
| Test F-Measure       | 0.81608 |

## Repository Structure

```text
├── scoring_result/       # Predicted segmentation masks
├── CelebAMask_Face_Parsing/
│   ├── dev-public/       # Training images and masks
│   ├── test-public/      # Test images
│   ├── model/            # Trained model checkpoint
│   ├── mobilenet_unet_cbam.py
│   ├── main.py
│   ├── run.py
│   └── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
python main.py
```

## Inference

```bash
python run.py
```

## Author

**Anamika Martin Kolady**

M.Sc. Artificial Intelligence, Nanyang Technological University (NTU)

