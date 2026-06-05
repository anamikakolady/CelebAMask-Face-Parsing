# MobileNet U-Net with CBAM for Face Parsing

A lightweight face parsing model developed for the AI6126 Advanced Computer Vision Face Parsing Challenge using the CelebAMask-HQ dataset.

## Sample Results
<p align="center">
  <img src="assets/sample_input.jpg" width="250" alt="Input Image">
  <img src="assets/sample_prediction.png" width="250" alt="Predicted Mask">
</p>

<p align="center">
  <em>Input Image (left) and Predicted Segmentation Mask (right).</em>
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

## Academic Integrity Notice

This repository is shared for educational purposes, academic discussion, and portfolio demonstration. It contains work completed as part of an academic project at Nanyang Technological University (NTU).

If you are a student working on a similar assignment, challenge, or coursework, you may refer to this repository to understand concepts, implementation approaches, and experimental methodologies. However, you must not copy, reproduce, or submit any portion of this work as your own.

Any use of ideas, code, results, or documentation from this repository should be appropriately cited in accordance with your institution's academic integrity policies.

The author does not grant permission for this repository to be used for plagiarism, academic misconduct, or unauthorized submission. The author assumes no responsibility for any misuse of the contents of this repository.

Students are encouraged to consult and comply with the academic integrity policies of their respective institutions before using any material from this repository.
