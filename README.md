# Image-Preprocessing-Parameter-Optimization-for-Adversarial-Defense
Developed a comprehensive optimization framework for two image‑based defense techniques, Gaussian blur and non‑local means denoising, to counter adversarial attacks on image classifiers. 

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org/)

This repository provides two comprehensive optimization frameworks for **image preprocessing‑based adversarial defenses**: **Gaussian blur** and **non‑local means denoising**. Given a set of adversarial examples generated with different perturbation budgets (epsilon), the framework automatically searches for optimal defense parameters that best balance **defense improvement** (increase in adversarial accuracy) against **degradation of clean images**.

Both frameworks produce detailed reports including visualizations, confusion matrices, confidence distributions, classification metrics, quality radar charts, and raw prediction data.

---

## Project Structure
```
.
├── combined_blur_defense.py      # Main script for Gaussian Blur optimization
├── combined_nlmeans_defense.py   # Main script for Non-Local Means optimization
├── model3.py                     # (Required) Contains the OptimizedCNN model class
├── best_model.pth                # (Required) Pre-trained model weights
├── adversarial_eval_samples/     # (Required) Directory containing .pkl adversarial datasets
│   ├── adv_samples_eps0.1.pkl
│   └── ...
├── comprehensive_blur_defense/   # Output directory for Gaussian results
└── comprehensive_nlmeans_defense/# Output directory for NLM results
```


## Features

- **Automatic parameter search** over:
  - Gaussian blur: kernel size (3,5,7) and sigma (0.1–1.5)
  - Non‑local means denoising: filter strength `h`, template window size, search window size
- **Custom scoring function** that non‑linearly weights defense improvement, clean image degradation, and blur/denoise intensity (with diminishing returns and extra penalties for excessive degradation)
- **Per‑epsilon detailed reports**, including:
  - Sample visualizations (original / adversarial / defended images, perturbation maps, probability distributions)
  - Confusion matrices (before and after defense)
  - Class‑wise defense analysis
  - Confidence distributions and change analysis
  - Parameter effect analysis
  - Quality assessment radar charts
  - Classification reports (precision, recall, F1)
  - Raw prediction data (CSV) and summary JSON
- **Global analysis** across all epsilon values:
  - Parameter trend plots (e.g., optimal sigma vs epsilon)
  - Accuracy trends
  - Defense quality distribution
  - Global optimization summary and report
- **Comparison between Gaussian blur and NLM** (if both outputs exist)

## Methodology
The Trade-off Score
Traditional defense evaluation often looks only at accuracy recovery. This project uses a custom Advanced Trade-off Score to ensure the defense is practical:

### $$Score = Defense_{Gain} - (\alpha \cdot Orig_{Degradation}) - (\beta \cdot Blur_{Strength})$$

- $$Defense_{Gain}$$: Improvement in accuracy on adversarial examples.

- $$Orig_{Degradation}$$: Loss of accuracy on clean images (non-linear penalty increases as degradation worsens).

- $$Blur_{Strength}$$: Penalty for excessive visual distortion.

- α: penalty coefficient for original image degradation. It increases non‑linearly when degradation exceeds certain thresholds, penalizing excessive loss of clean accuracy.

- β: penalty coefficient for blur/denoising strength. It grows with stronger blur or denoising, discouraging overly aggressive smoothing.


## Requirements

### Install the required packages:

```
pip install torch torchvision numpy matplotlib seaborn pandas scikit-learn opencv-python
```
Python 3.8+

PyTorch 1.9+

torchvision

OpenCV (for NLM denoising)

Other common libraries: numpy, matplotlib, seaborn, pandas, scikit-learn

## Usage

### Preparing Adversarial Samples

Place one or more `.pkl` files in the `adversarial_eval_samples/` directory. Each file corresponds to a specific perturbation budget (epsilon) and must be a Python dictionary containing the following keys:


| Key                  | Type          | Required | Shape              | Description                                                                 |
|----------------------|---------------|----------|--------------------|-----------------------------------------------------------------------------|
| `epsilon`            | float         | Yes      | –                  | Perturbation budget used to generate the samples (e.g., 0.1, 0.3).          |
| `adversarial_images` | torch.Tensor  | Yes      | `(N, C, H, W)`     | Adversarial examples. The value range should match your model's input (e.g., `[-1, 1]` or `[0, 1]`). |
| `labels`             | torch.Tensor  | Yes      | `(N,)`             | Ground‑truth labels (integers).                                             |
| `original_images`    | torch.Tensor  | No       | `(N, C, H, W)`     | Corresponding clean images. If provided, the framework will analyze the degradation caused by the defense. |
| `perturbations`      | torch.Tensor  | No       | `(N, C, H, W)`     | Adversarial perturbations (adversarial - original). Used for visualization. |

**Note on Data Consistency**: Ensure all tensors are moved to CPU before saving to the pickle file to avoid CUDA-related loading errors on different environments.

**File naming** is not strict, but we recommend using a format like `adv_samples_eps{epsilon}.pkl` (e.g., `adv_samples_eps0.1.pkl`) for clarity.

**Example creation script** (using FGSM, but any attack works):

```python
import torch
import pickle

# Assume you have a model, loader, and attack function
epsilon = 0.1
adv_images, orig_images, labels, perturbations = generate_adversarial_samples(...)

data = {
    'epsilon': epsilon,
    'adversarial_images': adv_images.cpu(),   # 确保在 CPU 上
    'labels': labels.cpu(),
    'original_images': orig_images.cpu() if orig_images is not None else None,
    'perturbations': perturbations.cpu() if perturbations is not None else None
}

with open(f'adversarial_eval_samples/adv_samples_eps{epsilon}.pkl', 'wb') as f:
    pickle.dump(data, f)
```

### Run Gaussian Blur Optimization
```bash
python combined_blur_defense.py
```
### Run Non-Local Means Optimization
```bash
python combined_nlmeans_defense.py
```
