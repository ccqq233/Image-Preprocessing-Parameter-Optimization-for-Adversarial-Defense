# Image-Preprocessing-Parameter-Optimization-for-Adversarial-Defense
Developed a comprehensive optimization framework for two image‑based defense techniques, Gaussian blur and non‑local means denoising, to counter adversarial attacks on image classifiers. 

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org/)

This repository provides two comprehensive optimization frameworks for **image preprocessing‑based adversarial defenses**: **Gaussian blur** and **non‑local means denoising**. Given a set of adversarial examples generated with different perturbation budgets (epsilon), the framework automatically searches for optimal defense parameters that best balance **defense improvement** (increase in adversarial accuracy) against **degradation of clean images**.

Both frameworks produce detailed reports including visualizations, confusion matrices, confidence distributions, classification metrics, quality radar charts, and raw prediction data.

---

## Project Structure
.
├── combined_blur_defense.py # Gaussian blur optimization
├── combined_nlmeans_defense.py # Non‑local means denoising optimization
├── model3.py # OptimizedCNN model definition (required)
├── best_model.pth # Pretrained model weights
├── adversarial_eval_samples/ # Folder containing adversarial .pkl files
│ ├── adv_samples_eps0.01.pkl
│ ├── adv_samples_eps0.03.pkl
│ └── ...
├── comprehensive_blur_defense/ # Output folder for Gaussian blur
└── comprehensive_nlmeans_defense/ # Output folder for NLM defense



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

---

## Requirements

Install the required packages:

```bash
pip install torch torchvision numpy matplotlib seaborn pandas scikit-learn opencv-python
Python 3.8+

PyTorch 1.9+

torchvision

OpenCV (for NLM denoising)

Other common libraries: numpy, matplotlib, seaborn, pandas, scikit-learn

Usage
1. Prepare the model and adversarial samples
Train a classifier (here OptimizedCNN from model3.py) and save its weights as best_model.pth.

Generate adversarial examples using your preferred attack (e.g., FGSM, PGD) and save them as .pkl files in adversarial_eval_samples/.
Each pickle file should contain a dictionary with keys:

epsilon: perturbation budget

adversarial_images: tensor of adversarial images

original_images: (optional) tensor of original clean images

labels: tensor of true labels

perturbations: (optional) tensor of perturbations

2. Run Gaussian blur optimization
bash
python combined_blur_defense.py
Output will be saved in ./comprehensive_blur_defense/.

3. Run non‑local means denoising optimization
bash
python combined_nlmeans_defense.py
Output will be saved in ./comprehensive_nlmeans_defense/.

If both optimizations have been run, the NLM script will automatically generate a comparison report between the two methods.

Output Explanation
For each epsilon value, a folder epsilon_<value>/ is created containing:

parameter_selection/

all_parameter_results.csv: evaluation results for all parameter combinations

best_parameters.json: best parameters and their scores

analysis/

defense_visualization.png / nlmeans_defense_visualization.png: sample images before/after defense

confusion_comparison.png / nlmeans_confusion_comparison.png: confusion matrices

defense_analysis.png / nlmeans_defense_analysis.png: class‑wise defense effectiveness

confidence_comparison.png / nlmeans_confidence_comparison.png: confidence distributions

parameter_effect_analysis.png / nlmeans_parameter_effect_analysis.png: parameter impact

quality_assessment_radar.png / nlmeans_quality_assessment_radar.png: quality radar chart

classification_comparison.png / nlmeans_classification_comparison.png: F1‑score comparison

classification_reports.json / nlmeans_classification_reports.json: detailed metrics

defense_raw_results.csv / nlmeans_defense_raw_results.csv: raw prediction data

comprehensive_defense_report.txt / nlmeans_comprehensive_defense_report.txt: textual summary

Global output files (in the root output folder):

global_optimization_summary.csv: aggregated results for all epsilons

global_analysis.png: trend plots across epsilons

global_optimization_report.txt: global insights and recommendations

execution_log.json: runtime information
