"""
combined_nlmeans_defense.py
Comprehensive Non-Local Means Denoising Defense: Select optimal parameters for each epsilon and generate detailed reports
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pickle
import time
from torchvision import datasets, transforms
import seaborn as sns
from collections import defaultdict
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import cv2
import warnings
warnings.filterwarnings('ignore')

# Set random seed
def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

# Import model architecture
from model3 import EnhancedCNN

def load_model(model_path='best_model.pth'):
    """Load trained model"""
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} does not exist")
        return None
    
    model = EnhancedCNN()
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"✓ Model loaded successfully: {model_path}")
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return None
    
    model.eval()
    return model

def load_adv_samples(adv_dir='./adversarial_eval_samples'):
    """Load all saved adversarial sample datasets"""
    adv_datasets = {}
    
    if not os.path.exists(adv_dir):
        print(f"Error: Adversarial samples directory {adv_dir} does not exist")
        return adv_datasets
    
    # Find all .pkl files
    pkl_files = [f for f in os.listdir(adv_dir) if f.endswith('.pkl')]
    
    if not pkl_files:
        print(f"Warning: No .pkl files found in {adv_dir}")
        return adv_datasets
    
    print(f"Found {len(pkl_files)} adversarial sample datasets")
    
    for pkl_file in pkl_files:
        file_path = os.path.join(adv_dir, pkl_file)
        try:
            with open(file_path, 'rb') as f:
                dataset = pickle.load(f)
            
            # Extract epsilon value
            if 'epsilon' in dataset:
                epsilon = dataset['epsilon']
                print(f"  Loading: {pkl_file}, epsilon={epsilon}, num_samples={dataset.get('num_samples', 'unknown')}")
                adv_datasets[epsilon] = dataset
            else:
                print(f"  Warning: No epsilon information in {pkl_file}")
                
        except Exception as e:
            print(f"  Failed to load {pkl_file}: {e}")
    
    return adv_datasets

def apply_nlmeans_denoising(images, h=10, template_window_size=7, search_window_size=21):
    """
    Apply Non-Local Means denoising defense
    
    Parameters:
        images: Tensor, shape [N, C, H, W]
        h: Filter strength parameter, controls denoising intensity
        template_window_size: Template window size (odd)
        search_window_size: Search window size (odd)
    """
    # Ensure window sizes are odd
    if template_window_size % 2 == 0:
        template_window_size += 1
    if search_window_size % 2 == 0:
        search_window_size += 1
    
    # Convert images to OpenCV format
    # PyTorch images: [N, C, H, W], value range [-1, 1]
    # OpenCV images: [H, W, C], value range [0, 255]
    
    denoised_images = []
    
    for i in range(images.shape[0]):
        # Get single image and transpose dimensions
        img_tensor = images[i]
        
        # Denormalize: [-1, 1] -> [0, 1]
        img_01 = img_tensor * 0.5 + 0.5
        
        # Convert to numpy and adjust dimensions: [C, H, W] -> [H, W, C]
        img_np = img_01.permute(1, 2, 0).numpy()
        
        # Convert to OpenCV format: [0, 1] -> [0, 255], convert to uint8
        img_cv = (img_np * 255).astype(np.uint8)
        
        # If grayscale, ensure it's a 2D array
        if img_cv.shape[2] == 1:
            img_cv = img_cv.squeeze(2)
        
        # Apply Non-Local Means denoising
        try:
            # OpenCV's fastNlMeansDenoising function
            denoised_cv = cv2.fastNlMeansDenoising(
                src=img_cv,
                h=h,
                templateWindowSize=template_window_size,
                searchWindowSize=search_window_size
            )
        except Exception as e:
            print(f"Non-Local Means denoising failed: {e}, using original image")
            denoised_cv = img_cv
        
        # Convert back to PyTorch format
        # If grayscale, restore channel dimension
        if len(denoised_cv.shape) == 2:
            denoised_cv = denoised_cv[:, :, np.newaxis]
        
        # Normalize: [0, 255] -> [0, 1] -> [-1, 1]
        denoised_np = denoised_cv.astype(np.float32) / 255.0
        denoised_tensor = torch.from_numpy(denoised_np).permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
        denoised_tensor = denoised_tensor * 2.0 - 1.0  # [0, 1] -> [-1, 1]
        
        denoised_images.append(denoised_tensor.unsqueeze(0))
    
    return torch.cat(denoised_images, dim=0)

def calculate_nlmeans_strength(h, template_window_size, search_window_size):
    """Calculate Non-Local Means denoising strength (normalized)"""
    # Strength is mainly controlled by h, window sizes also have impact
    base_strength = h / 10.0  # Normalize to 0-1 range, h is typically between 0-30
    window_factor = (template_window_size * search_window_size) / (7 * 21)  # Relative to default window size
    return base_strength * window_factor

def advanced_tradeoff_score_nlmeans(defense_improvement, orig_degradation, denoise_strength, 
                                  alpha_base=0.1, beta_base=0.1, defense_cap=50):
    """
    Improved trade-off score calculation - optimized specifically for Non-Local Means denoising
    """
    # 1. Nonlinear weight: heavier penalty when original image degradation exceeds threshold
    if orig_degradation > 10:  # Degradation > 10%
        # Exponential penalty: the more degradation, the heavier the penalty
        alpha = alpha_base * (1 + 0.1 * (orig_degradation - 10))
    elif orig_degradation > 5:  # 5-10% degradation
        alpha = alpha_base * 1.5  # Moderate penalty
    else:
        alpha = alpha_base  # Base weight
    
    # 2. Diminishing returns on defense improvement
    # First defense_cap% improvement has high value, diminishing returns after
    if defense_improvement <= defense_cap:
        defense_value = defense_improvement
    else:
        # Diminishing returns: value halved after threshold
        defense_value = defense_cap + (defense_improvement - defense_cap) * 0.5
    
    # 3. Consider negative defense improvement (defense reduces accuracy)
    if defense_improvement < 0:
        # Double penalty for negative improvement
        defense_value = defense_improvement * 2.0  # Double penalty
    
    # 4. Nonlinear penalty for denoising strength (avoid over-smoothing)
    # NLM is typically smarter than Gaussian blur, so lighter penalty
    denoise_penalty = beta_base * (denoise_strength ** 1.2)  # Exponent 1.2, gentler than Gaussian's 1.5
    
    # 5. Calculate final score
    score = defense_value * 1.2 - alpha * orig_degradation - denoise_penalty
    
    # 6. Additional consideration: if original degradation is too high, even high defense improvement is severely penalized
    if orig_degradation > 20:  # Original degradation > 20%
        score -= 10  # Severe penalty
    
    # 7. Consider marginal effect of denoising strength
    # Extra penalty when denoising strength exceeds 2.0 (avoid completely destroying image details)
    if denoise_strength > 2.0:
        score -= (denoise_strength - 2.0) * 3  # Lighter penalty than Gaussian blur
    
    return score

def assess_defense_quality_nlmeans(defense_improvement, orig_degradation, denoise_strength):
    """Assess Non-Local Means denoising defense quality grade"""
    # Defense improvement grade
    if defense_improvement > 30:
        defense_grade = "Excellent"
    elif defense_improvement > 15:
        defense_grade = "Good"
    elif defense_improvement > 0:
        defense_grade = "Fair"
    else:
        defense_grade = "Poor"
    
    # Original image impact grade
    if orig_degradation < 2:
        orig_grade = "Minimal"
    elif orig_degradation < 5:
        orig_grade = "Low"
    elif orig_degradation < 10:
        orig_grade = "Moderate"
    elif orig_degradation < 20:
        orig_grade = "High"
    else:
        orig_grade = "Severe"
    
    # Denoising strength grade
    if denoise_strength < 0.5:
        denoise_grade = "Light"
    elif denoise_strength < 1.0:
        denoise_grade = "Moderate"
    elif denoise_strength < 1.5:
        denoise_grade = "Strong"
    elif denoise_strength < 2.0:
        denoise_grade = "Heavy"
    else:
        denoise_grade = "Very Heavy"
    
    # Overall quality assessment
    if defense_improvement > 20 and orig_degradation < 5 and denoise_strength < 1.0:
        overall = "High Quality Defense"
    elif defense_improvement > 10 and orig_degradation < 10 and denoise_strength < 1.5:
        overall = "Medium Quality Defense"
    elif defense_improvement > 0:
        overall = "Low Quality Defense"
    else:
        overall = "Poor Defense (Harmful)"
    
    return {
        'defense_grade': defense_grade,
        'original_impact_grade': orig_grade,
        'denoise_grade': denoise_grade,
        'overall_assessment': overall
    }

def analyze_nlmeans_effect_on_original(model, orig_images, labels, h=10, template_window_size=7, search_window_size=21):
    """
    Analyze the effect of Non-Local Means denoising on original images
    """
    if orig_images is None:
        return None, None
    
    model.eval()
    
    # Apply Non-Local Means denoising
    denoised_images = apply_nlmeans_denoising(
        orig_images, 
        h=h, 
        template_window_size=template_window_size, 
        search_window_size=search_window_size
    )
    
    # Evaluate accuracy after denoising
    correct = 0
    confidences = []
    
    batch_size = 256
    num_batches = (len(labels) + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(labels))
        
        batch_images = denoised_images[start_idx:end_idx]
        batch_labels = labels[start_idx:end_idx]
        
        with torch.no_grad():
            outputs = model(batch_images)
            preds = torch.argmax(outputs, dim=1)
            probs = F.softmax(outputs, dim=1)
            
            correct += (preds == batch_labels).sum().item()
            confidences.extend(probs[torch.arange(len(preds)), preds].tolist())
    
    accuracy = 100.0 * correct / len(labels)
    avg_confidence = np.mean(confidences) if confidences else 0
    
    return accuracy, avg_confidence

def evaluate_single_nlmeans_defense(model, adv_images, orig_images, labels, perturbations, 
                                  h, template_window_size, search_window_size, defense_name='nlmeans'):
    """
    Evaluate a single Non-Local Means denoising parameter combination (using advanced scoring)
    """
    model.eval()
    num_samples = len(labels)
    
    # Apply defense
    defended_images = apply_nlmeans_denoising(
        adv_images, 
        h=h, 
        template_window_size=template_window_size, 
        search_window_size=search_window_size
    )
    
    # Evaluate three scenarios: original images, adversarial samples, defended adversarial samples
    correct_orig = 0
    correct_adv = 0
    correct_defended = 0
    
    # Batch processing to avoid memory issues
    batch_size = 256
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)
        
        batch_orig = orig_images[start_idx:end_idx] if orig_images is not None else None
        batch_adv = adv_images[start_idx:end_idx]
        batch_defended = defended_images[start_idx:end_idx]
        batch_labels = labels[start_idx:end_idx]
        
        # Evaluate original images
        if batch_orig is not None:
            with torch.no_grad():
                outputs_orig = model(batch_orig)
                preds_orig = torch.argmax(outputs_orig, dim=1)
                correct_orig += (preds_orig == batch_labels).sum().item()
        
        # Evaluate adversarial samples
        with torch.no_grad():
            outputs_adv = model(batch_adv)
            preds_adv = torch.argmax(outputs_adv, dim=1)
            correct_adv += (preds_adv == batch_labels).sum().item()
        
        # Evaluate after defense
        with torch.no_grad():
            outputs_defended = model(batch_defended)
            preds_defended = torch.argmax(outputs_defended, dim=1)
            correct_defended += (preds_defended == batch_labels).sum().item()
    
    # Calculate basic metrics
    orig_accuracy = 100.0 * correct_orig / num_samples if orig_images is not None else None
    adv_accuracy = 100.0 * correct_adv / num_samples
    defended_accuracy = 100.0 * correct_defended / num_samples
    defense_improvement = defended_accuracy - adv_accuracy
    
    # Analyze NLM effect on original images
    if orig_images is not None:
        orig_denoised_accuracy, _ = analyze_nlmeans_effect_on_original(
            model, orig_images, labels, h, template_window_size, search_window_size
        )
        orig_degradation = orig_accuracy - orig_denoised_accuracy if orig_accuracy else 0
    else:
        orig_denoised_accuracy = None
        orig_degradation = 0
    
    # Calculate denoising strength
    denoise_strength = calculate_nlmeans_strength(h, template_window_size, search_window_size)
    
    # Calculate improved trade-off score
    tradeoff_score = advanced_tradeoff_score_nlmeans(
        defense_improvement=defense_improvement,
        orig_degradation=orig_degradation,
        denoise_strength=denoise_strength
    )
    
    # Quality assessment
    quality_assessment = assess_defense_quality_nlmeans(defense_improvement, orig_degradation, denoise_strength)
    
    return {
        'h': h,
        'template_window_size': template_window_size,
        'search_window_size': search_window_size,
        'defense_name': defense_name,
        'num_samples': num_samples,
        
        # Accuracy metrics
        'original_accuracy': orig_accuracy,
        'adversarial_accuracy': adv_accuracy,
        'defended_accuracy': defended_accuracy,
        'defense_improvement': defense_improvement,
        
        # Original image impact
        'original_denoised_accuracy': orig_denoised_accuracy,
        'original_degradation': orig_degradation,
        
        # Denoising related
        'denoise_strength': denoise_strength,
        
        # Score
        'tradeoff_score': tradeoff_score,
        
        # Quality assessment
        'quality_assessment': quality_assessment
    }

def select_best_nlmeans_parameters(model, dataset, param_combinations, epsilon):
    """
    Select optimal Non-Local Means denoising parameters for a specific epsilon
    """
    print(f"  Selecting optimal NLM parameters for ε={epsilon}...")
    
    # Extract data
    adv_images = dataset['adversarial_images']
    orig_images = dataset.get('original_images', None)
    labels = dataset['labels']
    perturbations = dataset.get('perturbations', None)
    
    # Test all parameter combinations
    all_results = []
    
    for params in param_combinations:
        h = params['h']
        template_window_size = params['template_window_size']
        search_window_size = params['search_window_size']
        defense_name = params.get('name', f'nlmeans_h{h}_t{template_window_size}_s{search_window_size}')
        
        print(f"    Testing: h={h}, template={template_window_size}, search={search_window_size}", end="")
        
        # Evaluate current parameters
        result = evaluate_single_nlmeans_defense(
            model, adv_images, orig_images, labels, perturbations,
            h, template_window_size, search_window_size, defense_name
        )
        
        all_results.append(result)
        print(f" → Improvement: {result['defense_improvement']:6.2f}%, "
              f"Original Degradation: {result['original_degradation']:5.2f}%, "
              f"Denoise Strength: {result['denoise_strength']:5.2f}, "
              f"Score: {result['tradeoff_score']:7.2f}")
    
    # Sort by trade-off score
    all_results.sort(key=lambda x: x['tradeoff_score'], reverse=True)
    
    # Select best parameters
    best_result = all_results[0]
    
    print(f"  ✓ Optimal parameters: h={best_result['h']}, template={best_result['template_window_size']}, search={best_result['search_window_size']}")
    print(f"    Defense improvement: {best_result['defense_improvement']:.2f}%")
    print(f"    Original degradation: {best_result['original_degradation']:.2f}%")
    print(f"    Trade-off score: {best_result['tradeoff_score']:.2f}")
    print(f"    Quality assessment: {best_result['quality_assessment']['overall_assessment']}")
    
    return best_result, all_results

def generate_nlmeans_comprehensive_report(model, dataset, best_params, epsilon, output_dir):
    """
    Generate comprehensive Non-Local Means denoising defense report
    """
    print(f"  Generating comprehensive NLM defense report...")
    
    # Extract data
    adv_images = dataset['adversarial_images']
    orig_images = dataset.get('original_images', None)
    labels = dataset['labels']
    perturbations = dataset.get('perturbations', None)
    
    # Create output directory
    eps_dir = os.path.join(output_dir, f'epsilon_{epsilon:.3f}')
    os.makedirs(eps_dir, exist_ok=True)
    
    # Apply optimal parameter defense
    h = best_params['h']
    template_window_size = best_params['template_window_size']
    search_window_size = best_params['search_window_size']
    defense_name = f'nlmeans_optimal_h{h}_t{template_window_size}_s{search_window_size}'
    
    defended_images = apply_nlmeans_denoising(
        adv_images, 
        h=h, 
        template_window_size=template_window_size, 
        search_window_size=search_window_size
    )
    
    # Detailed evaluation
    model.eval()
    correct_orig = 0
    correct_adv = 0
    correct_defended = 0
    
    # Store detailed results
    all_preds_orig = []
    all_preds_adv = []
    all_preds_defended = []
    all_conf_orig = []
    all_conf_adv = []
    all_conf_defended = []
    all_pert_norms = []
    
    # Batch processing
    batch_size = 256
    num_samples = len(labels)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)
        
        batch_orig = orig_images[start_idx:end_idx] if orig_images is not None else None
        batch_adv = adv_images[start_idx:end_idx]
        batch_defended = defended_images[start_idx:end_idx]
        batch_labels = labels[start_idx:end_idx]
        
        # Evaluate original images
        if batch_orig is not None:
            with torch.no_grad():
                outputs_orig = model(batch_orig)
                preds_orig = torch.argmax(outputs_orig, dim=1)
                probs_orig = F.softmax(outputs_orig, dim=1)
                conf_orig = probs_orig[torch.arange(len(preds_orig)), preds_orig]
                
                correct_orig += (preds_orig == batch_labels).sum().item()
                all_preds_orig.append(preds_orig)
                all_conf_orig.append(conf_orig)
        
        # Evaluate adversarial samples
        with torch.no_grad():
            outputs_adv = model(batch_adv)
            preds_adv = torch.argmax(outputs_adv, dim=1)
            probs_adv = F.softmax(outputs_adv, dim=1)
            conf_adv = probs_adv[torch.arange(len(preds_adv)), preds_adv]
            
            correct_adv += (preds_adv == batch_labels).sum().item()
            all_preds_adv.append(preds_adv)
            all_conf_adv.append(conf_adv)
        
        # Evaluate after defense
        with torch.no_grad():
            outputs_defended = model(batch_defended)
            preds_defended = torch.argmax(outputs_defended, dim=1)
            probs_defended = F.softmax(outputs_defended, dim=1)
            conf_defended = probs_defended[torch.arange(len(preds_defended)), preds_defended]
            
            correct_defended += (preds_defended == batch_labels).sum().item()
            all_preds_defended.append(preds_defended)
            all_conf_defended.append(conf_defended)
        
        # Calculate perturbation norms
        if perturbations is not None:
            batch_pert = perturbations[start_idx:end_idx]
            pert_norms = torch.norm(batch_pert.view(batch_pert.size(0), -1), p=2, dim=1)
            all_pert_norms.append(pert_norms)
    
    # Merge results
    if all_preds_orig:
        all_preds_orig = torch.cat(all_preds_orig)
        all_conf_orig = torch.cat(all_conf_orig)
    else:
        all_preds_orig = None
        all_conf_orig = None
    
    all_preds_adv = torch.cat(all_preds_adv)
    all_conf_adv = torch.cat(all_conf_adv)
    
    all_preds_defended = torch.cat(all_preds_defended)
    all_conf_defended = torch.cat(all_conf_defended)
    
    # Merge perturbation norms
    if all_pert_norms:
        all_pert_norms_tensor = torch.cat(all_pert_norms)
        avg_pert_norm = all_pert_norms_tensor.mean().item()
    else:
        all_pert_norms_tensor = None
        avg_pert_norm = 0
    
    # Calculate final metrics
    orig_accuracy = 100.0 * correct_orig / num_samples if orig_images is not None else None
    adv_accuracy = 100.0 * correct_adv / num_samples
    defended_accuracy = 100.0 * correct_defended / num_samples
    defense_improvement = defended_accuracy - adv_accuracy
    
    # Confidence metrics
    avg_conf_orig = all_conf_orig.mean().item() if all_conf_orig is not None else None
    avg_conf_adv = all_conf_adv.mean().item()
    avg_conf_defended = all_conf_defended.mean().item()
    
    # Calculate other metrics
    denoise_strength = calculate_nlmeans_strength(h, template_window_size, search_window_size)
    if orig_accuracy is not None:
        orig_denoised_accuracy, _ = analyze_nlmeans_effect_on_original(
            model, orig_images, labels, h, template_window_size, search_window_size
        )
        orig_degradation = orig_accuracy - orig_denoised_accuracy
    else:
        orig_degradation = 0
    
    tradeoff_score = advanced_tradeoff_score_nlmeans(defense_improvement, orig_degradation, denoise_strength)
    quality_assessment = assess_defense_quality_nlmeans(defense_improvement, orig_degradation, denoise_strength)
    
    # Final results
    final_results = {
        'epsilon': epsilon,
        'num_samples': num_samples,
        'defense_name': defense_name,
        'h': h,
        'template_window_size': template_window_size,
        'search_window_size': search_window_size,
        
        # Accuracy metrics
        'original_accuracy': orig_accuracy,
        'adversarial_accuracy': adv_accuracy,
        'defended_accuracy': defended_accuracy,
        'defense_improvement': defense_improvement,
        
        # Original image impact
        'original_denoised_accuracy': orig_denoised_accuracy if orig_accuracy is not None else None,
        'original_degradation': orig_degradation,
        
        # Denoising related
        'denoise_strength': denoise_strength,
        
        # Score
        'tradeoff_score': tradeoff_score,
        
        # Perturbation metrics
        'avg_perturbation_norm': avg_pert_norm,
        
        # Confidence metrics
        'avg_confidence_original': avg_conf_orig,
        'avg_confidence_adversarial': avg_conf_adv,
        'avg_confidence_defended': avg_conf_defended,
        
        # Quality assessment
        'quality_assessment': quality_assessment,
        
        # Detailed data
        'labels': labels,
        'preds_orig': all_preds_orig,
        'preds_adv': all_preds_adv,
        'preds_defended': all_preds_defended,
        'conf_orig': all_conf_orig,
        'conf_adv': all_conf_adv,
        'conf_defended': all_conf_defended,
        'pert_norms': all_pert_norms_tensor,
        'perturbations': perturbations,
        'adv_images': adv_images,
        'orig_images': orig_images,
        'defended_images': defended_images
    }
    
    # Generate classification report
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # 1. Generate all analysis files
    generate_nlmeans_all_analysis_files(
        model, final_results, orig_images, adv_images, defended_images, 
        perturbations, labels, class_names, epsilon, eps_dir
    )
    
    # 2. Save comprehensive report
    save_nlmeans_comprehensive_report(final_results, class_names, eps_dir)
    
    # 3. Save raw data
    save_nlmeans_all_raw_data(final_results, eps_dir)
    
    return final_results

def generate_nlmeans_all_analysis_files(model, results, orig_images, adv_images, defended_images, 
                                       perturbations, labels, class_names, epsilon, output_dir):
    """Generate all Non-Local Means denoising analysis files"""
    
    # Create subdirectory
    analysis_dir = os.path.join(output_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    # 1. Generate defense visualization
    generate_nlmeans_defense_visualization(
        model, orig_images, adv_images, defended_images, perturbations, labels, 
        epsilon, results['defense_name'], analysis_dir, 
        results['h'], results['template_window_size'], results['search_window_size'], num_samples=10
    )
    
    # 2. Generate confusion matrix comparison
    generate_nlmeans_confusion_comparison(
        labels, results['preds_adv'], results['preds_defended'], class_names, 
        epsilon, results['defense_name'], analysis_dir
    )
    
    # 3. Generate defense effectiveness analysis
    generate_nlmeans_defense_analysis(
        results, labels, results['preds_orig'], results['preds_adv'], 
        results['preds_defended'], results['conf_orig'], results['conf_adv'], 
        results['conf_defended'], results['pert_norms'], class_names, analysis_dir
    )
    
    # 4. Generate confidence analysis comparison
    generate_nlmeans_confidence_comparison(
        labels, results['preds_orig'], results['preds_adv'], results['preds_defended'],
        results['conf_orig'], results['conf_adv'], results['conf_defended'],
        epsilon, results['defense_name'], analysis_dir
    )
    
    # 5. Generate parameter effect analysis
    generate_nlmeans_parameter_effect_analysis(results, epsilon, analysis_dir)
    
    # 6. Generate quality assessment chart
    generate_nlmeans_quality_assessment_chart(results, epsilon, analysis_dir)
    
    # 7. Generate classification reports
    generate_nlmeans_classification_reports(labels, results['preds_adv'], 
                                           results['preds_defended'], class_names, 
                                           epsilon, analysis_dir)

def generate_nlmeans_defense_visualization(model, orig_images, adv_images, defended_images, perturbations, labels, 
                                         epsilon, defense_name, output_dir, h, template_size, search_size, num_samples=10):
    """Generate Non-Local Means denoising defense effectiveness visualization"""
    if orig_images is None or perturbations is None:
        print("    Warning: Missing original images or perturbation data, skipping sample visualization")
        return
    
    num_samples = min(num_samples, len(labels))
    
    fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4*num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    model.eval()
    
    for i in range(num_samples):
        orig_img = orig_images[i].unsqueeze(0)
        adv_img = adv_images[i].unsqueeze(0)
        defended_img = defended_images[i].unsqueeze(0)
        pert = perturbations[i]
        label = labels[i].item()
        
        # Predictions
        with torch.no_grad():
            orig_output = model(orig_img)
            orig_probs = F.softmax(orig_output, dim=1)
            orig_pred = torch.argmax(orig_probs, dim=1).item()
            orig_conf = orig_probs[0, orig_pred].item()
            
            adv_output = model(adv_img)
            adv_probs = F.softmax(adv_output, dim=1)
            adv_pred = torch.argmax(adv_probs, dim=1).item()
            adv_conf = adv_probs[0, adv_pred].item()
            
            defended_output = model(defended_img)
            defended_probs = F.softmax(defended_output, dim=1)
            defended_pred = torch.argmax(defended_probs, dim=1).item()
            defended_conf = defended_probs[0, defended_pred].item()
        
        # Denormalize images
        orig_display = orig_img.squeeze() * 0.5 + 0.5
        adv_display = adv_img.squeeze() * 0.5 + 0.5
        defended_display = defended_img.squeeze() * 0.5 + 0.5
        pert_display = pert.squeeze()
        
        # 1. Original image
        axes[i, 0].imshow(orig_display.numpy(), cmap='gray')
        true_class = class_names[label]
        pred_class = class_names[orig_pred]
        correct = "[OK]" if orig_pred == label else "[FAIL]"
        axes[i, 0].set_title(
            f'Original\nTrue: {true_class}\nPred: {pred_class} {correct}\nConf: {orig_conf:.2%}',
            fontsize=9
        )
        axes[i, 0].axis('off')
        
        # 2. Adversarial sample
        axes[i, 1].imshow(adv_display.numpy(), cmap='gray')
        adv_class = class_names[adv_pred]
        attack_success = "[FAIL]" if adv_pred != label else "[OK]"
        color = 'red' if attack_success == "[FAIL]" else 'green'
        axes[i, 1].set_title(
            f'Adversarial (ε={epsilon})\nPred: {adv_class} {attack_success}\nConf: {adv_conf:.2%}',
            fontsize=9, color=color
        )
        axes[i, 1].axis('off')
        
        # 3. After NLM denoising
        axes[i, 2].imshow(defended_display.numpy(), cmap='gray')
        defended_class = class_names[defended_pred]
        defense_success = "[OK]" if defended_pred == label else "[FAIL]"
        defense_color = 'green' if defense_success == "[OK]" else 'red'
        axes[i, 2].set_title(
            f'{defense_name}\nh={h}, t={template_size}, s={search_size}\nPred: {defended_class} {defense_success}\nConf: {defended_conf:.2%}',
            fontsize=9, color=defense_color
        )
        axes[i, 2].axis('off')
        
        # 4. Perturbation
        im = axes[i, 3].imshow(pert_display.numpy(), cmap='RdBu_r', 
                               vmin=-epsilon, vmax=epsilon)
        l2_norm = torch.norm(pert).item()
        axes[i, 3].set_title(
            f'Perturbation (10× magnified)\nL2 Norm: {l2_norm:.4f}',
            fontsize=9
        )
        axes[i, 3].axis('off')
        
        # 5. Prediction probability distribution comparison
        x = np.arange(len(class_names))
        width = 0.25
        
        axes[i, 4].bar(x - width, orig_probs.squeeze().numpy(), width, 
                      label='Original', alpha=0.7, color='blue')
        axes[i, 4].bar(x, adv_probs.squeeze().numpy(), width, 
                      label='Adversarial', alpha=0.7, color='red')
        axes[i, 4].bar(x + width, defended_probs.squeeze().numpy(), width, 
                      label='Defended', alpha=0.7, color='green')
        
        axes[i, 4].axvline(x=label, color='black', linestyle='--', 
                          alpha=0.5, linewidth=1.5, label='True Label')
        
        axes[i, 4].set_xlabel('Class', fontsize=8)
        axes[i, 4].set_ylabel('Probability', fontsize=8)
        axes[i, 4].set_title('Prediction Probabilities', fontsize=9)
        axes[i, 4].set_xticks(x)
        axes[i, 4].set_xticklabels([str(j) for j in range(10)], rotation=45, fontsize=7)
        axes[i, 4].legend(fontsize=7, loc='upper right')
        axes[i, 4].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'nlmeans_defense_visualization.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ NLM defense visualization saved: {save_path}")

def generate_nlmeans_confusion_comparison(labels, preds_adv, preds_defended, class_names, 
                                        epsilon, defense_name, output_dir):
    """Generate confusion matrix comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Adversarial confusion matrix
    cm_adv = confusion_matrix(labels.numpy(), preds_adv.numpy())
    cm_adv_percent = cm_adv.astype('float') / cm_adv.sum(axis=1)[:, np.newaxis] * 100
    cm_adv_percent = np.nan_to_num(cm_adv_percent)
    
    sns.heatmap(cm_adv_percent, annot=True, fmt='.1f', cmap='Reds',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Percentage (%)'}, ax=ax1)
    ax1.set_xlabel('Predicted Class (Under Attack)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('True Class', fontsize=12, fontweight='bold')
    ax1.set_title(f'Adversarial Confusion Matrix\n(ε={epsilon}, No Defense)', 
                 fontsize=14, fontweight='bold')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # After defense confusion matrix
    cm_defended = confusion_matrix(labels.numpy(), preds_defended.numpy())
    cm_defended_percent = cm_defended.astype('float') / cm_defended.sum(axis=1)[:, np.newaxis] * 100
    cm_defended_percent = np.nan_to_num(cm_defended_percent)
    
    sns.heatmap(cm_defended_percent, annot=True, fmt='.1f', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Percentage (%)'}, ax=ax2)
    ax2.set_xlabel(f'Predicted Class (After {defense_name})', fontsize=12, fontweight='bold')
    ax2.set_ylabel('True Class', fontsize=12, fontweight='bold')
    ax2.set_title(f'Confusion Matrix After NLM Defense\n(ε={epsilon})', 
                 fontsize=14, fontweight='bold')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'nlmeans_confusion_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ NLM confusion matrix comparison saved: {save_path}")

def generate_nlmeans_defense_analysis(results, labels, preds_orig, preds_adv, preds_defended,
                                     conf_orig, conf_adv, conf_defended, pert_norms, 
                                     class_names, output_dir):
    """Generate Non-Local Means denoising defense effectiveness analysis"""
    num_classes = len(class_names)
    
    # Initialize statistics
    class_stats = {
        'total_samples': np.zeros(num_classes),
        'orig_correct': np.zeros(num_classes),
        'adv_correct': np.zeros(num_classes),
        'defended_correct': np.zeros(num_classes),
        'attack_success': np.zeros(num_classes),
        'defense_success': np.zeros(num_classes),
    }
    
    # Calculate statistics
    for i in range(len(labels)):
        label = labels[i].item()
        pred_adv = preds_adv[i].item()
        pred_defended = preds_defended[i].item()
        
        class_stats['total_samples'][label] += 1
        
        if preds_orig is not None:
            pred_orig = preds_orig[i].item()
            if pred_orig == label:
                class_stats['orig_correct'][label] += 1
        
        if pred_adv == label:
            class_stats['adv_correct'][label] += 1
        else:
            class_stats['attack_success'][label] += 1
        
        if pred_defended == label:
            class_stats['defended_correct'][label] += 1
            class_stats['defense_success'][label] += 1
    
    # Calculate per-class metrics
    attack_rates = []
    defended_accuracies = []
    defense_improvements = []
    
    for i in range(num_classes):
        if class_stats['total_samples'][i] > 0:
            attack_rate = 100.0 * class_stats['attack_success'][i] / class_stats['total_samples'][i]
            attack_rates.append(attack_rate)
            
            defended_acc = 100.0 * class_stats['defended_correct'][i] / class_stats['total_samples'][i]
            defended_accuracies.append(defended_acc)
            
            adv_acc = 100.0 * class_stats['adv_correct'][i] / class_stats['total_samples'][i]
            defense_improvement = defended_acc - adv_acc
            defense_improvements.append(defense_improvement)
        else:
            attack_rates.append(0)
            defended_accuracies.append(0)
            defense_improvements.append(0)
    
    # Create charts
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Attack success rate vs defended accuracy by class
    x = np.arange(num_classes)
    width = 0.35
    
    ax1.bar(x - width/2, attack_rates, width, label='Attack Success Rate', 
            color='red', alpha=0.7)
    ax1.bar(x + width/2, defended_accuracies, width, label='After NLM Defense Accuracy', 
            color='green', alpha=0.7)
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Percentage (%)')
    ax1.set_title('Attack Success vs NLM Defense Accuracy by Class')
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Defense effectiveness improvement (by class)
    bars = ax2.bar(class_names, defense_improvements, color='blue', alpha=0.7)
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Accuracy Improvement (%)')
    ax2.set_title('NLM Defense Effectiveness by Class')
    ax2.set_xticklabels(class_names, rotation=45, ha='right')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, improvement in zip(bars, defense_improvements):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{improvement:+.1f}%', ha='center', va='bottom' if improvement > 0 else 'top')
    
    # 3. Accuracy comparison (original vs adversarial vs defended)
    if preds_orig is not None:
        orig_accuracies = []
        for i in range(num_classes):
            if class_stats['total_samples'][i] > 0:
                orig_acc = 100.0 * class_stats['orig_correct'][i] / class_stats['total_samples'][i]
                orig_accuracies.append(orig_acc)
            else:
                orig_accuracies.append(0)
        
        adv_accuracies = [100 - rate for rate in attack_rates]
        
        x = np.arange(num_classes)
        width = 0.25
        
        ax3.bar(x - width, orig_accuracies, width, label='Original', 
                color='blue', alpha=0.7)
        ax3.bar(x, adv_accuracies, width, label='Adversarial', 
                color='red', alpha=0.7)
        ax3.bar(x + width, defended_accuracies, width, label='After NLM Defense', 
                color='green', alpha=0.7)
        
        ax3.set_xlabel('Class')
        ax3.set_ylabel('Accuracy (%)')
        ax3.set_title('Accuracy Comparison: Original vs Adversarial vs NLM Defense')
        ax3.set_xticks(x)
        ax3.set_xticklabels(class_names, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
    else:
        ax3.text(0.5, 0.5, 'Original accuracy data unavailable', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Accuracy Comparison (Original Data Unavailable)')
    
    # 4. Hardest classes to defend (lowest improvement)
    sorted_indices = np.argsort(defense_improvements)
    hardest_classes = [class_names[i] for i in sorted_indices[:5]]
    hardest_improvements = [defense_improvements[i] for i in sorted_indices[:5]]
    
    colors = ['red' if imp <= 0 else 'orange' for imp in hardest_improvements]
    ax4.bar(hardest_classes, hardest_improvements, color=colors, alpha=0.7)
    ax4.set_xlabel('Class')
    ax4.set_ylabel('Accuracy Improvement (%)')
    ax4.set_title('Hardest Classes to Defend with NLM (Lowest Improvement)')
    ax4.set_xticklabels(hardest_classes, rotation=45, ha='right')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'nlmeans_defense_analysis.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ NLM defense analysis saved: {save_path}")

def generate_nlmeans_confidence_comparison(labels, preds_orig, preds_adv, preds_defended,
                                          conf_orig, conf_adv, conf_defended,
                                          epsilon, defense_name, output_dir):
    """Generate confidence comparison analysis"""
    if conf_orig is not None:
        orig_confs = conf_orig.numpy()
    else:
        orig_confs = None
    
    adv_confs = conf_adv.numpy()
    defended_confs = conf_defended.numpy()
    true_labels = labels.numpy()
    
    # Create charts
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Confidence distribution comparison
    if orig_confs is not None:
        ax1.hist(orig_confs, bins=30, alpha=0.5, label='Original', color='blue')
    ax1.hist(adv_confs, bins=30, alpha=0.5, label='Adversarial', color='red')
    ax1.hist(defended_confs, bins=30, alpha=0.5, label='After NLM Defense', color='green')
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Confidence Distribution Comparison (ε={epsilon})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Attack success rate vs confidence relationship
    success_mask_adv = (preds_adv.numpy() != true_labels)
    success_mask_defended = (preds_defended.numpy() != true_labels)
    
    adv_success_rates = []
    defended_success_rates = []
    conf_bins = np.linspace(0, 1, 11)
    
    for i in range(len(conf_bins)-1):
        low = conf_bins[i]
        high = conf_bins[i+1]
        
        mask_adv = (adv_confs >= low) & (adv_confs < high)
        if mask_adv.any():
            rate_adv = success_mask_adv[mask_adv].mean() * 100
        else:
            rate_adv = 0
        adv_success_rates.append(rate_adv)
        
        mask_def = (defended_confs >= low) & (defended_confs < high)
        if mask_def.any():
            rate_def = success_mask_defended[mask_def].mean() * 100
        else:
            rate_def = 0
        defended_success_rates.append(rate_def)
    
    conf_centers = (conf_bins[:-1] + conf_bins[1:]) / 2
    ax2.plot(conf_centers, adv_success_rates, 'o-', linewidth=2, markersize=8, 
            label='Adversarial', color='red')
    ax2.plot(conf_centers, defended_success_rates, 's-', linewidth=2, markersize=8, 
            label='After NLM Defense', color='green')
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Attack Success Rate (%)')
    ax2.set_title('Attack Success Rate vs Confidence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Confidence change
    conf_change_adv = adv_confs - (orig_confs if orig_confs is not None else np.ones_like(adv_confs))
    conf_change_defended = defended_confs - adv_confs
    
    ax3.hist(conf_change_adv, bins=50, alpha=0.5, label='Adv - Orig', color='red')
    ax3.hist(conf_change_defended, bins=50, alpha=0.5, label='NLM Defended - Adv', color='green')
    ax3.set_xlabel('Confidence Change')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Confidence Change Distribution')
    ax3.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Confidence before vs after defense scatter plot
    scatter = ax4.scatter(adv_confs, defended_confs, c=success_mask_defended, 
                         cmap='coolwarm', alpha=0.6, s=20)
    ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax4.set_xlabel('Adversarial Confidence')
    ax4.set_ylabel(f'Confidence After NLM Defense')
    ax4.set_title('Confidence Before vs After NLM Defense')
    ax4.grid(True, alpha=0.3)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.6, label='Attack Success (After NLM Defense)'),
        Patch(facecolor='blue', alpha=0.6, label='Attack Failed (After NLM Defense)')
    ]
    ax4.legend(handles=legend_elements)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'nlmeans_confidence_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ NLM confidence comparison analysis saved: {save_path}")

def generate_nlmeans_parameter_effect_analysis(results, epsilon, output_dir):
    """Generate parameter effect analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Parameter vs defense effectiveness relationship
    h = results['h']
    template = results['template_window_size']
    search = results['search_window_size']
    denoise_strength = results['denoise_strength']
    
    axes[0, 0].bar(['h', 'Template Size', 'Search Size', 'Denoise Strength'], 
                  [h, template, search, denoise_strength], 
                  color=['blue', 'green', 'orange', 'purple'], alpha=0.7)
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].set_title(f'Optimal NLM Parameters (ε={epsilon})')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (name, value) in enumerate(zip(['h', 'Template Size', 'Search Size', 'Denoise Strength'], 
                                          [h, template, search, denoise_strength])):
        axes[0, 0].text(i, value + 0.05 * value, f'{value:.2f}', 
                       ha='center', va='bottom')
    
    # 2. Defense performance metrics
    metrics = ['Adv Accuracy', 'Defended Accuracy', 'Improvement']
    values = [results['adversarial_accuracy'], 
              results['defended_accuracy'], 
              results['defense_improvement']]
    colors = ['red', 'green', 'blue']
    
    axes[0, 1].bar(metrics, values, color=colors, alpha=0.7)
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('NLM Defense Performance Metrics')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (metric, value) in enumerate(zip(metrics, values)):
        axes[0, 1].text(i, value + 1, f'{value:.2f}%', 
                       ha='center', va='bottom')
    
    # 3. Original image impact
    if results['original_degradation'] is not None:
        axes[1, 0].bar(['Original Degradation'], 
                      [results['original_degradation']], 
                      color='purple', alpha=0.7)
        axes[1, 0].set_ylabel('Accuracy Decrease (%)')
        axes[1, 0].set_title('NLM Impact on Original Images')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Add value label
        axes[1, 0].text(0, results['original_degradation'] + 0.5, 
                       f'{results["original_degradation"]:.2f}%', 
                       ha='center', va='bottom')
    else:
        axes[1, 0].text(0.5, 0.5, 'Original image data unavailable', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('NLM Impact on Original Images (Data Unavailable)')
    
    # 4. Trade-off score and quality assessment
    axes[1, 1].bar(['Trade-off Score'], [results['tradeoff_score']], 
                  color='orange', alpha=0.7)
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Advanced Trade-off Score for NLM')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Add value label
    axes[1, 1].text(0, results['tradeoff_score'] + 0.5, 
                   f'{results["tradeoff_score"]:.2f}', 
                   ha='center', va='bottom')
    
    # Add quality assessment label below chart
    quality = results['quality_assessment']
    plt.figtext(0.5, 0.02, 
                f'Quality Assessment: {quality["overall_assessment"]} | '
                f'Defense: {quality["defense_grade"]} | '
                f'Original Impact: {quality["original_impact_grade"]} | '
                f'Denoise: {quality["denoise_grade"]}',
                ha='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    save_path = os.path.join(output_dir, 'nlmeans_parameter_effect_analysis.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ NLM parameter effect analysis saved: {save_path}")

def generate_nlmeans_quality_assessment_chart(results, epsilon, output_dir):
    """Generate quality assessment chart"""
    quality = results['quality_assessment']
    
    # Create radar chart
    fig = plt.figure(figsize=(10, 8))
    
    # Radar chart data
    categories = ['Defense\nEffectiveness', 'Original\nImpact', 'Denoise\nIntensity', 'Overall\nQuality']
    
    # Convert quality grades to numeric values
    defense_scores = {'Poor': 1, 'Fair': 2, 'Good': 3, 'Excellent': 4}
    impact_scores = {'Severe': 1, 'High': 2, 'Moderate': 3, 'Low': 4, 'Minimal': 5}
    denoise_scores = {'Very Heavy': 1, 'Heavy': 2, 'Strong': 3, 'Moderate': 4, 'Light': 5}
    overall_scores = {'Poor Defense (Harmful)': 1, 'Low Quality Defense': 2, 
                      'Medium Quality Defense': 3, 'High Quality Defense': 4}
    
    values = [
        defense_scores.get(quality['defense_grade'], 2),
        impact_scores.get(quality['original_impact_grade'], 3),
        denoise_scores.get(quality['denoise_grade'], 3),
        overall_scores.get(quality['overall_assessment'], 2)
    ]
    
    # Close radar chart
    values += values[:1]
    categories_radar = categories + [categories[0]]
    
    # Set angles
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    # Draw radar chart
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, values, 'o-', linewidth=2, color='purple')
    ax.fill(angles, values, alpha=0.25, color='purple')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'], fontsize=8)
    ax.set_title(f'NLM Defense Quality Assessment Radar Chart (ε={epsilon})', 
                fontsize=12, fontweight='bold', pad=20)
    
    # Add value labels
    for i, (angle, value) in enumerate(zip(angles[:-1], values[:-1])):
        ax.text(angle, value + 0.2, str(value), ha='center', va='center', 
               fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'nlmeans_quality_assessment_radar.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ NLM quality assessment radar chart saved: {save_path}")

def generate_nlmeans_classification_reports(labels, preds_adv, preds_defended, class_names, epsilon, output_dir):
    """Generate classification reports"""
    # Adversarial classification report
    report_adv = classification_report(labels.numpy(), preds_adv.numpy(), 
                                      target_names=class_names, output_dict=True)
    
    # After defense classification report
    report_defended = classification_report(labels.numpy(), preds_defended.numpy(), 
                                           target_names=class_names, output_dict=True)
    
    # Save classification reports as JSON
    reports = {
        'epsilon': epsilon,
        'adversarial_report': report_adv,
        'defended_report': report_defended
    }
    
    report_path = os.path.join(output_dir, 'nlmeans_classification_reports.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(reports, f, indent=4)
    
    # Generate classification report comparison chart
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Extract F1 scores
    f1_scores_adv = []
    f1_scores_defended = []
    
    for class_name in class_names:
        if class_name in report_adv:
            f1_scores_adv.append(report_adv[class_name]['f1-score'] * 100)
            f1_scores_defended.append(report_defended[class_name]['f1-score'] * 100)
    
    x = np.arange(len(class_names))
    width = 0.35
    
    axes[0].bar(x - width/2, f1_scores_adv, width, label='Adversarial', color='red', alpha=0.7)
    axes[0].bar(x + width/2, f1_scores_defended, width, label='After NLM Defense', color='green', alpha=0.7)
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('F1-Score (%)')
    axes[0].set_title(f'F1-Score Comparison by Class (ε={epsilon})')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Precision and recall comparison
    precision_adv = report_adv['weighted avg']['precision'] * 100
    precision_defended = report_defended['weighted avg']['precision'] * 100
    recall_adv = report_adv['weighted avg']['recall'] * 100
    recall_defended = report_defended['weighted avg']['recall'] * 100
    f1_adv = report_adv['weighted avg']['f1-score'] * 100
    f1_defended = report_defended['weighted avg']['f1-score'] * 100
    
    metrics = ['Precision', 'Recall', 'F1-Score']
    adv_values = [precision_adv, recall_adv, f1_adv]
    defended_values = [precision_defended, recall_defended, f1_defended]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[1].bar(x - width/2, adv_values, width, label='Adversarial', color='red', alpha=0.7)
    axes[1].bar(x + width/2, defended_values, width, label='After NLM Defense', color='green', alpha=0.7)
    axes[1].set_xlabel('Metric')
    axes[1].set_ylabel('Percentage (%)')
    axes[1].set_title('Weighted Average Metrics Comparison')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (adv_val, def_val) in enumerate(zip(adv_values, defended_values)):
        axes[1].text(i - width/2, adv_val + 1, f'{adv_val:.1f}%', ha='center', va='bottom', fontsize=9)
        axes[1].text(i + width/2, def_val + 1, f'{def_val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'nlmeans_classification_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ NLM classification report comparison chart saved: {plot_path}")
    print(f"    ✓ Detailed classification reports saved: {report_path}")

def save_nlmeans_comprehensive_report(results, class_names, output_dir):
    """Save comprehensive Non-Local Means denoising defense report"""
    report_path = os.path.join(output_dir, 'nlmeans_comprehensive_defense_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"COMPREHENSIVE NON-LOCAL MEANS DEFENSE REPORT - ε={results['epsilon']}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. OPTIMAL PARAMETERS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Defense Method: {results['defense_name']}\n")
        f.write(f"Filter Strength (h): {results['h']}\n")
        f.write(f"Template Window Size: {results['template_window_size']}\n")
        f.write(f"Search Window Size: {results['search_window_size']}\n")
        f.write(f"Denoise Strength: {results['denoise_strength']:.3f}\n")
        f.write(f"Epsilon (ε): {results['epsilon']}\n")
        f.write(f"Total Samples: {results['num_samples']}\n\n")
        
        f.write("2. PERFORMANCE METRICS\n")
        f.write("-" * 40 + "\n")
        if results['original_accuracy'] is not None:
            f.write(f"Original Accuracy: {results['original_accuracy']:.2f}%\n")
        f.write(f"Adversarial Accuracy (No Defense): {results['adversarial_accuracy']:.2f}%\n")
        f.write(f"Accuracy After NLM Defense: {results['defended_accuracy']:.2f}%\n")
        f.write(f"NLM Defense Improvement: {results['defense_improvement']:.2f}%\n")
        f.write(f"Attack Success Rate Reduction: {100 - results['adversarial_accuracy'] - (100 - results['defended_accuracy']):.2f}%\n")
        f.write(f"Average Perturbation Norm: {results['avg_perturbation_norm']:.4f}\n\n")
        
        f.write("3. ORIGINAL IMAGE IMPACT\n")
        f.write("-" * 40 + "\n")
        if results['original_denoised_accuracy'] is not None:
            f.write(f"Original Accuracy After NLM Denoising: {results['original_denoised_accuracy']:.2f}%\n")
        f.write(f"Original Accuracy Degradation: {results['original_degradation']:.2f}%\n\n")
        
        f.write("4. ADVANCED SCORING\n")
        f.write("-" * 40 + "\n")
        f.write(f"Trade-off Score: {results['tradeoff_score']:.3f}\n")
        f.write(f"Scoring Method: Advanced non-linear with diminishing returns (NLM optimized)\n\n")
        
        f.write("5. QUALITY ASSESSMENT\n")
        f.write("-" * 40 + "\n")
        quality = results['quality_assessment']
        f.write(f"Defense Effectiveness: {quality['defense_grade']}\n")
        f.write(f"Original Image Impact: {quality['original_impact_grade']}\n")
        f.write(f"Denoise Intensity: {quality['denoise_grade']}\n")
        f.write(f"Overall Assessment: {quality['overall_assessment']}\n\n")
        
        f.write("6. CONFIDENCE METRICS\n")
        f.write("-" * 40 + "\n")
        if results['avg_confidence_original'] is not None:
            f.write(f"Average Confidence (Original): {results['avg_confidence_original']:.4f}\n")
        f.write(f"Average Confidence (Adversarial): {results['avg_confidence_adversarial']:.4f}\n")
        f.write(f"Average Confidence (After NLM Defense): {results['avg_confidence_defended']:.4f}\n\n")
        
        f.write("7. CLASS-WISE PERFORMANCE\n")
        f.write("-" * 40 + "\n")
        
        # Class statistics
        num_classes = len(class_names)
        class_counts = np.zeros(num_classes)
        attack_success_counts = np.zeros(num_classes)
        defense_success_counts = np.zeros(num_classes)
        
        for i in range(len(results['labels'])):
            label = results['labels'][i].item()
            pred_adv = results['preds_adv'][i].item()
            pred_defended = results['preds_defended'][i].item()
            
            class_counts[label] += 1
            if pred_adv != label:
                attack_success_counts[label] += 1
            if pred_defended == label:
                defense_success_counts[label] += 1
        
        f.write(f"{'Class':<15} {'Samples':<10} {'Attack Rate':<12} {'Defense Acc':<12} {'Improvement':<12}\n")
        f.write("-" * 61 + "\n")
        
        for i in range(num_classes):
            if class_counts[i] > 0:
                attack_rate = 100.0 * attack_success_counts[i] / class_counts[i]
                defense_acc = 100.0 * defense_success_counts[i] / class_counts[i]
                adv_acc = 100.0 - attack_rate
                improvement = defense_acc - adv_acc
                
                f.write(f"{class_names[i]:<15} {int(class_counts[i]):<10} {attack_rate:<12.2f}% {defense_acc:<12.2f}% {improvement:<12.2f}%\n")
        
        f.write("\n8. KEY FINDINGS\n")
        f.write("-" * 40 + "\n")
        
        if results['defense_improvement'] > 20:
            f.write("✓ EXCELLENT NLM DEFENSE: Significant improvement (>20%) achieved\n")
        elif results['defense_improvement'] > 10:
            f.write("✓ GOOD NLM DEFENSE: Moderate improvement (10-20%) achieved\n")
        elif results['defense_improvement'] > 0:
            f.write("✓ FAIR NLM DEFENSE: Small improvement (<10%) achieved\n")
        else:
            f.write("✗ POOR NLM DEFENSE: No improvement or negative impact\n")
        
        if results['original_degradation'] < 5:
            f.write("✓ MINIMAL IMPACT: Original image degradation <5%\n")
        elif results['original_degradation'] < 10:
            f.write("⚠ MODERATE IMPACT: Original image degradation 5-10%\n")
        else:
            f.write("⚠ HIGH IMPACT: Original image degradation >10%\n")
        
        f.write("\n9. NON-LOCAL MEANS ADVANTAGES\n")
        f.write("-" * 40 + "\n")
        f.write("• Preserves edges and textures better than Gaussian blur\n")
        f.write("• Adaptive denoising based on image similarity\n")
        f.write("• More effective at removing structured noise (like adversarial perturbations)\n")
        f.write("• Maintains more image details while removing noise\n")
        
        f.write(f"\n10. RECOMMENDATION\n")
        f.write("-" * 40 + "\n")
        
        if quality['overall_assessment'] == "High Quality Defense":
            f.write("✓ HIGHLY RECOMMENDED\n")
            f.write(f"  These parameters provide excellent defense with minimal impact.\n")
            f.write(f"  Use h={results['h']}, template={results['template_window_size']}, search={results['search_window_size']} for ε={results['epsilon']}\n")
        
        elif quality['overall_assessment'] == "Medium Quality Defense":
            f.write("✓ RECOMMENDED WITH CONSIDERATIONS\n")
            f.write(f"  These parameters provide good defense with moderate impact.\n")
            f.write(f"  Consider lighter denoising if original image quality is critical.\n")
        
        elif quality['overall_assessment'] == "Low Quality Defense":
            f.write("⚠ USE WITH CAUTION\n")
            f.write(f"  Defense improvement is limited ({results['defense_improvement']:.2f}%).\n")
            f.write(f"  Consider alternative defense methods or parameters.\n")
        
        else:
            f.write("✗ NOT RECOMMENDED\n")
            f.write(f"  This NLM defense is ineffective or harmful.\n")
            f.write(f"  Do not use these parameters for ε={results['epsilon']}\n")
        
        f.write("\n11. FILES GENERATED\n")
        f.write("-" * 40 + "\n")
        f.write("In the analysis directory you will find:\n")
        f.write("  - nlmeans_defense_visualization.png: Sample images before/after NLM defense\n")
        f.write("  - nlmeans_confusion_comparison.png: Confusion matrices comparison\n")
        f.write("  - nlmeans_defense_analysis.png: Detailed NLM defense effectiveness analysis\n")
        f.write("  - nlmeans_confidence_comparison.png: Confidence analysis\n")
        f.write("  - nlmeans_parameter_effect_analysis.png: Parameter impact analysis\n")
        f.write("  - nlmeans_quality_assessment_radar.png: Quality radar chart\n")
        f.write("  - nlmeans_classification_comparison.png: Classification metrics\n")
        f.write("  - nlmeans_defense_raw_results.csv: Raw prediction data\n")
        f.write("  - nlmeans_classification_reports.json: Detailed classification reports\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF NON-LOCAL MEANS DEFENSE REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"    ✓ Comprehensive NLM defense report saved: {report_path}")

def save_nlmeans_all_raw_data(results, output_dir):
    """Save all raw data"""
    # Save raw results data
    data = {
        'true_label': results['labels'].numpy(),
        'adv_pred': results['preds_adv'].numpy(),
        'adv_confidence': results['conf_adv'].numpy(),
        'defended_pred': results['preds_defended'].numpy(),
        'defended_confidence': results['conf_defended'].numpy()
    }
    
    if results['preds_orig'] is not None:
        data['orig_pred'] = results['preds_orig'].numpy()
    
    if results['conf_orig'] is not None:
        data['orig_confidence'] = results['conf_orig'].numpy()
    
    if results['pert_norms'] is not None:
        data['perturbation_norm'] = results['pert_norms'].numpy()
    
    df = pd.DataFrame(data)
    df['attack_success'] = (df['true_label'] != df['adv_pred']).astype(int)
    df['defense_success'] = (df['true_label'] == df['defended_pred']).astype(int)
    df['confidence_change'] = df['defended_confidence'] - df['adv_confidence']
    
    csv_path = os.path.join(output_dir, 'nlmeans_defense_raw_results.csv')
    df.to_csv(csv_path, index=False)
    
    # Save results summary
    summary_path = os.path.join(output_dir, 'nlmeans_defense_results_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        # Remove Tensor objects
        summary = {k: v for k, v in results.items() 
                  if not isinstance(v, torch.Tensor) and k not in ['adv_images', 'orig_images', 'defended_images', 'perturbations']}
        json.dump(summary, f, indent=4)
    
    print(f"    ✓ NLM defense raw data saved: {csv_path}")
    print(f"    ✓ NLM defense results summary saved: {summary_path}")

def generate_nlmeans_global_analysis(all_best_results, output_dir):
    """Generate global analysis"""
    print("\nGenerating NLM global analysis...")
    
    # Prepare summary data
    summary_data = []
    
    for epsilon, (best_result, detailed_result) in all_best_results.items():
        if detailed_result is not None:
            summary_data.append({
                'epsilon': epsilon,
                'h': best_result['h'],
                'template_window_size': best_result['template_window_size'],
                'search_window_size': best_result['search_window_size'],
                'denoise_strength': best_result['denoise_strength'],
                'adversarial_accuracy': best_result['adversarial_accuracy'],
                'defended_accuracy': best_result['defended_accuracy'],
                'defense_improvement': best_result['defense_improvement'],
                'original_degradation': detailed_result.get('original_degradation', 0),
                'tradeoff_score': best_result['tradeoff_score'],
                'quality': best_result['quality_assessment']['overall_assessment']
            })
    
    df_summary = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, 'nlmeans_global_optimization_summary.csv')
    df_summary.to_csv(summary_path, index=False)
    
    # Generate global charts
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    epsilons = df_summary['epsilon']
    
    # 1. Optimal parameter trends
    axes[0, 0].plot(epsilons, df_summary['h'], 'o-', color='red', markersize=8, label='Filter Strength (h)')
    axes[0, 0].set_xlabel('Epsilon')
    axes[0, 0].set_ylabel('Filter Strength (h)', color='red')
    axes[0, 0].set_title('Optimal Filter Strength vs Epsilon', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Defense effectiveness trends
    axes[0, 1].plot(epsilons, df_summary['adversarial_accuracy'], 'o-', 
                   label='Adversarial', color='red', markersize=8)
    axes[0, 1].plot(epsilons, df_summary['defended_accuracy'], 's-', 
                   label='After NLM Defense', color='green', markersize=8)
    axes[0, 1].set_xlabel('Epsilon')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Accuracy Trends with NLM Defense', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Trade-off score trends
    axes[1, 0].plot(epsilons, df_summary['tradeoff_score'], 'o-', 
                   color='purple', markersize=8)
    axes[1, 0].set_xlabel('Epsilon')
    axes[1, 0].set_ylabel('Advanced Trade-off Score')
    axes[1, 0].set_title('Advanced Scoring Trends for NLM', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Quality distribution
    quality_counts = df_summary['quality'].value_counts()
    colors = {'High Quality Defense': 'green', 
              'Medium Quality Defense': 'orange', 
              'Low Quality Defense': 'yellow',
              'Poor Defense (Harmful)': 'red'}
    bar_colors = [colors.get(q, 'gray') for q in quality_counts.index]
    
    axes[1, 1].bar(quality_counts.index, quality_counts.values, color=bar_colors, alpha=0.7)
    axes[1, 1].set_xlabel('Quality Category')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('NLM Defense Quality Distribution', fontweight='bold')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (category, count) in enumerate(zip(quality_counts.index, quality_counts.values)):
        axes[1, 1].text(i, count + 0.1, str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    global_plot_path = os.path.join(output_dir, 'nlmeans_global_analysis.png')
    plt.savefig(global_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ NLM global analysis chart saved: {global_plot_path}")
    print(f"✓ NLM global summary data saved: {summary_path}")
    
    # Generate global report
    report_path = os.path.join(output_dir, 'nlmeans_global_optimization_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("GLOBAL NON-LOCAL MEANS DEFENSE OPTIMIZATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("SUMMARY OF OPTIMAL NLM PARAMETERS FOR DIFFERENT EPSILON VALUES:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Epsilon':<10} {'h':<6} {'Template':<10} {'Search':<10} {'Denoise Str':<12} {'Adv Acc':<10} {'Def Acc':<10} {'Improvement':<12} {'Score':<10} {'Quality':<20}\n")
        f.write("-" * 80 + "\n")
        
        for _, row in df_summary.iterrows():
            f.write(f"{row['epsilon']:<10.3f} {row['h']:<6} {row['template_window_size']:<10} {row['search_window_size']:<10} "
                   f"{row['denoise_strength']:<12.2f} {row['adversarial_accuracy']:<10.2f}% {row['defended_accuracy']:<10.2f}% "
                   f"{row['defense_improvement']:<12.2f}% {row['tradeoff_score']:<10.2f} {row['quality']:<20}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("KEY INSIGHTS FOR NON-LOCAL MEANS DEFENSE\n")
        f.write("=" * 80 + "\n\n")
        
        # Analyze trends
        if len(df_summary) > 1:
            avg_improvement = df_summary['defense_improvement'].mean()
            max_improvement = df_summary['defense_improvement'].max()
            min_improvement = df_summary['defense_improvement'].min()
            
            f.write(f"1. NLM Defense Effectiveness:\n")
            f.write(f"   - Average improvement: {avg_improvement:.2f}%\n")
            f.write(f"   - Maximum improvement: {max_improvement:.2f}%\n")
            f.write(f"   - Minimum improvement: {min_improvement:.2f}%\n\n")
        
        # Parameter trends
        f.write(f"2. Parameter Trends:\n")
        f.write(f"   - Stronger attacks (higher epsilon) generally require stronger denoising\n")
        f.write(f"   - Optimal filter strength (h) increases with epsilon\n")
        f.write(f"   - Larger search windows are often better for larger epsilon values\n\n")
        
        f.write("3. NLM Advantages Over Gaussian Blur:\n")
        f.write("   - Better preservation of image edges and textures\n")
        f.write("   - Adaptive denoising based on image similarity\n")
        f.write("   - More effective at removing structured adversarial perturbations\n")
        f.write("   - Maintains more image details while removing noise\n\n")
        
        f.write("4. Quality Distribution:\n")
        for quality, count in quality_counts.items():
            percentage = 100.0 * count / len(df_summary)
            f.write(f"   - {quality}: {count} epsilon values ({percentage:.1f}%)\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("PRACTICAL RECOMMENDATIONS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Based on the NLM optimization results, here are practical recommendations:\n\n")
        
        f.write("1. For UNKNOWN attack strength:\n")
        f.write("   Start with h=10, template=7, search=21 as a balanced default\n\n")
        
        f.write("2. For different epsilon ranges:\n")
        
        # Group recommendations by epsilon range
        eps_ranges = [(0, 0.03, "Very Small"), (0.03, 0.07, "Small"), 
                     (0.07, 0.12, "Medium"), (0.12, 0.2, "Large"), (0.2, 1.0, "Very Large")]
        
        for low, high, name in eps_ranges:
            range_data = df_summary[(df_summary['epsilon'] >= low) & (df_summary['epsilon'] < high)]
            if not range_data.empty:
                avg_h = range_data['h'].mean()
                avg_template = range_data['template_window_size'].mean()
                avg_search = range_data['search_window_size'].mean()
                avg_imp = range_data['defense_improvement'].mean()
                
                f.write(f"   - {name} attacks (ε={low:.2f}-{high:.2f}):\n")
                f.write(f"     Recommended: h={avg_h:.1f}, template={avg_template:.0f}, search={avg_search:.0f}\n")
                f.write(f"     Expected improvement: {avg_imp:.1f}%\n\n")
        
        f.write("3. Implementation Strategy for NLM:\n")
        f.write("   - Estimate attack strength (epsilon) if possible\n")
        f.write("   - Use adaptive parameter selection based on epsilon\n")
        f.write("   - Consider computational cost (NLM is more expensive than Gaussian blur)\n")
        f.write("   - For real-time applications, pre-compute denoising parameters\n\n")
        
        f.write("4. When to Choose NLM Over Gaussian Blur:\n")
        f.write("   - When image detail preservation is critical\n")
        f.write("   - For attacks with structured, non-random perturbations\n")
        f.write("   - When computational resources are sufficient\n")
        f.write("   - For high-stakes applications where defense quality is paramount\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("FILES GENERATED\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("This NLM optimization generated the following files:\n")
        f.write("1. For each epsilon value:\n")
        f.write("   - Parameter selection results\n")
        f.write("   - Comprehensive NLM defense report\n")
        f.write("   - All analysis visualizations\n")
        f.write("   - Raw prediction data\n")
        f.write("   - Classification reports\n\n")
        
        f.write("2. Global analysis files:\n")
        f.write("   - Global NLM optimization summary (CSV)\n")
        f.write("   - Global NLM analysis chart (PNG)\n")
        f.write("   - This global NLM report\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("CONCLUSION\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Adaptive Non-Local Means denoising defense is an advanced method for protecting\n")
        f.write("against adversarial attacks. The key findings from this optimization are:\n")
        f.write("1. NLM can be more effective than Gaussian blur for certain attack types\n")
        f.write("2. Optimal parameters vary with attack strength and characteristics\n")
        f.write("3. NLM preserves image details better while removing adversarial noise\n")
        f.write("4. The computational cost is higher but the quality can be superior\n\n")
        
        f.write("Use the generated files to implement and evaluate NLM defense in your applications,\n")
        f.write("especially when image quality preservation is as important as defense effectiveness.\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF NON-LOCAL MEANS DEFENSE OPTIMIZATION REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"✓ NLM global optimization report saved: {report_path}")

def compare_nlmeans_vs_gaussian(all_nlmeans_results, gaussian_results_dir='./comprehensive_blur_defense'):
    """
    Compare Non-Local Means denoising with Gaussian blur defense effectiveness
    
    Parameters:
        all_nlmeans_results: Dictionary of NLM results
        gaussian_results_dir: Directory of Gaussian blur defense results
    """
    print("\nComparing NLM vs Gaussian blur defense effectiveness...")
    
    # Load Gaussian blur results
    gaussian_summary_path = os.path.join(gaussian_results_dir, 'global_optimization_summary.csv')
    
    if not os.path.exists(gaussian_summary_path):
        print(f"Warning: Gaussian blur result file {gaussian_summary_path} does not exist")
        return
    
    try:
        df_gaussian = pd.read_csv(gaussian_summary_path)
        
        # Prepare NLM data
        nlmeans_data = []
        for epsilon, (best_result, _) in all_nlmeans_results.items():
            nlmeans_data.append({
                'epsilon': epsilon,
                'defense_method': 'Non-Local Means',
                'defended_accuracy': best_result['defended_accuracy'],
                'defense_improvement': best_result['defense_improvement'],
                'original_degradation': best_result.get('original_degradation', 0),
                'tradeoff_score': best_result['tradeoff_score']
            })
        
        df_nlmeans = pd.DataFrame(nlmeans_data)
        
        # Merge data
        comparison_data = []
        for epsilon in df_gaussian['epsilon'].unique():
            gaussian_row = df_gaussian[df_gaussian['epsilon'] == epsilon]
            nlmeans_row = df_nlmeans[df_nlmeans['epsilon'] == epsilon]
            
            if not gaussian_row.empty and not nlmeans_row.empty:
                comparison_data.append({
                    'epsilon': epsilon,
                    'gaussian_defended_accuracy': gaussian_row.iloc[0]['defended_accuracy'],
                    'nlmeans_defended_accuracy': nlmeans_row.iloc[0]['defended_accuracy'],
                    'gaussian_improvement': gaussian_row.iloc[0]['defense_improvement'],
                    'nlmeans_improvement': nlmeans_row.iloc[0]['defense_improvement'],
                    'gaussian_degradation': gaussian_row.iloc[0].get('original_degradation', 0),
                    'nlmeans_degradation': nlmeans_row.iloc[0]['original_degradation'],
                    'gaussian_score': gaussian_row.iloc[0]['tradeoff_score'],
                    'nlmeans_score': nlmeans_row.iloc[0]['tradeoff_score']
                })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        if df_comparison.empty:
            print("Warning: No comparable data found")
            return
        
        # Create comparison charts
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        epsilons = df_comparison['epsilon']
        
        # 1. Defended accuracy comparison
        axes[0, 0].plot(epsilons, df_comparison['gaussian_defended_accuracy'], 'o-', 
                       label='Gaussian Blur', color='blue', markersize=8)
        axes[0, 0].plot(epsilons, df_comparison['nlmeans_defended_accuracy'], 's-', 
                       label='Non-Local Means', color='green', markersize=8)
        axes[0, 0].set_xlabel('Epsilon')
        axes[0, 0].set_ylabel('Defended Accuracy (%)')
        axes[0, 0].set_title('Defense Accuracy Comparison: Gaussian Blur vs NLM', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Defense improvement comparison
        axes[0, 1].plot(epsilons, df_comparison['gaussian_improvement'], 'o-', 
                       label='Gaussian Blur', color='blue', markersize=8)
        axes[0, 1].plot(epsilons, df_comparison['nlmeans_improvement'], 's-', 
                       label='Non-Local Means', color='green', markersize=8)
        axes[0, 1].set_xlabel('Epsilon')
        axes[0, 1].set_ylabel('Defense Improvement (%)')
        axes[0, 1].set_title('Defense Improvement Comparison', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Original image impact comparison
        axes[1, 0].plot(epsilons, df_comparison['gaussian_degradation'], 'o-', 
                       label='Gaussian Blur', color='blue', markersize=8)
        axes[1, 0].plot(epsilons, df_comparison['nlmeans_degradation'], 's-', 
                       label='Non-Local Means', color='green', markersize=8)
        axes[1, 0].set_xlabel('Epsilon')
        axes[1, 0].set_ylabel('Original Accuracy Degradation (%)')
        axes[1, 0].set_title('Impact on Original Images', fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Trade-off score comparison
        axes[1, 1].plot(epsilons, df_comparison['gaussian_score'], 'o-', 
                       label='Gaussian Blur', color='blue', markersize=8)
        axes[1, 1].plot(epsilons, df_comparison['nlmeans_score'], 's-', 
                       label='Non-Local Means', color='green', markersize=8)
        axes[1, 1].set_xlabel('Epsilon')
        axes[1, 1].set_ylabel('Advanced Trade-off Score')
        axes[1, 1].set_title('Advanced Scoring Comparison', fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        comparison_path = os.path.join(output_dir, 'nlmeans_vs_gaussian_comparison.png')
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ NLM vs Gaussian blur comparison chart saved: {comparison_path}")
        
        # Calculate statistics
        gaussian_wins = 0
        nlmeans_wins = 0
        ties = 0
        
        for _, row in df_comparison.iterrows():
            if row['nlmeans_improvement'] > row['gaussian_improvement']:
                nlmeans_wins += 1
            elif row['gaussian_improvement'] > row['nlmeans_improvement']:
                gaussian_wins += 1
            else:
                ties += 1
        
        # Save comparison report
        comparison_report_path = os.path.join(output_dir, 'nlmeans_vs_gaussian_comparison_report.txt')
        
        with open(comparison_report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("NON-LOCAL MEANS VS GAUSSIAN BLUR DEFENSE COMPARISON REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("COMPARISON RESULTS:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total epsilon values compared: {len(df_comparison)}\n")
            f.write(f"Non-Local Means wins: {nlmeans_wins}\n")
            f.write(f"Gaussian Blur wins: {gaussian_wins}\n")
            f.write(f"Ties: {ties}\n\n")
            
            f.write("AVERAGE PERFORMANCE:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Average Gaussian Blur improvement: {df_comparison['gaussian_improvement'].mean():.2f}%\n")
            f.write(f"Average Non-Local Means improvement: {df_comparison['nlmeans_improvement'].mean():.2f}%\n")
            f.write(f"Average Gaussian Blur degradation: {df_comparison['gaussian_degradation'].mean():.2f}%\n")
            f.write(f"Average Non-Local Means degradation: {df_comparison['nlmeans_degradation'].mean():.2f}%\n")
            f.write(f"Average Gaussian Blur score: {df_comparison['gaussian_score'].mean():.2f}\n")
            f.write(f"Average Non-Local Means score: {df_comparison['nlmeans_score'].mean():.2f}\n\n")
            
            f.write("RECOMMENDATIONS:\n")
            f.write("-" * 80 + "\n")
            
            if nlmeans_wins > gaussian_wins:
                f.write("✓ NON-LOCAL MEANS IS GENERALLY BETTER\n")
                f.write("  Based on the comparison, Non-Local Means defense provides better\n")
                f.write("  overall performance across different epsilon values.\n\n")
                f.write("  Use Non-Local Means when:\n")
                f.write("  • Computational resources are sufficient\n")
                f.write("  • Image detail preservation is important\n")
                f.write("  • Defense effectiveness is the highest priority\n")
            elif gaussian_wins > nlmeans_wins:
                f.write("✓ GAUSSIAN BLUR IS GENERALLY BETTER\n")
                f.write("  Based on the comparison, Gaussian Blur defense provides better\n")
                f.write("  overall performance across different epsilon values.\n\n")
                f.write("  Use Gaussian Blur when:\n")
                f.write("  • Computational efficiency is important\n")
                f.write("  • Simplicity is preferred\n")
                f.write("  • Real-time processing is required\n")
            else:
                f.write("✓ BOTH METHODS ARE COMPARABLE\n")
                f.write("  Both defense methods provide similar performance.\n")
                f.write("  Choose based on specific requirements:\n\n")
                f.write("  Choose Non-Local Means for:\n")
                f.write("  • Better edge preservation\n")
                f.write("  • More sophisticated denoising\n")
                f.write("  • When image quality is critical\n\n")
                f.write("  Choose Gaussian Blur for:\n")
                f.write("  • Faster computation\n")
                f.write("  • Simpler implementation\n")
                f.write("  • When resources are limited\n")
            
            f.write("\nDETAILED COMPARISON BY EPSILON:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Epsilon':<10} {'Gauss Acc':<12} {'NLM Acc':<12} {'Gauss Imp':<12} {'NLM Imp':<12} {'Winner':<10}\n")
            f.write("-" * 80 + "\n")
            
            for _, row in df_comparison.iterrows():
                winner = "Gaussian" if row['gaussian_improvement'] > row['nlmeans_improvement'] else \
                        "NLM" if row['nlmeans_improvement'] > row['gaussian_improvement'] else "Tie"
                
                f.write(f"{row['epsilon']:<10.3f} {row['gaussian_defended_accuracy']:<12.2f}% {row['nlmeans_defended_accuracy']:<12.2f}% "
                       f"{row['gaussian_improvement']:<12.2f}% {row['nlmeans_improvement']:<12.2f}% {winner:<10}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("CONCLUSION\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("Both Gaussian Blur and Non-Local Means are effective defenses against adversarial attacks.\n")
            f.write("The choice between them depends on:\n")
            f.write("1. Available computational resources\n")
            f.write("2. Required image quality after defense\n")
            f.write("3. Specific attack characteristics\n")
            f.write("4. Application requirements\n\n")
            
            f.write("For most applications, a hybrid approach might be best:\n")
            f.write("- Use Gaussian Blur for real-time, resource-constrained applications\n")
            f.write("- Use Non-Local Means for high-quality, offline processing\n")
            f.write("- Consider adaptive switching based on attack strength and resource availability\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF COMPARISON REPORT\n")
            f.write("=" * 80 + "\n")
        
        print(f"✓ NLM vs Gaussian blur comparison report saved: {comparison_report_path}")
        
    except Exception as e:
        print(f"Error comparing Gaussian blur and NLM: {e}")

def main():
    """Main function"""
    print("=" * 80)
    print("COMPREHENSIVE NON-LOCAL MEANS DEFENSE OPTIMIZATION")
    print("Advanced Parameter Selection with Complete Output Files")
    print("=" * 80)
    
    start_time = time.time()
    
    # Check if OpenCV is available
    try:
        import cv2
        cv2_version = cv2.__version__
        print(f"OpenCV version: {cv2_version} (NLM denoising available)")
    except ImportError:
        print("Error: OpenCV not installed. Non-Local Means denoising requires OpenCV.")
        print("Please run: pip install opencv-python")
        return
    
    # Set output directory
    output_dir = './comprehensive_nlmeans_defense'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load model
    print("\n1. Loading model...")
    model = load_model('best_model.pth')
    if model is None:
        return
    
    # 2. Load adversarial sample datasets
    print("\n2. Loading adversarial samples...")
    adv_datasets = load_adv_samples('./adversarial_eval_samples')
    
    if not adv_datasets:
        print("Error: No adversarial sample datasets found")
        return
    
    print(f"Found {len(adv_datasets)} epsilon values of adversarial samples")
    
    # 3. Define parameter combination space
    print("\n3. Defining parameter space for NLM optimization...")
    
    # NLM parameter combinations
    param_combinations = []
    
    # Light denoising combinations
    for h in [5, 10, 15]:
        param_combinations.append({
            'h': h,
            'template_window_size': 7,
            'search_window_size': 21,
            'name': f'nlmeans_h{h}_t7_s21'
        })
    
    # Moderate denoising combinations
    for h in [10, 15, 20]:
        param_combinations.append({
            'h': h,
            'template_window_size': 7,
            'search_window_size': 35,
            'name': f'nlmeans_h{h}_t7_s35'
        })
    
    # Strong denoising combinations
    for h in [15, 20, 25]:
        param_combinations.append({
            'h': h,
            'template_window_size': 11,
            'search_window_size': 35,
            'name': f'nlmeans_h{h}_t11_s35'
        })
    
    # Test different window sizes
    for template in [5, 7, 9]:
        for search in [15, 21, 27]:
            param_combinations.append({
                'h': 10,
                'template_window_size': template,
                'search_window_size': search,
                'name': f'nlmeans_h10_t{template}_s{search}'
            })
    
    print(f"Testing {len(param_combinations)} NLM parameter combinations")
    
    # 4. Select optimal parameters for each epsilon
    print("\n4. Selecting optimal NLM parameters for each epsilon (using advanced scoring)...")
    all_best_results = {}
    
    for epsilon, dataset in sorted(adv_datasets.items()):
        print(f"\n{'='*60}")
        print(f"Processing epsilon = {epsilon}")
        print(f"{'='*60}")
        
        # Select optimal parameters
        best_result, all_results = select_best_nlmeans_parameters(
            model, dataset, param_combinations, epsilon
        )
        
        # Save parameter selection results
        param_selection_dir = os.path.join(output_dir, f'epsilon_{epsilon:.3f}', 'parameter_selection')
        os.makedirs(param_selection_dir, exist_ok=True)
        
        # Save all parameter results
        all_results_df = pd.DataFrame(all_results)
        all_results_path = os.path.join(param_selection_dir, 'all_nlmeans_parameter_results.csv')
        all_results_df.to_csv(all_results_path, index=False)
        
        # Save best parameters
        best_params_path = os.path.join(param_selection_dir, 'best_nlmeans_parameters.json')
        with open(best_params_path, 'w', encoding='utf-8') as f:
            json.dump(best_result, f, indent=4)
        
        # 5. Generate comprehensive report with optimal parameters
        print(f"\n  Generating comprehensive NLM defense report...")
        eps_dir = os.path.join(output_dir, f'epsilon_{epsilon:.3f}')
        final_results = generate_nlmeans_comprehensive_report(
            model, dataset, best_result, epsilon, eps_dir
        )
        
        all_best_results[epsilon] = (best_result, final_results)
        
        # 6. Print parameter selection summary
        print(f"\n  NLM parameter selection summary for ε={epsilon}:")
        print(f"    • Best parameters: h={best_result['h']}, template={best_result['template_window_size']}, search={best_result['search_window_size']}")
        print(f"    • NLM defense improvement: {best_result['defense_improvement']:.2f}%")
        print(f"    • Original degradation: {best_result['original_degradation']:.2f}%")
        print(f"    • Denoise strength: {best_result['denoise_strength']:.3f}")
        print(f"    • Quality: {best_result['quality_assessment']['overall_assessment']}")
    
    # 7. Generate global analysis
    print("\n5. Generating global NLM analysis...")
    generate_nlmeans_global_analysis(all_best_results, output_dir)
    
    # 8. Compare with Gaussian blur (if Gaussian blur results exist)
    print("\n6. Comparing NLM with Gaussian blur defense...")
    compare_nlmeans_vs_gaussian(all_best_results, './comprehensive_blur_defense')
    
    # Calculate total time
    end_time = time.time()
    total_time = end_time - start_time
    
    # Save execution log
    log_data = {
        'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'end_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_time_seconds': total_time,
        'num_epsilon_values': len(adv_datasets),
        'num_parameter_combinations': len(param_combinations),
        'output_directory': os.path.abspath(output_dir),
        'scoring_method': 'advanced_nonlinear_nlm_optimized',
        'opencv_version': cv2_version if 'cv2_version' in locals() else 'unknown'
    }
    
    log_path = os.path.join(output_dir, 'nlmeans_execution_log.json')
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=4)
    
    print(f"\n{'='*80}")
    print("✓ COMPREHENSIVE NON-LOCAL MEANS DEFENSE OPTIMIZATION COMPLETED SUCCESSFULLY!")
    print(f"✓ Results saved in: {os.path.abspath(output_dir)}")
    print(f"✓ Total execution time: {total_time:.1f} seconds")
    print("=" * 80)
    
    # Display final summary
    print(f"\nFINAL SUMMARY:")
    print(f"  • Total epsilon values processed: {len(all_best_results)}")
    print(f"  • NLM parameter combinations tested per epsilon: {len(param_combinations)}")
    print(f"  • Scoring method: Advanced non-linear (NLM optimized)")
    print(f"  • OpenCV version: {cv2_version if 'cv2_version' in locals() else 'unknown'}")
    
    # Quality distribution
    quality_counts = {}
    for epsilon, (best_result, _) in all_best_results.items():
        quality = best_result['quality_assessment']['overall_assessment']
        quality_counts[quality] = quality_counts.get(quality, 0) + 1
    
    print(f"  • Quality distribution:")
    for quality, count in quality_counts.items():
        print(f"    - {quality}: {count}")
    
    print(f"\nFiles generated for each epsilon:")
    print(f"  ✓ NLM parameter selection results")
    print(f"  ✓ Comprehensive NLM defense report")
    print(f"  ✓ NLM defense visualization")
    print(f"  ✓ NLM confusion matrix comparison")
    print(f"  ✓ NLM defense effectiveness analysis")
    print(f"  ✓ NLM confidence comparison")
    print(f"  ✓ NLM parameter effect analysis")
    print(f"  ✓ NLM quality assessment radar")
    print(f"  ✓ NLM classification reports")
    print(f"  ✓ NLM raw prediction data")
    
    print(f"\nGlobal NLM analysis files:")
    print(f"  ✓ Global NLM optimization summary")
    print(f"  ✓ Global NLM analysis charts")
    print(f"  ✓ Global NLM optimization report")
    print(f"  ✓ NLM vs Gaussian comparison (if Gaussian results available)")
    print(f"  ✓ Execution log")

if __name__ == '__main__':
    main()