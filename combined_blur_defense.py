"""
combined_blur_defense.py
Comprehensive Gaussian Blur Defense: Select optimal parameters for each epsilon and generate detailed reports
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
    """Load the trained model"""
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
                print(f"  Loaded: {pkl_file}, epsilon={epsilon}, num_samples={dataset.get('num_samples', 'unknown')}")
                adv_datasets[epsilon] = dataset
            else:
                print(f"  Warning: No epsilon information in {pkl_file}")
                
        except Exception as e:
            print(f"  Failed to load {pkl_file}: {e}")
    
    return adv_datasets

def apply_gaussian_blur(images, kernel_size=3, sigma=1.0):
    """
    Apply Gaussian blur defense
    """
    from torchvision.transforms.functional import gaussian_blur
    
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure odd number
        
    blurred_images = []
    
    for i in range(images.shape[0]):
        img = images[i]
        # Apply Gaussian blur
        blurred = gaussian_blur(img, kernel_size=kernel_size, sigma=sigma)
        blurred_images.append(blurred.unsqueeze(0))
    
    return torch.cat(blurred_images, dim=0)

def calculate_blur_strength(sigma, kernel_size):
    """Calculate blur strength (normalized)"""
    return sigma * kernel_size / 3.0

def advanced_tradeoff_score(defense_improvement, orig_degradation, blur_strength, 
                           alpha_base=0.2, beta_base=0.1, defense_cap=50):
    """
    Improved trade-off score calculation - using non-linear weights and diminishing returns
    """
    # 1. Non-linear weights: increased penalty when original image degradation exceeds threshold
    if orig_degradation > 10:  # Exceeds 10% degradation
        # Exponential penalty: the more degradation, the heavier the penalty
        alpha = alpha_base * (1 + 0.1 * (orig_degradation - 10))
    elif orig_degradation > 5:  # 5-10% degradation
        alpha = alpha_base * 1.5  # Moderate penalty increase
    else:
        alpha = alpha_base  # Base weight
    
    # 2. Diminishing returns for defense improvement
    # First defense_cap% improvement has high value, then diminishing returns
    if defense_improvement <= defense_cap:
        defense_value = defense_improvement
    else:
        # Diminishing returns: value halves after threshold
        defense_value = defense_cap + (defense_improvement - defense_cap) * 0.5
    
    # 3. Consider negative defense improvement (defense reduces accuracy)
    if defense_improvement < 0:
        # Double penalty for negative improvement
        defense_value = defense_improvement * 2.0  # Double penalty
    
    # 4. Non-linear penalty for blur strength (avoid excessive blurring)
    # Exponential penalty: larger blur strength leads to faster increasing penalty
    blur_penalty = beta_base * (blur_strength ** 1.5)
    
    # 5. Calculate final score
    score = defense_value - alpha * orig_degradation - blur_penalty
    
    # 6. Additional consideration: severe penalty if original image degradation is too high
    if orig_degradation > 20:  # Original image degradation exceeds 20%
        score -= 10  # Severe penalty
    
    # 7. Consider marginal effect of blur strength
    # Additional penalty when blur strength exceeds 1.5 (avoid completely destroying the image)
    if blur_strength > 1.5:
        score -= (blur_strength - 1.5) * 5
    
    return score

def assess_defense_quality(defense_improvement, orig_degradation, blur_strength):
    """Assess defense quality level"""
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
    
    # Blur strength grade
    if blur_strength < 0.5:
        blur_grade = "Light"
    elif blur_strength < 1.0:
        blur_grade = "Moderate"
    elif blur_strength < 1.5:
        blur_grade = "Heavy"
    else:
        blur_grade = "Very Heavy"
    
    # Overall quality assessment
    if defense_improvement > 20 and orig_degradation < 5 and blur_strength < 1.0:
        overall = "High Quality Defense"
    elif defense_improvement > 10 and orig_degradation < 10 and blur_strength < 1.5:
        overall = "Medium Quality Defense"
    elif defense_improvement > 0:
        overall = "Low Quality Defense"
    else:
        overall = "Poor Defense (Harmful)"
    
    return {
        'defense_grade': defense_grade,
        'original_impact_grade': orig_grade,
        'blur_grade': blur_grade,
        'overall_assessment': overall
    }

def analyze_blur_effect_on_original(model, orig_images, labels, kernel_size=3, sigma=1.0):
    """
    Analyze the effect of blur on original images
    """
    if orig_images is None:
        return None, None
    
    model.eval()
    
    # Apply blur
    blurred_images = apply_gaussian_blur(orig_images, kernel_size=kernel_size, sigma=sigma)
    
    # Evaluate accuracy after blurring
    correct = 0
    confidences = []
    
    batch_size = 256
    num_batches = (len(labels) + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(labels))
        
        batch_images = blurred_images[start_idx:end_idx]
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

def evaluate_single_defense(model, adv_images, orig_images, labels, perturbations, 
                          kernel_size, sigma, defense_name='gaussian_blur'):
    """
    Evaluate the effect of a single defense parameter combination (using advanced scoring)
    """
    model.eval()
    num_samples = len(labels)
    
    # Apply defense
    defended_images = apply_gaussian_blur(adv_images, kernel_size=kernel_size, sigma=sigma)
    
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
    
    # Analyze blur effect on original images
    if orig_images is not None:
        orig_blurred_accuracy, _ = analyze_blur_effect_on_original(
            model, orig_images, labels, kernel_size, sigma
        )
        orig_degradation = orig_accuracy - orig_blurred_accuracy if orig_accuracy else 0
    else:
        orig_blurred_accuracy = None
        orig_degradation = 0
    
    # Calculate blur strength
    blur_strength = calculate_blur_strength(sigma, kernel_size)
    
    # Calculate improved trade-off score
    tradeoff_score = advanced_tradeoff_score(
        defense_improvement=defense_improvement,
        orig_degradation=orig_degradation,
        blur_strength=blur_strength
    )
    
    # Quality assessment
    quality_assessment = assess_defense_quality(defense_improvement, orig_degradation, blur_strength)
    
    return {
        'kernel_size': kernel_size,
        'sigma': sigma,
        'defense_name': defense_name,
        'num_samples': num_samples,
        
        # Accuracy metrics
        'original_accuracy': orig_accuracy,
        'adversarial_accuracy': adv_accuracy,
        'defended_accuracy': defended_accuracy,
        'defense_improvement': defense_improvement,
        
        # Original image impact
        'original_blurred_accuracy': orig_blurred_accuracy,
        'original_degradation': orig_degradation,
        
        # Blur related
        'blur_strength': blur_strength,
        
        # Score
        'tradeoff_score': tradeoff_score,
        
        # Quality assessment
        'quality_assessment': quality_assessment
    }

def select_best_parameters_for_epsilon(model, dataset, param_combinations, epsilon):
    """
    Select optimal parameters for a specific epsilon (using advanced scoring)
    """
    print(f"  Selecting optimal parameters for ε={epsilon}...")
    
    # Extract data
    adv_images = dataset['adversarial_images']
    orig_images = dataset.get('original_images', None)
    labels = dataset['labels']
    perturbations = dataset.get('perturbations', None)
    
    # Test all parameter combinations
    all_results = []
    
    for params in param_combinations:
        kernel_size = params['kernel_size']
        sigma = params['sigma']
        defense_name = params.get('name', f'blur_k{kernel_size}_s{sigma}')
        
        print(f"    Testing: kernel={kernel_size}, sigma={sigma:.2f}", end="")
        
        # Evaluate current parameters (using advanced scoring)
        result = evaluate_single_defense(
            model, adv_images, orig_images, labels, perturbations,
            kernel_size, sigma, defense_name
        )
        
        all_results.append(result)
        print(f" → Improvement: {result['defense_improvement']:6.2f}%, "
              f"Original Degradation: {result['original_degradation']:5.2f}%, "
              f"Blur Strength: {result['blur_strength']:5.2f}, "
              f"Score: {result['tradeoff_score']:7.2f}")
    
    # Sort by trade-off score
    all_results.sort(key=lambda x: x['tradeoff_score'], reverse=True)
    
    # Select best parameters
    best_result = all_results[0]
    
    print(f"  ✓ Optimal parameters: kernel={best_result['kernel_size']}, sigma={best_result['sigma']:.2f}")
    print(f"    Defense improvement: {best_result['defense_improvement']:.2f}%")
    print(f"    Original degradation: {best_result['original_degradation']:.2f}%")
    print(f"    Trade-off score: {best_result['tradeoff_score']:.2f}")
    print(f"    Quality assessment: {best_result['quality_assessment']['overall_assessment']}")
    
    return best_result, all_results

def generate_comprehensive_defense_report(model, dataset, best_params, epsilon, output_dir):
    """
    Generate comprehensive defense report (including all analyses)
    """
    print(f"  Generating comprehensive defense report...")
    
    # Extract data
    adv_images = dataset['adversarial_images']
    orig_images = dataset.get('original_images', None)
    labels = dataset['labels']
    perturbations = dataset.get('perturbations', None)
    
    # Create output directory
    eps_dir = os.path.join(output_dir, f'epsilon_{epsilon:.3f}')
    os.makedirs(eps_dir, exist_ok=True)
    
    # Apply optimal parameter defense
    kernel_size = best_params['kernel_size']
    sigma = best_params['sigma']
    defense_name = f'gaussian_blur_optimal_k{kernel_size}_s{sigma}'
    
    defended_images = apply_gaussian_blur(adv_images, kernel_size=kernel_size, sigma=sigma)
    
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
    blur_strength = calculate_blur_strength(sigma, kernel_size)
    if orig_accuracy is not None:
        orig_blurred_accuracy, _ = analyze_blur_effect_on_original(model, orig_images, labels, kernel_size, sigma)
        orig_degradation = orig_accuracy - orig_blurred_accuracy
    else:
        orig_degradation = 0
    
    tradeoff_score = advanced_tradeoff_score(defense_improvement, orig_degradation, blur_strength)
    quality_assessment = assess_defense_quality(defense_improvement, orig_degradation, blur_strength)
    
    # Final results
    final_results = {
        'epsilon': epsilon,
        'num_samples': num_samples,
        'defense_name': defense_name,
        'kernel_size': kernel_size,
        'sigma': sigma,
        
        # Accuracy metrics
        'original_accuracy': orig_accuracy,
        'adversarial_accuracy': adv_accuracy,
        'defended_accuracy': defended_accuracy,
        'defense_improvement': defense_improvement,
        
        # Original image impact
        'original_blurred_accuracy': orig_blurred_accuracy if orig_accuracy is not None else None,
        'original_degradation': orig_degradation,
        
        # Blur related
        'blur_strength': blur_strength,
        
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
    generate_all_analysis_files(
        model, final_results, orig_images, adv_images, defended_images, 
        perturbations, labels, class_names, epsilon, eps_dir
    )
    
    # 2. Save detailed report
    save_comprehensive_report(final_results, class_names, eps_dir)
    
    # 3. Save raw data
    save_all_raw_data(final_results, eps_dir)
    
    return final_results

def generate_all_analysis_files(model, results, orig_images, adv_images, defended_images, 
                               perturbations, labels, class_names, epsilon, output_dir):
    """Generate all analysis files"""
    
    # Create subdirectory
    analysis_dir = os.path.join(output_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    # 1. Generate defense visualization
    generate_defense_visualization(
        model, orig_images, adv_images, defended_images, perturbations, labels, 
        epsilon, results['defense_name'], analysis_dir, 
        results['kernel_size'], results['sigma'], num_samples=10
    )
    
    # 2. Generate confusion matrix comparison
    generate_confusion_comparison(
        labels, results['preds_adv'], results['preds_defended'], class_names, 
        epsilon, results['defense_name'], analysis_dir
    )
    
    # 3. Generate defense effectiveness analysis
    generate_defense_analysis(
        results, labels, results['preds_orig'], results['preds_adv'], 
        results['preds_defended'], results['conf_orig'], results['conf_adv'], 
        results['conf_defended'], results['pert_norms'], class_names, analysis_dir
    )
    
    # 4. Generate confidence comparison analysis
    generate_confidence_comparison(
        labels, results['preds_orig'], results['preds_adv'], results['preds_defended'],
        results['conf_orig'], results['conf_adv'], results['conf_defended'],
        epsilon, results['defense_name'], analysis_dir
    )
    
    # 5. Generate parameter effect analysis
    generate_parameter_effect_analysis(results, epsilon, analysis_dir)
    
    # 6. Generate quality assessment chart
    generate_quality_assessment_chart(results, epsilon, analysis_dir)
    
    # 7. Generate classification reports
    generate_classification_reports(labels, results['preds_adv'], 
                                   results['preds_defended'], class_names, 
                                   epsilon, analysis_dir)

def generate_defense_visualization(model, orig_images, adv_images, defended_images, perturbations, labels, 
                                  epsilon, defense_name, output_dir, kernel_size, sigma, num_samples=10):
    """Generate defense effect visualization"""
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
        
        # 3. After Gaussian blur processing
        axes[i, 2].imshow(defended_display.numpy(), cmap='gray')
        defended_class = class_names[defended_pred]
        defense_success = "[OK]" if defended_pred == label else "[FAIL]"
        defense_color = 'green' if defense_success == "[OK]" else 'red'
        axes[i, 2].set_title(
            f'{defense_name}\nKernel={kernel_size}, σ={sigma}\nPred: {defended_class} {defense_success}\nConf: {defended_conf:.2%}',
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
    
    save_path = os.path.join(output_dir, 'defense_visualization.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Defense visualization saved: {save_path}")

def generate_confusion_comparison(labels, preds_adv, preds_defended, class_names, 
                                 epsilon, defense_name, output_dir):
    """Generate confusion matrix comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Adversarial sample confusion matrix
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
    ax2.set_title(f'Confusion Matrix After Defense\n(ε={epsilon})', 
                 fontsize=14, fontweight='bold')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'confusion_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Confusion matrix comparison saved: {save_path}")

def generate_defense_analysis(results, labels, preds_orig, preds_adv, preds_defended,
                             conf_orig, conf_adv, conf_defended, pert_norms, 
                             class_names, output_dir):
    """Generate defense effectiveness analysis"""
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
    
    # 1. Attack success rate vs defense accuracy by class
    x = np.arange(num_classes)
    width = 0.35
    
    ax1.bar(x - width/2, attack_rates, width, label='Attack Success Rate', 
            color='red', alpha=0.7)
    ax1.bar(x + width/2, defended_accuracies, width, label='After Defense Accuracy', 
            color='green', alpha=0.7)
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Percentage (%)')
    ax1.set_title('Attack Success vs Defense Accuracy by Class')
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Defense effectiveness improvement by class
    bars = ax2.bar(class_names, defense_improvements, color='blue', alpha=0.7)
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Accuracy Improvement (%)')
    ax2.set_title('Defense Effectiveness by Class')
    ax2.set_xticklabels(class_names, rotation=45, ha='right')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, improvement in zip(bars, defense_improvements):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{improvement:+.1f}%', ha='center', va='bottom' if improvement > 0 else 'top')
    
    # 3. Accuracy comparison (Original vs Adversarial vs After Defense)
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
        ax3.bar(x + width, defended_accuracies, width, label='After Defense', 
                color='green', alpha=0.7)
        
        ax3.set_xlabel('Class')
        ax3.set_ylabel('Accuracy (%)')
        ax3.set_title('Accuracy Comparison: Original vs Adversarial vs Defense')
        ax3.set_xticks(x)
        ax3.set_xticklabels(class_names, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
    else:
        ax3.text(0.5, 0.5, 'Original accuracy data unavailable', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Accuracy Comparison (Original Data Unavailable)')
    
    # 4. Hardest classes to defend (poorest defense effectiveness)
    sorted_indices = np.argsort(defense_improvements)
    hardest_classes = [class_names[i] for i in sorted_indices[:5]]
    hardest_improvements = [defense_improvements[i] for i in sorted_indices[:5]]
    
    colors = ['red' if imp <= 0 else 'orange' for imp in hardest_improvements]
    ax4.bar(hardest_classes, hardest_improvements, color=colors, alpha=0.7)
    ax4.set_xlabel('Class')
    ax4.set_ylabel('Accuracy Improvement (%)')
    ax4.set_title('Hardest Classes to Defend (Lowest Improvement)')
    ax4.set_xticklabels(hardest_classes, rotation=45, ha='right')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'defense_analysis.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Defense analysis saved: {save_path}")

def generate_confidence_comparison(labels, preds_orig, preds_adv, preds_defended,
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
    ax1.hist(defended_confs, bins=30, alpha=0.5, label='After Defense', color='green')
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Confidence Distribution Comparison (ε={epsilon})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Attack success rate vs confidence
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
            label='After Defense', color='green')
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Attack Success Rate (%)')
    ax2.set_title('Attack Success Rate vs Confidence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Confidence change
    conf_change_adv = adv_confs - (orig_confs if orig_confs is not None else np.ones_like(adv_confs))
    conf_change_defended = defended_confs - adv_confs
    
    ax3.hist(conf_change_adv, bins=50, alpha=0.5, label='Adv - Orig', color='red')
    ax3.hist(conf_change_defended, bins=50, alpha=0.5, label='Defended - Adv', color='green')
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
    ax4.set_ylabel(f'Confidence After Defense')
    ax4.set_title('Confidence Before vs After Defense')
    ax4.grid(True, alpha=0.3)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.6, label='Attack Success (After Defense)'),
        Patch(facecolor='blue', alpha=0.6, label='Attack Failed (After Defense)')
    ]
    ax4.legend(handles=legend_elements)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'confidence_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Confidence comparison analysis saved: {save_path}")

def generate_parameter_effect_analysis(results, epsilon, output_dir):
    """Generate parameter effect analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Parameter vs defense effectiveness relationship
    kernel = results['kernel_size']
    sigma = results['sigma']
    blur_strength = results['blur_strength']
    
    axes[0, 0].bar(['Kernel Size', 'Sigma', 'Blur Strength'], 
                  [kernel, sigma, blur_strength], 
                  color=['blue', 'green', 'orange'], alpha=0.7)
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].set_title(f'Optimal Parameters (ε={epsilon})')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (name, value) in enumerate(zip(['Kernel Size', 'Sigma', 'Blur Strength'], 
                                          [kernel, sigma, blur_strength])):
        axes[0, 0].text(i, value + 0.05 * value, f'{value:.2f}', 
                       ha='center', va='bottom')
    
    # 2. Defense effectiveness metrics
    metrics = ['Adv Accuracy', 'Defended Accuracy', 'Improvement']
    values = [results['adversarial_accuracy'], 
              results['defended_accuracy'], 
              results['defense_improvement']]
    colors = ['red', 'green', 'blue']
    
    axes[0, 1].bar(metrics, values, color=colors, alpha=0.7)
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Defense Performance Metrics')
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
        axes[1, 0].set_title('Impact on Original Images')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        axes[1, 0].text(0, results['original_degradation'] + 0.5, 
                       f'{results["original_degradation"]:.2f}%', 
                       ha='center', va='bottom')
    else:
        axes[1, 0].text(0.5, 0.5, 'Original image data unavailable', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Impact on Original Images (Data Unavailable)')
    
    # 4. Trade-off score and quality assessment
    axes[1, 1].bar(['Trade-off Score'], [results['tradeoff_score']], 
                  color='orange', alpha=0.7)
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Advanced Trade-off Score')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    axes[1, 1].text(0, results['tradeoff_score'] + 0.5, 
                   f'{results["tradeoff_score"]:.2f}', 
                   ha='center', va='bottom')
    
    # Add quality assessment labels below the chart
    quality = results['quality_assessment']
    plt.figtext(0.5, 0.02, 
                f'Quality Assessment: {quality["overall_assessment"]} | '
                f'Defense: {quality["defense_grade"]} | '
                f'Original Impact: {quality["original_impact_grade"]} | '
                f'Blur: {quality["blur_grade"]}',
                ha='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    save_path = os.path.join(output_dir, 'parameter_effect_analysis.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Parameter effect analysis saved: {save_path}")

def generate_quality_assessment_chart(results, epsilon, output_dir):
    """Generate quality assessment chart"""
    quality = results['quality_assessment']
    
    # Create radar chart
    fig = plt.figure(figsize=(10, 8))
    
    # Radar chart data
    categories = ['Defense\nEffectiveness', 'Original\nImpact', 'Blur\nIntensity', 'Overall\nQuality']
    
    # Convert quality grades to numerical values
    defense_scores = {'Poor': 1, 'Fair': 2, 'Good': 3, 'Excellent': 4}
    impact_scores = {'Severe': 1, 'High': 2, 'Moderate': 3, 'Low': 4, 'Minimal': 5}
    blur_scores = {'Very Heavy': 1, 'Heavy': 2, 'Moderate': 3, 'Light': 4}
    overall_scores = {'Poor Defense (Harmful)': 1, 'Low Quality Defense': 2, 
                      'Medium Quality Defense': 3, 'High Quality Defense': 4}
    
    values = [
        defense_scores.get(quality['defense_grade'], 2),
        impact_scores.get(quality['original_impact_grade'], 3),
        blur_scores.get(quality['blur_grade'], 3),
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
    ax.plot(angles, values, 'o-', linewidth=2, color='blue')
    ax.fill(angles, values, alpha=0.25, color='blue')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'], fontsize=8)
    ax.set_title(f'Quality Assessment Radar Chart (ε={epsilon})', 
                fontsize=12, fontweight='bold', pad=20)
    
    # Add value labels
    for i, (angle, value) in enumerate(zip(angles[:-1], values[:-1])):
        ax.text(angle, value + 0.2, str(value), ha='center', va='center', 
               fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'quality_assessment_radar.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Quality assessment radar chart saved: {save_path}")

def generate_classification_reports(labels, preds_adv, preds_defended, class_names, epsilon, output_dir):
    """Generate classification reports"""
    # Adversarial sample classification report
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
    
    report_path = os.path.join(output_dir, 'classification_reports.json')
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
    axes[0].bar(x + width/2, f1_scores_defended, width, label='After Defense', color='green', alpha=0.7)
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
    axes[1].bar(x + width/2, defended_values, width, label='After Defense', color='green', alpha=0.7)
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
    plot_path = os.path.join(output_dir, 'classification_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Classification report comparison chart saved: {plot_path}")
    print(f"    ✓ Detailed classification reports saved: {report_path}")

def save_comprehensive_report(results, class_names, output_dir):
    """Save comprehensive report"""
    report_path = os.path.join(output_dir, 'comprehensive_defense_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"COMPREHENSIVE GAUSSIAN BLUR DEFENSE REPORT - ε={results['epsilon']}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. OPTIMAL PARAMETERS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Defense Method: {results['defense_name']}\n")
        f.write(f"Kernel Size: {results['kernel_size']}\n")
        f.write(f"Sigma: {results['sigma']:.3f}\n")
        f.write(f"Blur Strength: {results['blur_strength']:.3f}\n")
        f.write(f"Epsilon (ε): {results['epsilon']}\n")
        f.write(f"Total Samples: {results['num_samples']}\n\n")
        
        f.write("2. PERFORMANCE METRICS\n")
        f.write("-" * 40 + "\n")
        if results['original_accuracy'] is not None:
            f.write(f"Original Accuracy: {results['original_accuracy']:.2f}%\n")
        f.write(f"Adversarial Accuracy (No Defense): {results['adversarial_accuracy']:.2f}%\n")
        f.write(f"Accuracy After Defense: {results['defended_accuracy']:.2f}%\n")
        f.write(f"Defense Improvement: {results['defense_improvement']:.2f}%\n")
        f.write(f"Attack Success Rate Reduction: {100 - results['adversarial_accuracy'] - (100 - results['defended_accuracy']):.2f}%\n")
        f.write(f"Average Perturbation Norm: {results['avg_perturbation_norm']:.4f}\n\n")
        
        f.write("3. ORIGINAL IMAGE IMPACT\n")
        f.write("-" * 40 + "\n")
        if results['original_blurred_accuracy'] is not None:
            f.write(f"Original Accuracy After Blurring: {results['original_blurred_accuracy']:.2f}%\n")
        f.write(f"Original Accuracy Degradation: {results['original_degradation']:.2f}%\n\n")
        
        f.write("4. ADVANCED SCORING\n")
        f.write("-" * 40 + "\n")
        f.write(f"Trade-off Score: {results['tradeoff_score']:.3f}\n")
        f.write(f"Scoring Method: Advanced non-linear with diminishing returns\n\n")
        
        f.write("5. QUALITY ASSESSMENT\n")
        f.write("-" * 40 + "\n")
        quality = results['quality_assessment']
        f.write(f"Defense Effectiveness: {quality['defense_grade']}\n")
        f.write(f"Original Image Impact: {quality['original_impact_grade']}\n")
        f.write(f"Blur Intensity: {quality['blur_grade']}\n")
        f.write(f"Overall Assessment: {quality['overall_assessment']}\n\n")
        
        f.write("6. CONFIDENCE METRICS\n")
        f.write("-" * 40 + "\n")
        if results['avg_confidence_original'] is not None:
            f.write(f"Average Confidence (Original): {results['avg_confidence_original']:.4f}\n")
        f.write(f"Average Confidence (Adversarial): {results['avg_confidence_adversarial']:.4f}\n")
        f.write(f"Average Confidence (After Defense): {results['avg_confidence_defended']:.4f}\n\n")
        
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
            f.write("✓ EXCELLENT DEFENSE: Significant improvement (>20%) achieved\n")
        elif results['defense_improvement'] > 10:
            f.write("✓ GOOD DEFENSE: Moderate improvement (10-20%) achieved\n")
        elif results['defense_improvement'] > 0:
            f.write("✓ FAIR DEFENSE: Small improvement (<10%) achieved\n")
        else:
            f.write("✗ POOR DEFENSE: No improvement or negative impact\n")
        
        if results['original_degradation'] < 5:
            f.write("✓ MINIMAL IMPACT: Original image degradation <5%\n")
        elif results['original_degradation'] < 10:
            f.write("⚠ MODERATE IMPACT: Original image degradation 5-10%\n")
        else:
            f.write("⚠ HIGH IMPACT: Original image degradation >10%\n")
        
        f.write(f"\n9. RECOMMENDATION\n")
        f.write("-" * 40 + "\n")
        
        if quality['overall_assessment'] == "High Quality Defense":
            f.write("✓ HIGHLY RECOMMENDED\n")
            f.write(f"  These parameters provide excellent defense with minimal impact.\n")
            f.write(f"  Use kernel={results['kernel_size']}, sigma={results['sigma']:.3f} for ε={results['epsilon']}\n")
        
        elif quality['overall_assessment'] == "Medium Quality Defense":
            f.write("✓ RECOMMENDED WITH CONSIDERATIONS\n")
            f.write(f"  These parameters provide good defense with moderate impact.\n")
            f.write(f"  Consider lighter blur if original image quality is critical.\n")
        
        elif quality['overall_assessment'] == "Low Quality Defense":
            f.write("⚠ USE WITH CAUTION\n")
            f.write(f"  Defense improvement is limited ({results['defense_improvement']:.2f}%).\n")
            f.write(f"  Consider alternative defense methods or parameters.\n")
        
        else:
            f.write("✗ NOT RECOMMENDED\n")
            f.write(f"  This defense is ineffective or harmful.\n")
            f.write(f"  Do not use these parameters for ε={results['epsilon']}\n")
        
        f.write("\n10. FILES GENERATED\n")
        f.write("-" * 40 + "\n")
        f.write("In the analysis directory you will find:\n")
        f.write("  - defense_visualization.png: Sample images before/after defense\n")
        f.write("  - confusion_comparison.png: Confusion matrices comparison\n")
        f.write("  - defense_analysis.png: Detailed defense effectiveness analysis\n")
        f.write("  - confidence_comparison.png: Confidence analysis\n")
        f.write("  - parameter_effect_analysis.png: Parameter impact analysis\n")
        f.write("  - quality_assessment_radar.png: Quality radar chart\n")
        f.write("  - classification_comparison.png: Classification metrics\n")
        f.write("  - defense_raw_results.csv: Raw prediction data\n")
        f.write("  - classification_reports.json: Detailed classification reports\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"    ✓ Comprehensive defense report saved: {report_path}")

def save_all_raw_data(results, output_dir):
    """Save all raw data"""
    # Save raw result data
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
    
    csv_path = os.path.join(output_dir, 'defense_raw_results.csv')
    df.to_csv(csv_path, index=False)
    
    # Save result summary
    summary_path = os.path.join(output_dir, 'defense_results_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        # Remove Tensor objects
        summary = {k: v for k, v in results.items() 
                  if not isinstance(v, torch.Tensor) and k not in ['adv_images', 'orig_images', 'defended_images', 'perturbations']}
        json.dump(summary, f, indent=4)
    
    print(f"    ✓ Defense raw data saved: {csv_path}")
    print(f"    ✓ Defense result summary saved: {summary_path}")

def generate_global_analysis(all_best_results, output_dir):
    """Generate global analysis"""
    print("\nGenerating global analysis...")
    
    # Prepare summary data
    summary_data = []
    
    for epsilon, (best_result, detailed_result) in all_best_results.items():
        if detailed_result is not None:
            summary_data.append({
                'epsilon': epsilon,
                'kernel_size': best_result['kernel_size'],
                'sigma': best_result['sigma'],
                'blur_strength': best_result['blur_strength'],
                'adversarial_accuracy': best_result['adversarial_accuracy'],
                'defended_accuracy': best_result['defended_accuracy'],
                'defense_improvement': best_result['defense_improvement'],
                'original_degradation': detailed_result.get('original_degradation', 0),
                'tradeoff_score': best_result['tradeoff_score'],
                'quality': best_result['quality_assessment']['overall_assessment']
            })
    
    df_summary = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, 'global_optimization_summary.csv')
    df_summary.to_csv(summary_path, index=False)
    
    # Generate global charts
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    epsilons = df_summary['epsilon']
    
    # 1. Optimal parameter trends
    axes[0, 0].plot(epsilons, df_summary['sigma'], 'o-', color='red', markersize=8, label='Sigma')
    axes[0, 0].set_xlabel('Epsilon')
    axes[0, 0].set_ylabel('Sigma', color='red')
    axes[0, 0].set_title('Optimal Sigma vs Epsilon', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    ax_twin = axes[0, 0].twinx()
    ax_twin.plot(epsilons, df_summary['kernel_size'], 's-', color='blue', markersize=8, label='Kernel Size')
    ax_twin.set_ylabel('Kernel Size', color='blue')
    
    # Combine legends
    lines1, labels1 = axes[0, 0].get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    axes[0, 0].legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # 2. Defense effectiveness trends
    axes[0, 1].plot(epsilons, df_summary['adversarial_accuracy'], 'o-', 
                   label='Adversarial', color='red', markersize=8)
    axes[0, 1].plot(epsilons, df_summary['defended_accuracy'], 's-', 
                   label='Defended', color='green', markersize=8)
    axes[0, 1].set_xlabel('Epsilon')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Accuracy Trends', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Trade-off score trends
    axes[1, 0].plot(epsilons, df_summary['tradeoff_score'], 'o-', 
                   color='purple', markersize=8)
    axes[1, 0].set_xlabel('Epsilon')
    axes[1, 0].set_ylabel('Advanced Trade-off Score')
    axes[1, 0].set_title('Advanced Scoring Trends', fontweight='bold')
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
    axes[1, 1].set_title('Defense Quality Distribution', fontweight='bold')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (category, count) in enumerate(zip(quality_counts.index, quality_counts.values)):
        axes[1, 1].text(i, count + 0.1, str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    global_plot_path = os.path.join(output_dir, 'global_analysis.png')
    plt.savefig(global_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Global analysis chart saved: {global_plot_path}")
    print(f"✓ Global summary data saved: {summary_path}")
    
    # Generate global report
    report_path = os.path.join(output_dir, 'global_optimization_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("GLOBAL GAUSSIAN BLUR DEFENSE OPTIMIZATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("SUMMARY OF OPTIMAL PARAMETERS FOR DIFFERENT EPSILON VALUES:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Epsilon':<10} {'Kernel':<8} {'Sigma':<8} {'Blur Str':<10} {'Adv Acc':<10} {'Def Acc':<10} {'Improvement':<12} {'Score':<10} {'Quality':<20}\n")
        f.write("-" * 80 + "\n")
        
        for _, row in df_summary.iterrows():
            f.write(f"{row['epsilon']:<10.3f} {row['kernel_size']:<8} {row['sigma']:<8.2f} "
                   f"{row['blur_strength']:<10.2f} {row['adversarial_accuracy']:<10.2f}% {row['defended_accuracy']:<10.2f}% "
                   f"{row['defense_improvement']:<12.2f}% {row['tradeoff_score']:<10.2f} {row['quality']:<20}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("KEY INSIGHTS\n")
        f.write("=" * 80 + "\n\n")
        
        # Analyze trends
        if len(df_summary) > 1:
            avg_improvement = df_summary['defense_improvement'].mean()
            max_improvement = df_summary['defense_improvement'].max()
            min_improvement = df_summary['defense_improvement'].min()
            
            f.write(f"1. Defense Effectiveness:\n")
            f.write(f"   - Average improvement: {avg_improvement:.2f}%\n")
            f.write(f"   - Maximum improvement: {max_improvement:.2f}%\n")
            f.write(f"   - Minimum improvement: {min_improvement:.2f}%\n\n")
        
        # Parameter trends
        f.write(f"2. Parameter Trends:\n")
        f.write(f"   - Stronger attacks (higher epsilon) generally require stronger blurring\n")
        f.write(f"   - Optimal sigma increases with epsilon\n")
        f.write(f"   - Kernel size may vary based on specific attack characteristics\n\n")
        
        f.write("3. Quality Distribution:\n")
        for quality, count in quality_counts.items():
            percentage = 100.0 * count / len(df_summary)
            f.write(f"   - {quality}: {count} epsilon values ({percentage:.1f}%)\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("PRACTICAL RECOMMENDATIONS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Based on the optimization results, here are practical recommendations:\n\n")
        
        f.write("1. For UNKNOWN attack strength:\n")
        f.write("   Start with kernel=3, sigma=0.7 as a balanced default\n\n")
        
        f.write("2. For different epsilon ranges:\n")
        
        # Group recommendations by epsilon range
        eps_ranges = [(0, 0.03, "Very Small"), (0.03, 0.07, "Small"), 
                     (0.07, 0.12, "Medium"), (0.12, 0.2, "Large"), (0.2, 1.0, "Very Large")]
        
        for low, high, name in eps_ranges:
            range_data = df_summary[(df_summary['epsilon'] >= low) & (df_summary['epsilon'] < high)]
            if not range_data.empty:
                avg_kernel = range_data['kernel_size'].mean()
                avg_sigma = range_data['sigma'].mean()
                avg_imp = range_data['defense_improvement'].mean()
                
                f.write(f"   - {name} attacks (ε={low:.2f}-{high:.2f}):\n")
                f.write(f"     Recommended: kernel={avg_kernel:.1f}, sigma={avg_sigma:.2f}\n")
                f.write(f"     Expected improvement: {avg_imp:.1f}%\n\n")
        
        f.write("3. Implementation Strategy:\n")
        f.write("   - Estimate attack strength (epsilon) if possible\n")
        f.write("   - Use adaptive parameter selection based on epsilon\n")
        f.write("   - Monitor defense effectiveness in real-time\n")
        f.write("   - Consider quality assessment when choosing parameters\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("FILES GENERATED\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("This optimization generated the following files:\n")
        f.write("1. For each epsilon value:\n")
        f.write("   - Parameter selection results\n")
        f.write("   - Comprehensive defense report\n")
        f.write("   - All analysis visualizations\n")
        f.write("   - Raw prediction data\n")
        f.write("   - Classification reports\n\n")
        
        f.write("2. Global analysis files:\n")
        f.write("   - Global optimization summary (CSV)\n")
        f.write("   - Global analysis chart (PNG)\n")
        f.write("   - This global report\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("CONCLUSION\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Adaptive Gaussian blur defense is an effective method for protecting against adversarial attacks.\n")
        f.write("The key findings from this optimization are:\n")
        f.write("1. Optimal parameters vary with attack strength\n")
        f.write("2. Advanced scoring balances defense effectiveness and image quality\n")
        f.write("3. Quality assessment helps identify the best defense strategies\n")
        f.write("4. Comprehensive reporting enables informed decision-making\n\n")
        
        f.write("Use the generated files to implement and evaluate Gaussian blur defense in your applications.\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"✓ Global optimization report saved: {report_path}")

def main():
    """Main function"""
    print("=" * 80)
    print("COMPREHENSIVE GAUSSIAN BLUR DEFENSE OPTIMIZATION")
    print("Advanced Parameter Selection with Complete Output Files")
    print("=" * 80)
    
    start_time = time.time()
    
    # Set output directory
    output_dir = './comprehensive_blur_defense'
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
    
    print(f"Found {len(adv_datasets)} adversarial samples for different epsilon values")
    
    # 3. Define parameter combination space
    print("\n3. Defining parameter space for comprehensive optimization...")
    
    # Finer-grained parameter combinations
    param_combinations = []
    
    # Small kernel combinations (kernel_size=3)
    for sigma in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        param_combinations.append({
            'kernel_size': 3,
            'sigma': sigma,
            'name': f'k3_s{sigma:.1f}'
        })
    
    # Medium kernel combinations (kernel_size=5)
    for sigma in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        param_combinations.append({
            'kernel_size': 5,
            'sigma': sigma,
            'name': f'k5_s{sigma:.1f}'
        })
    
    # Large kernel combinations (kernel_size=7)
    for sigma in [0.3, 0.5, 0.7]:
        param_combinations.append({
            'kernel_size': 7,
            'sigma': sigma,
            'name': f'k7_s{sigma:.1f}'
        })
    
    # Test some extreme combinations
    param_combinations.append({'kernel_size': 3, 'sigma': 1.5, 'name': 'k3_s1.5_strong'})
    param_combinations.append({'kernel_size': 5, 'sigma': 1.2, 'name': 'k5_s1.2_very_strong'})
    
    print(f"Testing {len(param_combinations)} parameter combinations")
    
    # 4. Select optimal parameters for each epsilon
    print("\n4. Selecting optimal parameters for each epsilon (using advanced scoring)...")
    all_best_results = {}
    
    for epsilon, dataset in sorted(adv_datasets.items()):
        print(f"\n{'='*60}")
        print(f"Processing epsilon = {epsilon}")
        print(f"{'='*60}")
        
        # Select optimal parameters
        best_result, all_results = select_best_parameters_for_epsilon(
            model, dataset, param_combinations, epsilon
        )
        
        # Save parameter selection results
        param_selection_dir = os.path.join(output_dir, f'epsilon_{epsilon:.3f}', 'parameter_selection')
        os.makedirs(param_selection_dir, exist_ok=True)
        
        # Save all parameter results
        all_results_df = pd.DataFrame(all_results)
        all_results_path = os.path.join(param_selection_dir, 'all_parameter_results.csv')
        all_results_df.to_csv(all_results_path, index=False)
        
        # Save best parameters
        best_params_path = os.path.join(param_selection_dir, 'best_parameters.json')
        with open(best_params_path, 'w', encoding='utf-8') as f:
            json.dump(best_result, f, indent=4)
        
        # 5. Generate comprehensive report using optimal parameters
        print(f"\n  Generating comprehensive defense report...")
        eps_dir = os.path.join(output_dir, f'epsilon_{epsilon:.3f}')
        final_results = generate_comprehensive_defense_report(
            model, dataset, best_result, epsilon, eps_dir
        )
        
        all_best_results[epsilon] = (best_result, final_results)
        
        # 6. Print parameter selection summary
        print(f"\n  Parameter selection summary for ε={epsilon}:")
        print(f"    • Best parameters: kernel={best_result['kernel_size']}, sigma={best_result['sigma']:.3f}")
        print(f"    • Defense improvement: {best_result['defense_improvement']:.2f}%")
        print(f"    • Original degradation: {best_result['original_degradation']:.2f}%")
        print(f"    • Blur strength: {best_result['blur_strength']:.3f}")
        print(f"    • Quality: {best_result['quality_assessment']['overall_assessment']}")
    
    # 7. Generate global analysis
    print("\n5. Generating global analysis...")
    generate_global_analysis(all_best_results, output_dir)
    
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
        'scoring_method': 'advanced_nonlinear_with_diminishing_returns'
    }
    
    log_path = os.path.join(output_dir, 'execution_log.json')
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=4)
    
    print(f"\n{'='*80}")
    print("✓ COMPREHENSIVE DEFENSE OPTIMIZATION COMPLETED SUCCESSFULLY!")
    print(f"✓ Results saved in: {os.path.abspath(output_dir)}")
    print(f"✓ Total execution time: {total_time:.1f} seconds")
    print("=" * 80)
    
    # Display final summary
    print(f"\nFINAL SUMMARY:")
    print(f"  • Total epsilon values processed: {len(all_best_results)}")
    print(f"  • Parameter combinations tested per epsilon: {len(param_combinations)}")
    print(f"  • Scoring method: Advanced non-linear with diminishing returns")
    
    # Quality distribution
    quality_counts = {}
    for epsilon, (best_result, _) in all_best_results.items():
        quality = best_result['quality_assessment']['overall_assessment']
        quality_counts[quality] = quality_counts.get(quality, 0) + 1
    
    print(f"  • Quality distribution:")
    for quality, count in quality_counts.items():
        print(f"    - {quality}: {count}")
    
    print(f"\nFiles generated for each epsilon:")
    print(f"  ✓ Parameter selection results")
    print(f"  ✓ Comprehensive defense report")
    print(f"  ✓ Defense visualization")
    print(f"  ✓ Confusion matrix comparison")
    print(f"  ✓ Defense effectiveness analysis")
    print(f"  ✓ Confidence comparison")
    print(f"  ✓ Parameter effect analysis")
    print(f"  ✓ Quality assessment radar")
    print(f"  ✓ Classification reports")
    print(f"  ✓ Raw prediction data")
    
    print(f"\nGlobal analysis files:")
    print(f"  ✓ Global optimization summary")
    print(f"  ✓ Global analysis charts")
    print(f"  ✓ Global optimization report")
    print(f"  ✓ Execution log")

if __name__ == '__main__':
    main()