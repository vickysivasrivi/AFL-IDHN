#!/usr/bin/env python3

"""
Performance Report and Visualization Generator

Author: Vignesh Siva
Date: August 2025
Version: 1.0

This script compares the performance of two trained models (e.g., adaptive vs. 
baseline) by evaluating them on the same test data. It generates a detailed 
side-by-side comparison table and creates several visualizations to highlight 
differences in overall accuracy, per-class F1-scores, and performance on 
minority classes.
"""


import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Import the evaluation functions from your evaluate.py script ---
from evaluate import load_test_data, evaluate_model, CLASS_NAMES

# --- Configuration ---
# Define the paths to your saved model weights
ADAPTIVE_MODEL_PATH = os.path.join("saved_models", "final_global_model_adaptive.weights.h5")
BASELINE_MODEL_PATH = os.path.join("saved_models", "final_global_model_baseline.weights.h5")
DATA_FILE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "data", "Preprocessed", "cicids2017_preprocessed.csv"
)

MINORITY_CLASSES = [
    'Bot', 'Heartbleed', 'Infiltration', 'Web Attack - Brute Force',
    'Web Attack - Sql Injection', 'Web Attack - XSS'
]

def generate_visualizations(adaptive_report: dict, baseline_report: dict):
    """
    Takes the two report dictionaries and generates a summary table and plots.
    """
    # ---- 1. Create and Print the Summary Table ----
    f1_scores = {
        'Class': CLASS_NAMES + ['MACRO AVG F1', 'WEIGHTED AVG F1'],
        'Baseline F1': [baseline_report[c]['f1-score'] for c in CLASS_NAMES] +
                       [baseline_report['macro avg']['f1-score'], baseline_report['weighted avg']['f1-score']],
        'Adaptive F1': [adaptive_report[c]['f1-score'] for c in CLASS_NAMES] +
                       [adaptive_report['macro avg']['f1-score'], adaptive_report['weighted avg']['f1-score']]
    }
    df_summary = pd.DataFrame(f1_scores).round(4)
    df_summary['Winner'] = np.where(df_summary['Adaptive F1'] > df_summary['Baseline F1'], 'Adaptive',
                                    np.where(df_summary['Adaptive F1'] < df_summary['Baseline F1'], 'Baseline', 'Tie'))
    
    print("\n" + "="*80)
    print(" " * 20 + "F1-Score Comparison: Adaptive vs. Baseline")
    print("="*80)
    print(df_summary.to_string(index=False))
    print("="*80 + "\n")
    
    # ---- 2. Plot Overall Performance Comparison ----
    metrics = ['accuracy', 'macro avg', 'weighted avg']
    labels = ['Overall Accuracy', 'Macro Avg F1-Score', 'Weighted Avg F1-Score']
    baseline_scores = [baseline_report[m]['f1-score'] if m != 'accuracy' else baseline_report[m] for m in metrics]
    adaptive_scores = [adaptive_report[m]['f1-score'] if m != 'accuracy' else adaptive_report[m] for m in metrics]
    
    x = np.arange(len(labels)); width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, baseline_scores, width, label='Baseline', color='skyblue')
    rects2 = ax.bar(x + width/2, adaptive_scores, width, label='Adaptive', color='sandybrown')
    ax.set_ylabel('Scores'); ax.set_title('Overall Performance Comparison', fontsize=16)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.legend(); ax.set_ylim(0, 1.1)
    ax.bar_label(rects1, padding=3, fmt='%.4f'); ax.bar_label(rects2, padding=3, fmt='%.4f')
    fig.tight_layout()
    plt.savefig("result_images/summary_performance_comparison.png")
    print("Saved summary performance graph to 'summary_performance_comparison.png'")

    # ---- 3. Plot Per-Class F1-Score Comparison ----
    df_plot_all = pd.DataFrame({
        'Class': CLASS_NAMES,
        'Baseline': [baseline_report[c]['f1-score'] for c in CLASS_NAMES],
        'Adaptive': [adaptive_report[c]['f1-score'] for c in CLASS_NAMES]
    }).melt(id_vars='Class', var_name='Model', value_name='F1-Score')
    plt.figure(figsize=(15, 7)); sns.barplot(data=df_plot_all, x='Class', y='F1-Score', hue='Model', palette=['skyblue', 'sandybrown'])
    plt.title('Per-Class F1-Score Comparison', fontsize=16); plt.xticks(rotation=45, ha="right"); plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout()
    plt.savefig("result_images/per_class_f1_comparison.png")
    print("Saved per-class F1-score graph to 'per_class_f1_comparison.png'")

    # ---- 4. Plot Minority Class F1-Score Comparison ----
    df_minority = pd.DataFrame({
        'Class': MINORITY_CLASSES,
        'Baseline': [baseline_report[c]['f1-score'] for c in MINORITY_CLASSES],
        'Adaptive': [adaptive_report[c]['f1-score'] for c in MINORITY_CLASSES]
    }).melt(id_vars='Class', var_name='Model', value_name='F1-Score')
    plt.figure(figsize=(12, 6)); sns.barplot(data=df_minority, x='Class', y='F1-Score', hue='Model', palette=['skyblue', 'sandybrown'])
    plt.title('F1-Score Comparison on Minority (Rare) Attack Classes', fontsize=16); plt.xticks(rotation=30, ha="right"); plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout()
    plt.savefig("result_images/minority_class_f1_comparison.png")
    print("Saved minority class F1-score graph to 'minority_class_f1_comparison.png'")


if __name__ == "__main__":
    # Check if both model files exist before starting
    if not os.path.exists(ADAPTIVE_MODEL_PATH):
        print(f"FATAL ERROR: Adaptive model file not found at '{ADAPTIVE_MODEL_PATH}'")
        sys.exit(1)
    if not os.path.exists(BASELINE_MODEL_PATH):
        print(f"FATAL ERROR: Baseline model file not found at '{BASELINE_MODEL_PATH}'")
        sys.exit(1)

    # Load the test data ONCE to be used for both evaluations
    test_data = load_test_data(DATA_FILE_PATH)
    
    # Evaluate both models and get their reports as dictionaries
    adaptive_report = evaluate_model(ADAPTIVE_MODEL_PATH, test_data)
    baseline_report = evaluate_model(BASELINE_MODEL_PATH, test_data)

    # Check if evaluations were successful before trying to generate plots
    if adaptive_report and baseline_report:
        sns.set_style("whitegrid")
        generate_visualizations(adaptive_report, baseline_report)
        print("\nAll reports and graphs have been generated successfully.")
    else:
        print("\nCould not generate reports due to an error in model evaluation.")