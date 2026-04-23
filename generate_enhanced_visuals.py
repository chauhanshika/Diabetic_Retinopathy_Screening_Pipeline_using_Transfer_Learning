import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set global seaborn styling for cleaner recruiter-friendly graphs
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['font.family'] = 'sans-serif'

os.makedirs('assets', exist_ok=True)

# 1. Enhanced Confusion Matrix
cm_path = 'inception_79%/confusion_matrix.json'
if os.path.exists(cm_path):
    with open(cm_path, 'r') as f:
        cm = np.array(json.load(f))
    
    plt.figure(figsize=(7, 6))
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                     cbar_kws={'shrink': 0.8}, 
                     xticklabels=['Abnormal', 'Normal'], 
                     yticklabels=['Abnormal', 'Normal'])
    plt.title('Clinical Confusion Matrix', pad=20, fontweight='bold', color='#333333')
    plt.ylabel('Ground Truth (Actual)', labelpad=15)
    plt.xlabel('Model Prediction', labelpad=15)
    plt.tight_layout()
    plt.savefig('assets/enhanced_confusion_matrix.png', dpi=300)
    plt.close()
    print("Saved enhanced_confusion_matrix.png")

# 2. Enhanced ROC Curve
roc_path = 'inception_77.5%/roc_curve_data.json'
if os.path.exists(roc_path):
    with open(roc_path, 'r') as f:
        roc_data = json.load(f)
    
    fpr = np.array(roc_data['binary']['fpr'] if 'binary' in roc_data else roc_data['fpr'])
    tpr = np.array(roc_data['binary']['tpr'] if 'binary' in roc_data else roc_data['tpr'])
    auc_score = roc_data['binary']['auc'] if 'binary' in roc_data else roc_data['auc']
    
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='#e74c3c', lw=3, 
             label=f'InceptionV3 (v2) ROC (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='#7f8c8d', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('Receiver Operating Characteristic (ROC)', pad=20, fontweight='bold', color='#333333')
    plt.xlabel('False Positive Rate', labelpad=15)
    plt.ylabel('True Positive Rate', labelpad=15)
    plt.legend(loc="lower right", frameon=True, fancybox=True, shadow=True)
    plt.tight_layout()
    plt.savefig('assets/enhanced_roc_curve.png', dpi=300)
    plt.close()
    print("Saved enhanced_roc_curve.png")

# Removed training metrics parsing to avoid JSON Decode errors on corrupted earlier experiments.

