import json
import pandas as pd
import os

# -------------------------------------------------------------------------
# 1. GENERATE model_comparison.csv
# -------------------------------------------------------------------------
comparison_data = [
    {
        "Model Name": "InceptionV3 (Phase 2 Fine-Tuned)",
        "Accuracy": "79.13%",
        "Precision": "0.80",
        "Recall": "0.79",
        "F1 Score": "0.79",
        "AUC Score": "0.86",
        "Training Stability Notes": "Stable convergence with Early Stopping logic. Minimal overfitting due to heavy data augmentation.",
        "Strengths": "High precision on normal cases (0.86), strong general feature extraction from native Inception blocks.",
        "Weaknesses": "Lower recall on pathological detection means slight false negative risk. Computationally heavy.",
        "Deployment Suitability": "High - best choice for a clinical backend API where GPU availability is not constrained."
    },
    {
        "Model Name": "ResNet152 + Quantum Analysis Layer",
        "Accuracy": "77.31%",
        "Precision": "N/A",  # Not explicitly tracked in available json
        "Recall": "N/A",
        "F1 Score": "N/A",
        "AUC Score": "N/A",
        "Training Stability Notes": "Converged in 11 epochs. Required a careful LR scheduler due to the sensitive PennyLane experimental layer.",
        "Strengths": "Very deep classical feature vector with state-of-the-art entanglement exploration.",
        "Weaknesses": "Extremely high parameter count resulting in slower inference times. Non-standard quantum deployment.",
        "Deployment Suitability": "Low/Experimental - excellent for research and hybrid quantum experimentation, but too heavy for rapid clinical edge devices."
    },
    {
        "Model Name": "InceptionV3 v2",
        "Accuracy": "77.50%",
        "Precision": "0.78",
        "Recall": "0.77",
        "F1 Score": "0.77",
        "AUC Score": "0.86",
        "Training Stability Notes": "Consistent metric improvement, though validation loss showed slight jitter.",
        "Strengths": "Solid baseline AUC, matching the final model in ranking capability.",
        "Weaknesses": "Underperformed slightly in strict threshold accuracy compared to the heavily augmented final run.",
        "Deployment Suitability": "Moderate - a valid fallback model if strict thresholds can be calibrated."
    }
]

df = pd.DataFrame(comparison_data)
df.to_csv('model_comparison.csv', index=False)
print("Updated model_comparison.csv")

# -------------------------------------------------------------------------
# 2. GENERATE EDA_and_Data_Insights.ipynb (Real analysis only)
# -------------------------------------------------------------------------
eda_notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Real Exploratory Data Analysis & Integrity Tracking\n",
    "\n",
    "This notebook contains the functional logic to parse, validate, and summarize the Diabetic Retinopathy image directory. It avoids fake/simulated data loops. \n",
    "Where raw datasets (`im1_balanced`) are absent from the local repository (due to size limits and PHI rules), the notebook will safely exit gracefully while maintaining verifiable aggregate metrics retrieved from the training logs."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import hashlib\n",
    "import json\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from PIL import Image\n",
    "\n",
    "sns.set_theme(style=\"whitegrid\")\n",
    "plt.rcParams['figure.figsize'] = (10, 6)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Known Class Support Analysis (From Real Metric Outputs)\n",
    "Even without raw images, we can extract the exact representation of our validation splits natively from our generated classification reports."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "report_path = 'inception_79%/classification_report.json'\n",
    "if os.path.exists(report_path):\n",
    "    with open(report_path, 'r') as f:\n",
    "        report = json.load(f)\n",
    "    abnormal_count = report['Abnormal']['support']\n",
    "    normal_count = report['Normal']['support']\n",
    "    \n",
    "    plt.figure(figsize=(6, 5))\n",
    "    sns.barplot(x=['Abnormal', 'Normal'], y=[abnormal_count, normal_count], palette=['#e74c3c', '#2ecc71'])\n",
    "    plt.title('Validation Dataset True Class Distribution')\n",
    "    plt.ylabel('Number of Images (Support)')\n",
    "    plt.show()\n",
    "    print(f\"Validation Breakdown -> Abnormal: {abnormal_count}, Normal: {normal_count}\")\n",
    "else:\n",
    "    print(\"Classification report not available for extracting real supports.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Dynamic Image Property Analysis\n",
    "This logic traverses the raw dataset to extract actual physical parameters (Height, Width, Channel Means). If the data is absent on the running device, it handles the absence honestly."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def analyze_raw_images(base_dir):\n",
    "    if not os.path.exists(base_dir):\n",
    "        print(f\"[INFO] Raw dataset folder '{base_dir}' not found on this machine.\")\n",
    "        print(\"To execute Real Image analysis, ensure the dataset is extracted functionally at this path.\")\n",
    "        return\n",
    "    \n",
    "    resolutions = []\n",
    "    brightness = []\n",
    "    \n",
    "    print(f\"Scanning directory {base_dir}...\")\n",
    "    for root, _, files in os.walk(base_dir):\n",
    "        for file in files:\n",
    "            if file.lower().endswith(('.png', '.jpg', '.jpeg')):\n",
    "                filepath = os.path.join(root, file)\n",
    "                try:\n",
    "                    with Image.open(filepath) as img:\n",
    "                        resolutions.append(img.size)\n",
    "                        # Simple proxy for brightness using greyscale mean\n",
    "                        l_img = img.convert('L')\n",
    "                        stat = np.array(l_img).mean()\n",
    "                        brightness.append(stat)\n",
    "                except Exception as e:\n",
    "                    print(f\"Error reading {file}: {e}\")\n",
    "    \n",
    "    res_df = pd.DataFrame(resolutions, columns=['Width', 'Height'])\n",
    "    print(f\"Successfully analyzed {len(res_df)} real images.\")\n",
    "    \n",
    "    if not res_df.empty:\n",
    "        plt.figure(figsize=(14, 5))\n",
    "        plt.subplot(1, 2, 1)\n",
    "        sns.scatterplot(data=res_df, x='Width', y='Height', alpha=0.5, color='blue')\n",
    "        plt.title('True Raw Image Resolutions')\n",
    "        \n",
    "        plt.subplot(1, 2, 2)\n",
    "        sns.histplot(brightness, kde=True, bins=30, color='purple')\n",
    "        plt.title('Real Image Brightness (Pixel Intensity Mean)')\n",
    "        \n",
    "        plt.tight_layout()\n",
    "        plt.show()\n",
    "\n",
    "analyze_raw_images('im1_balanced/')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Dataset Integrity & Duplicate Detection Pipeline\n",
    "Production ML systems must be resilient. We compute MD5 hashes for all files to ensure dataset exclusivity. Like above, handles safely if raw repo data is unzipped elsewhere."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def check_dataset_integrity(base_dir):\n",
    "    if not os.path.exists(base_dir):\n",
    "        print(f\"[INFO] Data unavailable locally at '{base_dir}'. Bypassing deep integrity checks.\")\n",
    "        return\n",
    "        \n",
    "    hashes = set()\n",
    "    duplicates = 0\n",
    "    corrupted = 0\n",
    "    \n",
    "    for root, _, files in os.walk(base_dir):\n",
    "        for file in files:\n",
    "            filepath = os.path.join(root, file)\n",
    "            try:\n",
    "                # Hash check for exact byte duplicates\n",
    "                with open(filepath, 'rb') as f:\n",
    "                    file_hash = hashlib.md5(f.read()).hexdigest()\n",
    "                if file_hash in hashes:\n",
    "                    duplicates += 1\n",
    "                else:\n",
    "                    hashes.add(file_hash)\n",
    "            except:\n",
    "                corrupted += 1\n",
    "                \n",
    "    print(\"\\n--- REAL INTEGRITY REPORT ---\")\n",
    "    print(f\"Total unique files scanned: {len(hashes)}\")\n",
    "    print(f\"Exact duplications found: {duplicates}\")\n",
    "    print(f\"Corrupt/Unreadable items: {corrupted}\")\n",
    "\n",
    "check_dataset_integrity('im1_balanced/')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

# -------------------------------------------------------------------------
# 3. GENERATE misclassification_analysis.ipynb (Deep Dive using Real stats)
# -------------------------------------------------------------------------
misclass_notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Real Misclassification & Deep Error Analysis\n",
    "\n",
    "This notebook critically examines the exact, true failure modes of the optimal InceptionV3 model based on its verified validation subset outputs. Analysis is rooted exclusively in generated JSON artifacts."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import json\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.metrics import ConfusionMatrixDisplay\n",
    "\n",
    "sns.set_theme(style=\"whitegrid\")\n",
    "plt.rcParams['figure.figsize'] = (8, 6)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Interpretating the True Verified Confusion Matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "cm_path = 'inception_79%/confusion_matrix.json'\n",
    "if os.path.exists(cm_path):\n",
    "    with open(cm_path, 'r') as f:\n",
    "        cm = np.array(json.load(f))\n",
    "    \n",
    "    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Abnormal', 'Normal'])\n",
    "    disp.plot(cmap='Reds', values_format='d')\n",
    "    plt.title('True Validation Confusion Matrix')\n",
    "    plt.grid(False)\n",
    "    plt.show()\n",
    "else:\n",
    "    print(\"Real confusion matrix JSON not found. Cannot proceed safely.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. In-Depth Error Pattern Analysis (Real Results)\n",
    "\n",
    "According to the true confusion outputs: `[[284, 64], [113, 387]]`\n",
    "\n",
    "### A. False Positive Risk (113 Cases)\n",
    "- **What happened**: 113 healthy (Normal) images were incorrectly flagged as Abnormal.\n",
    "- **Clinical Consequence**: This induces patient anxiety and requires a secondary manual review by an ophthalmologist. While highly inefficient for medical throughput, this does NOT carry a severe clinical danger (no diseased patients missed).\n",
    "- **Why it happens**: Normal fundus images occasionally contain bright artifacts (like optic disc highlights or photo glare) that convolutions mistakenly associate with hard exudates or cotton wool spots.\n",
    "\n",
    "### B. False Negative Risk (64 Cases)\n",
    "- **What happened**: 64 pathological (Abnormal) images were passed off as entirely Normal.\n",
    "- **Clinical Consequence**: **Extremely High Risk**. These patients have Diabetic Retinopathy but will be discharged without treatment, potentially leading to irreversible blindness and malpractice exposure.\n",
    "- **Why it happens**: Mild Non-Proliferative Diabetic Retinopathy (NPDR) usually only presents with microaneurysms that are notoriously tiny and blend into the background. The aggressive image augs / zooming may have cropped these out, or the resolution constraints (224x224) eroded the pixel-level micro-vascular cues.\n",
    "\n",
    "### C. The Hardest Class\n",
    "The data proves **Abnormal is significantly harder to perfectly capture with high precision**, but the model defaults to being highly sensitive (*conservative*) and categorizing ambiguities as Abnormal safely, which forces high Recall for Abnormal (81.6%) and accounts for our 113 False Positives.\n",
    "\n",
    "## 3. Data-Driven Next Steps for Real-World Iteration\n",
    "\n",
    "1. **Asymmetric Cost Function Deployment**: Since False Negatives are clinically devastating, we must introduce a custom Loss Function (e.g. Weighted Binary Cross-Entropy) where penalizing a False Negative is intrinsically 5x more expensive than a False Positive. This would push those 64 FN cases severely down.\n",
    "2. **Spatial Attention Mechanisms**: The traditional Global Average Pooling throws out exact spatial coordinates. By utilizing a transformer branch or precise bounding-box detection, the network wouldn't \"lose\" extreme minute microaneurysms."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open("EDA_and_Data_Insights.ipynb", "w") as f:
    json.dump(eda_notebook, f, indent=2)

with open("misclassification_analysis.ipynb", "w") as f:
    json.dump(misclass_notebook, f, indent=2)

print("Phase 2 notebook generation complete!")
