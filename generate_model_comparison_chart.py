import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

df = pd.read_csv('model_comparison.csv')
# Convert accuracy percentage to float
df['Accuracy Float'] = df['Accuracy'].str.rstrip('%').astype(float) / 100.0

plt.figure(figsize=(10, 6))
ax = sns.barplot(
    data=df, 
    x='Accuracy Float', 
    y='Model Name', 
    palette='viridis'
)
plt.title('Validation Accuracy Comparison Across Target Architectures', pad=20, fontweight='bold')
plt.xlabel('Accuracy', labelpad=15)
plt.ylabel('')
plt.xlim(0.70, 0.85)

# Annotate bars
for i, p in enumerate(ax.patches):
    ax.annotate(f"{p.get_width():.2%}", 
                (p.get_width() + 0.002, p.get_y() + p.get_height() / 2.), 
                ha='left', va='center', fontweight='bold', color='#333')

plt.tight_layout()
plt.savefig('assets/model_comparison_chart.png', dpi=300)
print("Generated model_comparison_chart.png")
