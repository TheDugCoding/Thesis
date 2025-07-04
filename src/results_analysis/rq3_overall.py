import matplotlib.pyplot as plt
import re
from collections import defaultdict

# Full paths to your files
files = [
    ("C:/Users/lucad/OneDrive/Desktop/experiments_results/new_results/rq3_results/rq3_summary/metrics_summary_20.txt", 20),
    ("C:/Users/lucad/OneDrive/Desktop/experiments_results/new_results/rq3_results/rq3_summary/metrics_summary_100.txt", 100),
    ("C:/Users/lucad/OneDrive/Desktop/experiments_results/new_results/rq3_results/rq3_summary/metrics_summary_500.txt", 500),
    ("C:/Users/lucad/OneDrive/Desktop/experiments_results/new_results/rq3_results/rq3_summary/metrics_summary_1000.txt", 1000),
    ("C:/Users/lucad/OneDrive/Desktop/experiments_results/new_results/rq3_results/rq3_summary/metrics_summary_2000.txt", 2000),
    ("C:/Users/lucad/OneDrive/Desktop/experiments_results/new_results/rq3_results/rq3_summary/metrics_summary_5000.txt", 5000),
    ("C:/Users/lucad/OneDrive/Desktop/experiments_results/new_results/rq3_results/rq3_summary/metrics_summary_10000.txt", 10000),
]

model_name_map = {
    "simple_framework": "PAF",
    "complex_framework_without_flex_fronts": "SEF",
    "graphsage": "GraphSAGE",
    "gin": "GIN",
    "complex_framework_gin_without_flex_fronts": "SEF - GIN variation",
    "simple_framework_gin_without_flex_fronts": "PAF - GIN variation",
    "graphsage_all_features": "GraphSAGE - TD",
}

# List of models you want to include in the plot
models_to_plot = [
    "simple_framework",
    "complex_framework_without_flex_fronts",
    "graphsage",
    "gin",
    "complex_framework_gin_without_flex_fronts",
    "simple_framework_gin_without_flex_fronts",
    "graphsage_all_features"
]

# Collect data
pr_auc_data = defaultdict(list)

for file_path, sample_size in files:
    with open(file_path, 'r') as f:
        content = f.read()

    models = re.findall(r"Model: (.*?)\n(.*?)(?=\nModel:|\Z)", content, re.DOTALL)

    for model_name, metrics_block in models:
        # Remove suffix like _20, _100, etc.
        base_model_name = re.sub(r'_\d+$', '', model_name)

        if base_model_name not in models_to_plot:
            continue  # Skip models not in the desired list

        pr_auc_match = re.search(r"PR_AUC: Mean = ([\d.]+), Std = ([\d.]+)", metrics_block)
        if pr_auc_match:
            mean = float(pr_auc_match.group(1))
            std = float(pr_auc_match.group(2))
            pr_auc_data[base_model_name].append((sample_size, mean, std))

# Plot
plt.figure(figsize=(8, 6))

for model_name in models_to_plot:
    if model_name in pr_auc_data:
        values = sorted(pr_auc_data[model_name], key=lambda x: x[0])
        x = [v[0] for v in values]
        y = [v[1] for v in values]
        yerr = [v[2] for v in values]

        label = model_name_map.get(model_name, model_name)
        linestyle = '--' if label in ["GraphSAGE", "PAF", "SEF"] else '-'  # Dotted for GraphSAGE and PAF - GIN variation, solid otherwise
        plt.errorbar(x, y, yerr=yerr, label=label, capsize=4, marker='o', linestyle=linestyle)

plt.xlabel("Training Sample Size")
plt.ylabel("PR-AUC")
plt.title("PR-AUC vs Training Sample Size")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
plt.grid(True)
plt.tight_layout()
plt.savefig("pr_auc_vs_sample_size_gra_all.jpg", bbox_inches="tight")
plt.show()