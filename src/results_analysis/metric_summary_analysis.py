import os
import matplotlib.pyplot as plt
import numpy as np
import re
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Parse metrics from the text file
def parse_metrics_txt(filepath):
    metrics_results = {}
    current_model = None

    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith("Model:"):
                current_model = line.split("Model:")[1].strip()
                metrics_results[current_model] = {}
            elif current_model:
                match = re.match(r'(\w+): Mean = ([\d\.nan]+), Std = ([\d\.nan]+)', line)
                if match:
                    metric = match.group(1).lower()
                    mean = float(match.group(2)) if match.group(2) != "nan" else np.nan
                    std = float(match.group(3)) if match.group(3) != "nan" else np.nan
                    metrics_results[current_model][metric] = (mean, std)
    return metrics_results

# Path configuration
input_file = "C:/Users/lucad/OneDrive/Desktop/experiments_results/new_results/rq2_results/rq2_ex1_results/metrics_summary.txt"
results_path = "C:/Users/lucad/OneDrive/Desktop/temp"
n_runs = 20

# Parse the file
metrics_results = parse_metrics_txt(input_file)

# # Models you want to plot
# models_to_plot = [
#     "graphsage_and_mlp",
#     "framework_dgi_and_mlp",
#     "simple_framework_without_flex_fronts",
#     "simple_framework_without_flex_fronts_only_degree",
#     "simple_framework_gin_without_flex_fronts",
#     "complex_framework_without_flex_fronts",
#     "complex_framework_without_flex_fronts_only_degree",
#     "complex_framework_gin_without_flex_fronts",
#     "graphsage_all_features",
#     "graphsage",
#     "gin",
#     "gin_all_features"
# ]
#
# # Optional: Map internal names to cleaner display names
# model_name_map = {
#     "graphsage_and_mlp": "GraphSAGE + MLP",
#     "framework_dgi_and_mlp": "DGI + MLP",
#     "simple_framework_without_flex_fronts": "PAF",
#     "simple_framework_gin_without_flex_fronts": "PAF - GIN variation",
#     "complex_framework_without_flex_fronts": "SEF",
#     "complex_framework_gin_without_flex_fronts": "SEF - GIN variation",
#     "graphsage_all_features": "GraphSAGE TD",
#     "graphsage": "GraphSAGE",
#     "gin": "GIN",
#     "gin_all_features": "GIN TD",
#     "complex_framework_without_flex_fronts_only_degree": "SEF OD",
#     "simple_framework_without_flex_fronts_only_degree": "PAF OD",
# }


models_to_plot = [
    "complex_framework_without_flex_fronts",
    "complex_framework_without_flex_fronts_first_layer_not_frozen",
    "complex_framework_without_flex_fronts_last_layer_not_frozen",
    "complex_framework_without_flex_fronts_GIN_encoder",
    "complex_framework_without_flex_fronts_INFONCE",
    "complex_framework_without_flex_fronts_free_neighbours"
]
model_name_map = {
    "complex_framework_without_flex_fronts": "SEF",
    "complex_framework_without_flex_fronts_first_layer_not_frozen": "SEF-FLNF",
    "complex_framework_without_flex_fronts_last_layer_not_frozen": "SEF-LLNF",
    "complex_framework_without_flex_fronts_GIN_encoder": "SEF-GINenc",
    "complex_framework_without_flex_fronts_INFONCE": "SEF-INFONCE",
    "complex_framework_without_flex_fronts_free_neighbours": "SEF-FN"
}


# Metrics to plot
metrics_to_plot = ["pr_auc", "accuracy", "precision", "recall", "f1", "train_time"]

# Plotting
for metric_name in metrics_to_plot:
    # Filter models
    filtered_names = [name for name in models_to_plot if name in metrics_results]
    means = [metrics_results[name][metric_name][0] for name in filtered_names]
    stds = [metrics_results[name][metric_name][1] for name in filtered_names]

    # Sort by mean descending
    sorted_indices = sorted(range(len(means)), key=lambda i: means[i], reverse=True)
    sorted_internal_names = [filtered_names[i] for i in sorted_indices]
    sorted_display_names = [model_name_map.get(name, name) for name in sorted_internal_names]
    sorted_means = [means[i] for i in sorted_indices]
    sorted_stds = [stds[i] for i in sorted_indices]

    if metric_name == "pr_auc":
        metric_name = "pr-auc"
    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(sorted_display_names, sorted_means, xerr=sorted_stds, capsize=5, color='skyblue', edgecolor='black')
    plt.xlabel(metric_name.upper())
    plt.title(f"{metric_name.upper()} across Models ({n_runs} runs)")
    plt.gca().invert_yaxis()  # Highest value at top
    legend_elements = [
        Patch(facecolor='skyblue', edgecolor='black', label='Mean'),
        Line2D([0], [0], color='black', lw=2, label='Standard Deviation')
    ]
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=True)
    plt.tight_layout()

    # Save & show
    file_path = os.path.join(results_path, f"{metric_name}_comparison_rq2.png")
    plt.savefig(file_path)
    plt.show()
