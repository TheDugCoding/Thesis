import os
import re
import matplotlib.pyplot as plt
import numpy as np

# Path where your metrics_summary files are stored
folder_path = "C:/Users/lucad/OneDrive/Desktop/thesis/code/Thesis/src/modeling/testing/rq3_ex1_results"  # change to your actual path
pattern = r"metrics_summary_(\d+)\.txt"

# Store results as: results[file][model][metric] = (mean, std)
results = {}

for filename in os.listdir(folder_path):
    match = re.match(pattern, filename)
    if match:
        run_label = filename
        filepath = os.path.join(folder_path, filename)
        with open(filepath, 'r') as f:
            lines = f.readlines()

        current_model = None
        for line in lines:
            line = line.strip()
            if line.startswith("Model:"):
                current_model = line.split(":")[1].strip()
                if run_label not in results:
                    results[run_label] = {}
                results[run_label][current_model] = {}
            elif current_model and line and "TRAIN_TIME" not in line:
                parts = line.strip().split(":")
                if len(parts) >= 2:
                    metric = parts[0].strip()
                    match_values = re.findall(r"Mean = ([\d\.nan]+), Std = ([\d\.nan]+)", parts[1])
                    if match_values:
                        mean, std = match_values[0]
                        if mean != 'nan':
                            results[run_label][current_model][metric] = (float(mean), float(std))

# Get all unique metrics
metrics = sorted({metric for file in results.values() for model in file.values() for metric in model})

# Plotting for each metric
for metric in metrics:
    plt.figure(figsize=(10, 6))
    bar_width = 0.15
    run_labels = sorted(results.keys())
    model_names = list(next(iter(results.values())).keys())
    x = np.arange(len(model_names))

    for i, run_label in enumerate(run_labels):
        means = [results[run_label][model].get(metric, (0, 0))[0] for model in model_names]
        stds = [results[run_label][model].get(metric, (0, 0))[1] for model in model_names]

        plt.bar(x + i * bar_width, means, bar_width, yerr=stds, label=f"{run_label}", capsize=5)

    plt.xticks(x + bar_width * (len(run_labels) - 1) / 2, model_names, rotation=45)
    plt.ylabel(metric)
    plt.title(f"Comparison of {metric} Across Runs")
    plt.legend(title="Metrics File")
    plt.tight_layout()
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.savefig(f"{metric}_comparison_across_runs.png")
    plt.show()