import os
import re
import statistics
from collections import defaultdict

# Directory containing the files (change this if needed)
directory = "C:/Users/lucad/OneDrive/Desktop/experiments_results/rq2_ex1_results(new)/rq2_ex1_results/evaluation"

# File prefix and number of runs
file_prefix = "evaluation_performance_metrics_run"
num_runs = 20

# Metrics to extract
metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "pr_auc Score (class 0)"]

# Dictionary to store metrics per model
results = defaultdict(lambda: defaultdict(list))

# Regex patterns
model_pattern = re.compile(r"----(.+?)----")
metric_patterns = {
    metric: re.compile(rf"{re.escape(metric)}:\s*([0-9.]+)") for metric in metrics
}

# Go through each file
for i in range(1, num_runs + 1):
    filename = os.path.join(directory, f"{file_prefix}{i}.txt")
    if not os.path.exists(filename):
        print(f"[!] File not found: {filename}")
        continue

    with open(filename, "r") as f:
        current_model = None
        for line in f:
            line = line.strip()
            model_match = model_pattern.match(line)
            if model_match:
                current_model = model_match.group(1).strip()
                continue
            for metric, pattern in metric_patterns.items():
                match = pattern.match(line)
                if match and current_model:
                    value = float(match.group(1))
                    results[current_model][metric].append(value)

# Print the results
print("\n=== Aggregated Metrics per Model (Mean ± Std) ===")
for model, model_metrics in results.items():
    print(f"\nModel: {model}")
    for metric in metrics:
        values = model_metrics[metric]
        if values:
            mean = statistics.mean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            print(f"  {metric}: {mean:.4f} ± {std:.4f}")
        else:
            print(f"  {metric}: No data")