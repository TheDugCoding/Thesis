import re
from collections import defaultdict
import matplotlib.pyplot as plt

# Path to your log file
log_path = 'C:/Users/lucad/OneDrive/Desktop/thesis/code/Thesis/src/modeling/testing/training_log_per_epoch.txt'

# Containers
epoch_data = defaultdict(lambda: {'train': {}, 'val': {}})
current_epoch = None
mode = None  # 'train' or 'val'

# Patterns
epoch_pattern = re.compile(r'---(TRAINING|VALIDATION)--- Epoch (\d+)')
loss_pattern = re.compile(r'Loss (\w+): ([\d.]+)')
metric_pattern = re.compile(r'(\w+) Metrics --- Accuracy: ([\d.]+), Recall: ([\d.]+), F1: ([\d.]+), AUC-PR: ([\d.]+)')

with open(log_path, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        epoch_match = epoch_pattern.match(line)
        if epoch_match:
            mode, epoch = epoch_match.groups()
            current_epoch = int(epoch)
            continue

        loss_match = loss_pattern.match(line)
        if loss_match:
            model, loss = loss_match.groups()
            epoch_data[current_epoch]['train'][model] = float(loss)
            continue

        metric_match = metric_pattern.match(line)
        if metric_match:
            model, acc, recall, f1, auc = metric_match.groups()
            epoch_data[current_epoch]['val'][model] = {
                'Accuracy': float(acc),
                'Recall': float(recall),
                'F1': float(f1),
                'AUC-PR': float(auc)
            }

# ---- PRINT RESULTS ----
for epoch in sorted(epoch_data.keys()):
    print(f"\n📅 Epoch {epoch}")
    print("  🔧 Training Losses:")
    for model, loss in epoch_data[epoch]['train'].items():
        print(f"    - {model}: {loss:.4f}")

    print("  🧪 Validation Metrics:")
    for model, metrics in epoch_data[epoch]['val'].items():
        print(f"    - {model}: "
              f"Acc={metrics['Accuracy']:.4f}, "
              f"Rec={metrics['Recall']:.4f}, "
              f"F1={metrics['F1']:.4f}, "
              f"AUC-PR={metrics['AUC-PR']:.4f}")

# Extract all unique models
all_models = set()
for epoch_info in epoch_data.values():
    all_models.update(epoch_info['train'].keys())
    all_models.update(epoch_info['val'].keys())
all_models = sorted(all_models)

epochs = sorted(epoch_data.keys())
loss_dict = {model: [] for model in all_models}
f1_dict = {model: [] for model in all_models}
auc_pr_dict = {model: [] for model in all_models}

for epoch in epochs:
    for model in all_models:
        # Training loss
        loss_dict[model].append(epoch_data[epoch]['train'].get(model, None))

        # Validation metrics
        val_metrics = epoch_data[epoch]['val'].get(model, {})
        f1_dict[model].append(val_metrics.get('F1', None))
        auc_pr_dict[model].append(val_metrics.get('AUC-PR', None))


# ---- PLOTTING ----

def plot_metric(data_dict, title, ylabel):
    plt.figure(figsize=(10, 5))
    for model in all_models:
        values = data_dict[model]
        if any(v is not None for v in values):
            plt.plot(epochs, values, label=model)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# 1. Training Loss
plot_metric(loss_dict, 'Training Loss per Epoch', 'Loss')

# 2. F1 Score
plot_metric(f1_dict, 'Validation F1 Score per Epoch', 'F1 Score')

# 3. AUC-PR
plot_metric(auc_pr_dict, 'Validation AUC-PR per Epoch', 'AUC-PR')