import os

import torch
import torch_geometric
from sklearn.metrics import ConfusionMatrixDisplay
from torch_geometric.loader import NeighborLoader
import numpy as np
import matplotlib.pyplot as plt

from src.data_preprocessing.preprocess import EllipticDataset
from src.modeling.testing.models_to_test_rq1_ex1 import model_list_rq1_ex1
from src.modeling.testing.models_to_test_rq2_ex1 import model_list_rq2_ex1
from src.modeling.testing.models_to_test_rq3_ex1 import model_list_rq3_ex1
from src.modeling.utils.modeling_utils import train, validate, evaluate
from src.utils import get_data_folder, get_data_sub_folder, get_src_sub_folder

script_dir = get_data_folder()
relative_path_processed = 'processed'
relative_path_trained_model_rq1_ex1 = 'modeling/testing/rq1_ex1_results/trained_models'
relative_path_trained_model_rq2_ex1 = 'modeling/testing/rq2_ex1_results/trained_models'
relative_path_trained_model_rq3_ex1 = 'modeling/testing/rq3_ex1_results/trained_models'
relative_path_trained_dgi = 'modeling/pre_training/topological_pre_training/trained_models'
relative_path_rq1_ex1_results = 'modeling/testing/rq1_ex1_results'
relative_path_rq2_ex1_results = 'modeling/testing/rq2_ex1_results'
relative_path_rq3_ex1_results = 'modeling/testing/rq3_ex1_results'
processed_data_path = get_data_sub_folder(relative_path_processed)
trained_dgi_model_path = get_src_sub_folder(relative_path_trained_dgi)

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch_geometric.is_xpu_available():
    device = torch.device('xpu')
else:
    device = torch.device('cpu')

# Load your dataset
data = EllipticDataset(root=processed_data_path)
data = data[1]

# define the epochs for training
epochs = 1
# define if we want to train or only evaluate
evaluate_only = False

# number of times we train and evaluate each model
n_runs = 2  # set your N here
pr_auc_results = {name: [] for name in model_list_rq1_ex1(data)}  # accumulate PR AUCs

# which rq do we want to answer?
rq_run = 'rq1_ex1'

#early stopping logic
patience = 5
# how much a model should improve for each epoch to consider it an improvement
min_delta = 0.005

if rq_run == 'rq1_ex1':
    # models to test
    models_to_compare = model_list_rq1_ex1(data)
    trained_model_path = get_src_sub_folder(relative_path_trained_model_rq1_ex1)
    results_path = get_src_sub_folder(relative_path_rq1_ex1_results)
elif rq_run == 'rq2_ex1':
    # models to test
    models_to_compare = model_list_rq2_ex1(data)
    trained_model_path = get_src_sub_folder(relative_path_trained_model_rq2_ex1)
    results_path = get_src_sub_folder(relative_path_rq2_ex1_results)
elif rq_run == 'rq3_ex1':
    # models to test
    models_to_compare = model_list_rq3_ex1(data)
    trained_model_path = get_src_sub_folder(relative_path_trained_model_rq3_ex1)
    results_path = get_src_sub_folder(relative_path_rq3_ex1_results)
else:
    raise ValueError(
        f"Invalid option for `rq_run`: '{rq_run}'.\n"
        "Please select one of the following:\n"
        "- rq1_ex1\n"
        "- rq2_ex1\n"
        "- rq3_ex1"
    )

# remove all entries in the subfolders, it is necessary otherwise we may keep information of previous runs
for subdir, dirs, files in os.walk(results_path):
    # we keep the trained models
    if 'trained_models' in subdir:
        continue
    for file in files:
        file_path = os.path.join(subdir, file)
        os.remove(file_path)

for run in range(n_runs):
    print(f"\n======== RUN {run + 1}/{n_runs} ========\n")

    for name, components in models_to_compare.items():
        components['model'] = components['model'].to(device)

    # setting the early stopping value for each model
    best_auc_pr = {name: 0 for name in models_to_compare}
    epochs_no_improve = {name: 0 for name in models_to_compare}
    early_stop_flags = {name: False for name in models_to_compare}

    if not evaluate_only:
        # Run training
        file_path_training_results = os.path.join(results_path, f"training_log/training_log_per_epoch_run_{run + 1}.txt")
        with open(file_path_training_results, "w") as file:
            for epoch in range(epochs):

                #train the framework
                log = (f"\n\n---TRAINING--- Epoch {epoch + 1:02d}\n")
                print(log)
                file.write(log)
                #train the models, framework need a special variable
                for name, components in models_to_compare.items():
                    # Skip training if early stopped
                    if early_stop_flags[name]:
                        continue

                    if 'framework' in name:
                        loss_gnn = train(components['train_set'], components['model'], components['optimizer'], device, components['criterion'], True)
                    else:
                        loss_gnn = train(components['train_set'], components['model'], components['optimizer'], device,
                                         components['criterion'], False)
                    log = (f"Loss {name}: {loss_gnn:.6f}\n")
                    print(log)
                    file.write(log)

                #validation
                log = (
                    f"---VALIDATION--- Epoch {epoch + 1:02d}\n")
                file.write(log)

                for name, components in models_to_compare.items():
                    # Skip validation if early stopped
                    if early_stop_flags[name]:
                        continue

                    if 'framework' in name:
                        accuracy_gnn, precision_gnn, recall_gnn, f1_gnn, auc_pr_gnn = validate(components['val_set'], components['model'], device, True)
                    else:
                        accuracy_gnn, precision_gnn, recall_gnn, f1_gnn, auc_pr_gnn = validate(components['val_set'], components['model'], device,
                                                                                False)
                    # Logging
                    log = (
                        f"{name} Metrics --- Accuracy: {accuracy_gnn:.4f}, Precision: {precision_gnn:.4f}, Recall: {recall_gnn:.4f}, "
                        f"F1: {f1_gnn:.4f}, AUC-PR: {auc_pr_gnn:.4f}\n"
                    )
                    print(log)
                    file.write(log)

                    # Early stopping logic
                    if auc_pr_gnn > best_auc_pr[name] + min_delta:
                        best_auc_pr[name] = auc_pr_gnn
                        epochs_no_improve[name] = 0
                    else:
                        epochs_no_improve[name] += 1
                        if epochs_no_improve[name] >= patience:
                            log = f"Early stopping {name} at epoch {epoch + 1}\n"
                            print(log)
                            file.write(log)
                            early_stop_flags[name] = True

        for name, components in models_to_compare.items():
            torch.save(components['model'].state_dict(), os.path.join(trained_model_path, f'{name}_gnn_trained.pth'))
    else:
        for name, components in models_to_compare.items():
            components['model'].load_state_dict(
                torch.load(os.path.join(trained_model_path, f'{name}_gnn_trained.pth'), map_location=device))

    # Inference
    print("\n----EVALUATION----\n")
    file_path_evaluation_results = os.path.join(results_path, f"evaluation/evaluation_performance_metrics_run{run + 1}.txt")
    file_path_evaluation_results_confusion_matrix = os.path.join(results_path, f"confusion_matrix/evaluation_performance_metrics_run{run + 1}.txt")
    with open(file_path_evaluation_results, "w") as f:
        f.write("----EVALUATION----\n")
        for name, components in models_to_compare.items():
            if 'framework' in name:
                accuracy, precision, recall, f1, pr_auc, confusion_matrix_model, pr_auc_curve, fig_pr_curve = evaluate(components['model'], components['test_set'], device, name, True)
            else:
                accuracy, precision, recall, f1, pr_auc, confusion_matrix_model, pr_auc_curve, fig_pr_curve = evaluate(components['model'], components['test_set'], device,
                                                                                name, False)
            f.write(f"----{name}----\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
            f.write(f"pr_auc Score (class 0): {pr_auc:.4f}\n")

            # Save PR AUC for this run
            pr_auc_results[name].append(pr_auc)

            disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_model)
            disp.plot()
            plt.title(f'Confusion Matrix {name}')
            plt.savefig(os.path.join(results_path, f"confusion_matrix/confusion_matrix_{name}_plot_run{run + 1}.png"))
            print(confusion_matrix_model)

            fig_pr_curve.savefig(os.path.join(results_path, f"precision_recall_curve/precision_recall_curve_{name}_plot_run{run + 1}.png"))



    # make sure that the cache is empty between runs
    torch.cuda.empty_cache()

# Save and plot average + std PR AUC
file_path_evaluation_over_all_results = os.path.join(results_path, f"pr_auc_summary.txt")
with open(file_path_evaluation_over_all_results, "w") as f:
    for name, scores in pr_auc_results.items():
        mean_auc = np.mean(scores)
        std_auc = np.std(scores)
        f.write(f"{name} - Mean PR AUC: {mean_auc:.4f}, Std: {std_auc:.4f}\n")
        print(f"{name}: Mean={mean_auc:.4f}, Std={std_auc:.4f}")

# Plotting (horizontal bar chart with lateral names)
names = list(pr_auc_results.keys())
means = [np.mean(pr_auc_results[name]) for name in names]
stds = [np.std(pr_auc_results[name]) for name in names]

file_path_evaluation_pr_auc_comparison = os.path.join(results_path, "pr_auc_comparison.png")
plt.figure(figsize=(10, 6))
plt.barh(names, means, xerr=stds, capsize=5)
plt.xlabel("PR AUC")
plt.title(f"Average PR AUC over {n_runs} runs")
plt.tight_layout()
plt.savefig(file_path_evaluation_pr_auc_comparison)
plt.show()