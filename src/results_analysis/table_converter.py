import re
import pandas as pd

# Define model_name_map
model_name_map = {
    "simple_framework": "PAF",
    "complex_framework_without_flex_fronts": "SEF",
    "graphsage": "GraphSAGE",
    "gin": "GIN",
    "complex_framework_gin_without_flex_fronts": "SEFGINvariation",
    "simple_framework_gin_without_flex_fronts": "PAFGINvariation",
    "graphsage_all_features": "GraphSAGETD",
}

# Define the order of the columns
columns_order = ['Model', 'Precision_mean', 'Precision_std', 'Accuracy_mean', 'Accuracy_std',
                 'Recall_mean', 'Recall_std', 'F1_mean', 'F1_std', 'PR_AUC_mean', 'PR_AUC_std']


# Function to parse the input file and reorder the values
def process_model_file(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    models_data = []
    model_data = {}

    # Parse the file line by line
    for line in lines:
        line = line.strip()

        if line.startswith("Model:"):
            # If a new model starts, save the previous one and reset model_data
            if model_data:
                # Apply the name mapping
                model_name = model_data['Model']
                for key, value in model_name_map.items():
                    if key in model_name:
                        model_data['Model'] = value
                        break
                models_data.append(model_data)
            model_data = {'Model': line.split(": ")[1]}

        elif line.startswith("PR_AUC"):
            # Extract PR_AUC
            pr_auc = re.findall(r"Mean = ([\d\.]+), Std = ([\d\.]+)", line)
            if pr_auc:
                model_data['PR_AUC_mean'], model_data['PR_AUC_std'] = pr_auc[0]

        elif line.startswith("ACCURACY"):
            # Extract Accuracy
            accuracy = re.findall(r"Mean = ([\d\.]+), Std = ([\d\.]+)", line)
            if accuracy:
                model_data['Accuracy_mean'], model_data['Accuracy_std'] = accuracy[0]

        elif line.startswith("PRECISION"):
            # Extract Precision
            precision = re.findall(r"Mean = ([\d\.]+), Std = ([\d\.]+)", line)
            if precision:
                model_data['Precision_mean'], model_data['Precision_std'] = precision[0]

        elif line.startswith("RECALL"):
            # Extract Recall
            recall = re.findall(r"Mean = ([\d\.]+), Std = ([\d\.]+)", line)
            if recall:
                model_data['Recall_mean'], model_data['Recall_std'] = recall[0]

        elif line.startswith("F1"):
            # Extract F1
            f1 = re.findall(r"Mean = ([\d\.]+), Std = ([\d\.]+)", line)
            if f1:
                model_data['F1_mean'], model_data['F1_std'] = f1[0]

    # Don't forget to append the last model
    if model_data:
        model_name = model_data['Model']
        for key, value in model_name_map.items():
            if key in model_name:
                model_data['Model'] = value
                break
        models_data.append(model_data)

    # Convert data to DataFrame
    df = pd.DataFrame(models_data)

    # Reorder columns as per the given order
    df = df[columns_order]

    # Write the output to a file
    df.to_csv(output_file, index=False)


# Input and output file paths
input_file = "C:/Users/lucad/OneDrive/Desktop/experiments_results/new_results/rq3_results/rq3_summary/metrics_summary_10000.txt"  # Replace with your actual file name
output_file = 'reordered_models_data.csv'

# Process the file
process_model_file(input_file, output_file)

print("Processing complete. Output saved to", output_file)
