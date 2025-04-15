import matplotlib.pyplot as plt

# Read the file
epochs = []
losses = []

with open('C://Users//lucad//OneDrive//Desktop//training results//training model//code_v2//training_log_without_topological_feature.txt', 'r') as f:
    for line in f:
        if "Epoch" in line:
            parts = line.strip().split(',')
            epoch = int(parts[0].split()[1])
            loss = float(parts[1].split()[1])
            epochs.append(epoch)
            losses.append(loss)

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(epochs, losses, marker='o', linestyle='-', color='royalblue')
plt.title('Training Loss Over Epochs, without topological features')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.tight_layout()

plt.savefig("C://Users//lucad//OneDrive//Desktop//training results//training model//code_v2//training_log_without_topological_feature.png")

plt.show()