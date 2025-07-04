import matplotlib.pyplot as plt

# Read the file
epochs = []
losses = []

with open('C:/Users/lucad/OneDrive/Desktop/temp/training_log_elliptic_with_features_topo_false.txt', 'r') as f:
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
plt.title('DGI Training Loss Over 30 Epochs, trained on Rabobank, ecr_20 and Elliptic++')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.tight_layout()

plt.savefig("C:/Users/lucad/OneDrive/Desktop/temp/plot.png")

plt.show()