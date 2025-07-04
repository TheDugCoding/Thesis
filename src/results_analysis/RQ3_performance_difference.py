import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

# Data for the differences in performance
sample_sizes = [20, 100, 500, 1000, 2000, 5000, 10000]

# Differences in performance from the two tables
gin_pafgin_diff = [0.0197, 0.0138, 0.0065, 0.0661, 0.1002, 0.0757, 0.0783]
graphsage_paf_diff = [-0.0033, 0.0541, 0.0265, 0.0469, 0.0976, 0.0793, 0.0966]

# Calculate the Pearson correlation coefficient and p-value
correlation, p_value = pearsonr(gin_pafgin_diff, graphsage_paf_diff)

# Plot
plt.figure(figsize=(10, 6))

# Plot for the first table (GIN vs PAF-GIN)
plt.plot(sample_sizes, gin_pafgin_diff, marker='o', label="GIN vs PAF-GIN", color='blue')

# Plot for the second table (GraphSAGE vs PAF)
plt.plot(sample_sizes, graphsage_paf_diff, marker='o', label="GraphSAGE vs PAF", color='orange')

# Customize plot
plt.title(f'Difference in PR AUC Performance for Different Models\nPearson Correlation: {correlation:.2f}')
plt.xlabel('Sample Size')
plt.ylabel('Difference in PR AUC')
plt.xscale('log')  # Log scale for sample size
plt.legend()


# Show plot
plt.grid(True)
plt.savefig("pair_gin_pag_gin_graphsage_paf")
plt.show()

# Print the Pearson correlation coefficient and p-value
print(f"Pearson Correlation Coefficient: {correlation:.2f}")
print(f"P-value: {p_value:.4f}")
