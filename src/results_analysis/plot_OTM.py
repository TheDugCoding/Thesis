import matplotlib.pyplot as plt

# Data
frameworks_perf = ['PAF-GraphSAGE', 'PAF-GIN', 'SEF-GraphSAGE', 'SEF-GIN']
asymptotic_perf = [0.2110, 0.1840, 0.0275, -0.1546]

frameworks_otm = ['PAF', 'SEF']
otm_values = [0.1975, -0.0635]

# Plot Asymptotic Performance
plt.figure(figsize=(8, 4))
plt.bar(frameworks_perf, asymptotic_perf)
plt.title('Asymptotic Performance')
plt.ylabel('Asymptotic Performance')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("asymptotic.jpg", bbox_inches="tight", dpi=300)
plt.show()

# Plot OTM
plt.figure(figsize=(6, 4))
plt.bar(frameworks_otm, otm_values, color='orange')
plt.title('OTM')
plt.ylabel('OTM Value')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("OTM.jpg", bbox_inches="tight", dpi=300)
plt.show()