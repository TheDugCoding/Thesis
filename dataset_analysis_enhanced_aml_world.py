import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image
from scipy.stats import skew
from matplotlib.transforms import Bbox
import warnings
warnings.filterwarnings("ignore")



raw_df = pd.read_csv("C:/Users/lucad/OneDrive/Desktop/thesis/code/Thesis/Data/enchanced_aml_world/SAML-D.csv")

print(raw_df.shape)
df = raw_df.sample(n=100000, random_state=1)

print(df.head())

sns.countplot(data=df, x='Is_laundering')

plt.figure(figsize=(25, 6))
sns.countplot(data=df, x='Laundering_type')
plt.show()

print(df.columns)
print(df.info())

class_distribution = df['Is_laundering'].value_counts()

plt.figure(figsize=(10, 6))
plt.pie(class_distribution, labels=['Non-Laundering Transactions', 'Suspicious Transactions'], autopct='%1.1f%%', colors=['skyblue', 'lightcoral'])

plt.title('Class Distribution')
plt.axis('equal')

plt.show();

#checking how many unique values there are here
accounts_combined = pd.concat([df['Sender_account'], df['Receiver_account']], axis=0).nunique()

print(f"Number of accounts (): {accounts_combined}")

