#IMPORT STATEMENTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Data/AML_world/LI-Small_Trans.csv")

#features
print(df.columns)
print(df.head(5))
print(df.shape)

#check datset composition
print(df.isna().sum())
print(df.duplicated().sum())
print(df.drop_duplicates(inplace=True))

#payment format in money laundering
df["Payment Format"].unique()
payment_format=df[df["Is Laundering"]==1]["Payment Format"].value_counts()


plt.figure(figsize=(15,4))
plt.subplot(1,2,1)
sns.barplot(x=payment_format.index,y=payment_format.values)

plt.subplot(1,2,2)
plt.pie(payment_format.values,labels=payment_format.index,wedgeprops=dict(width=0.4)); #use semicolon to avoid texts in O/P
plt.title("Payment Format Distribution in money laundering")
plt.show()

#payment format in not money laundering

df["Payment Format"].unique()
payment_format=df[df["Is Laundering"]==0]["Payment Format"].value_counts()

plt.figure(figsize=(15,4))
plt.subplot(1,2,1)
sns.barplot(x=payment_format.index,y=payment_format.values)

plt.subplot(1,2,2)
plt.pie(payment_format.values,labels=payment_format.index,wedgeprops=dict(width=0.4)); #use semicolon to avoid texts in O/P
plt.title("Payment Format Distribution in clean activities")
plt.show()

#correlation matrix
cr=df.corr(numeric_only=True)
sns.heatmap(cr,annot=True,cmap="Blues")
plt.show()

#show amount of is laundering
print(df["Is Laundering"].value_counts())
sns.countplot(data=df, x='Is Laundering')
plt.show()

#check how many transfers are from different banks
same_bank_count = (df['From Bank'] == df['To Bank']).sum()  # Count same-bank transactions
cross_bank_count = (df['From Bank'] != df['To Bank']).sum()  # Count cross-bank transactions

print(f"Number of same-bank transactions: {same_bank_count}")
print(f"Number of cross-bank transactions: {cross_bank_count}")

# Plot the counts
plt.figure(figsize=(15, 8))
counts = pd.Series({'Same Bank': same_bank_count, 'Cross Bank': cross_bank_count})
counts.plot(kind='bar', color=['green', 'red'], title='Same Bank vs Cross Bank Transactions')
plt.ylabel('Number of Transactions')
plt.title("NON ML transfer type")
plt.show()

#check how many ML transfers are from different banks
laundering_df = df[df['Is Laundering'] == 1]
same_bank_count = (laundering_df['From Bank'] == laundering_df['To Bank']).sum()  # Count same-bank transactions
cross_bank_count = (laundering_df['From Bank'] != laundering_df['To Bank']).sum()  # Count cross-bank transactions

print(f"Number of same-bank transactions: {same_bank_count}")
print(f"Number of cross-bank transactions: {cross_bank_count}")

# Plot the counts
plt.figure(figsize=(15, 8))
counts = pd.Series({'Same Bank': same_bank_count, 'Cross Bank': cross_bank_count})
counts.plot(kind='bar', color=['green', 'red'], title='Same Bank vs Cross Bank Transactions')
plt.ylabel('Number of Transactions')
plt.title("ML transfer type")
plt.show()