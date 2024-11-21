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