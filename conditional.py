import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report 

# Read the file
df = pd.read_csv(r"/Users/homerliu/Desktop/coding/titanic/Titanic-Dataset.csv")
df = df.drop(['Name', 'Cabin'], axis=1)
# print(df.head())
# print(df.columns)

# Set the label on category
lb = LabelEncoder()
df['Sex'] = lb.fit_transform(df['Sex']) # male=1, female=0
df['Embarked'] = lb.fit_transform(df['Embarked'])
df['Ticket'] = lb.fit_transform(df['Ticket'])

# Conditional Probability
# P(Survived = yes | Sex = male)
sum_male = 0
for i in range(0, len(df['Sex'])):
	if df['Sex'][i] == 1:
		sum_male += 1
sum_survived = 0
for i in range(0, len(df['Sex'])):
	if df['Sex'][i] == 1:
		if df['Survived'][i] == 1:
			sum_survived += 1
print("P(Survived = yes | Sex = male) =", round(sum_survived/sum_male, 2))

# P(Survived = yes | Sex = female)
sum_female = 0
for i in range(0, len(df['Sex'])):
	if df['Sex'][i] == 0:
		sum_female += 1
sum_survived = 0
for i in range(0, len(df['Sex'])):
	if df['Sex'][i] == 0:
		if df['Survived'][i] == 1:
			sum_survived += 1
print("P(Survived = yes | Sex = female) =", round(sum_survived/sum_female, 2))

# P(Survived = yes  | Age >= 60)
sum_age_more_40 = 0
for i in range(0, len(df['Age'])):
	if df['Age'][i] >= 60:
		sum_age_more_40 += 1
sum_survived = 0
for i in range(0, len(df['Age'])):
	if df['Age'][i] >= 60:
		if df['Survived'][i] == 1:
			sum_survived += 1
print("P(Survived = yes | Age >= 60) =", round(sum_survived/sum_age_more_40, 2))

# P(Survived = yes  | Age < 60)
sum_age_less_40 = 0
for i in range(0, len(df['Age'])):
	if df['Age'][i] < 60:
		sum_age_less_40 += 1
sum_survived = 0
for i in range(0, len(df['Age'])):
	if df['Age'][i] < 60:
		if df['Survived'][i] == 1:
			sum_survived += 1
print("P(Survived = yes | Age < 60) =", round(sum_survived/sum_age_less_40, 2))

# P(Survived = yes  | Pclass = 1)
sum_pclass_1 = 0
for i in range(0, len(df['Pclass'])):
	if df['Pclass'][i] == 1:
		sum_pclass_1 += 1
sum_survived = 0
for i in range(0, len(df['Pclass'])):
	if df['Pclass'][i] == 1:
		if df['Survived'][i] == 1:
			sum_survived += 1
print("P(Survived = yes | Pclass = 1) =", round(sum_survived/sum_pclass_1, 2))

# P(Survived = yes  | Pclass = 2)
sum_pclass_2 = 0
for i in range(0, len(df['Pclass'])):
	if df['Pclass'][i] == 2:
		sum_pclass_2 += 1
sum_survived = 0
for i in range(0, len(df['Pclass'])):
	if df['Pclass'][i] == 2:
		if df['Survived'][i] == 1:
			sum_survived += 1
print("P(Survived = yes | Pclass = 2) =", round(sum_survived/sum_pclass_2, 2))

# P(Survived = yes  | Pclass = 3)
sum_pclass_3 = 0
for i in range(0, len(df['Pclass'])):
	if df['Pclass'][i] == 3:
		sum_pclass_3 += 1
sum_survived = 0
for i in range(0, len(df['Pclass'])):
	if df['Pclass'][i] == 3:
		if df['Survived'][i] == 1:
			sum_survived += 1
print("P(Survived = yes | Pclass = 3) =", round(sum_survived/sum_pclass_3, 2))

# P(Survived = yes | Sex = female, Age >= 60, Pclass = 1)
sum = 0
for i in range(0, len(df['Pclass'])):
	if df['Sex'][i] == 0:
		if df['Age'][i] >= 60:
			if df['Pclass'][i] == 1:
				sum += 1
sum_survived = 0
for i in range(0, len(df['Pclass'])):
	if df['Sex'][i] == 0:
		if df['Age'][i] >= 60:
			if df['Pclass'][i] == 1:
				if df['Survived'][i] == 1:
					sum_survived += 1
print("P(Survived = yes | Sex = female, Age >= 60, Pclass = 1) = ",round(sum_survived/sum, 2))












