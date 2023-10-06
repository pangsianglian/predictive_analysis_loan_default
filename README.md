# predictive_analysis_loan_default
# 1. Introduction 👋
<center><img src="https://www.investopedia.com/thmb/X8yWQhI0Vg3OqsANPIIU5zoaslI=/5894x3901/filters:no_upscale():max_bytes(150000):strip_icc()/loan-text-written-on-wooden-block-with-stacked-coins-955530262-08c17c0eb3634e6daf780bdd9ac1d194.jpg" alt="Loan Picture" width="700" height="700"></center><br>

## Data Set Problems 🤔
👉 This dataset contains information about loan default classification based on borrowers general information and its diagnosis. Machine learning model is needed in order **to predict the loan defaulter from the borrowers** that might be suitable for the lender.

---

## Objectives of Notebook 📌
👉 **This notebook aims to:**
*   Dataset exploration using various types of data visualization.
*   Build various ML models that can predict loan defaulter.

👨‍💻 **The machine learning models used in this project are:** 
1. Linear Logistic Regression
2. Linear Support Vector Machine (SVM)
3. K Neighbours
4. Naive Bayes (Categorical & Gaussian)
5. Decision Tree
6. Random Forest

---

## Data Set Description 🧾

👉 There are **17 variables** in this data set:
*   **2 unique identifier** variables,
*   **7 categorical** variables,
*   **4 continuous** variables, and
*   **4 date** variables.

<br>

👉 The following is the **structure of the data set**.


<table style="width:100%">
<thead>
<tr>
<th style="text-align:center; font-weight: bold; font-size:14px">Variable Name</th>
<th style="text-align:center; font-weight: bold; font-size:14px">Description</th>
<th style="text-align:center; font-weight: bold; font-size:14px">Sample Data</th>
</tr>
</thead>
<tbody>
<tr>
<td><b>customer_id</b></td>
<td>Unique IDs of the customer</td>
<td>CUST-00004912; CUST-00001895; ...</td>
</tr>
<tr>
<td><b>loan_id</b></td>
<td>Unique ID assigned on loan application</td>
<td>LN00004170; LN00000024; ...</td>
</tr>
<tr>
<td><b>loan_type</b></td>
<td>Category of loan <br> (Car Loan, Personal Loan, Home Loan or Education Loan)</td>
<td>Education Loan; Personal Loan; ...</td>
</tr>
<tr>
<td><b>loan_amount</b></td>
<td>Total amount of loan <br> (Unit:INR) </td>
<td>1860; 55886; ...</td>
</tr>
<tr>
<td><b>interest_rate</b></td>
<td>Interest rate of the loan</td>
<td>0.070634819; 0.099123126; ...</td>
</tr>
<tr>
<td><b>loan_term</b></td>
<td>Term of the loan <br> (Unit:Month)</td>
<td>15; 56; ...</td>
</tr>
<tr>
<td><b>employment_type</b></td>
<td>Type of the employment <br> (Full-time, Part-time or Self-employed)</td>
<td>Full-time; Self-employed; ...</td>
</tr>
<tr>
<td><b>income_level</b></td>
<td>Type of the employment <br> (High, Low or Medium)</td>
<td>Low; Medium; ...</td>
</tr>
<tr>
<td><b>credit_score</b></td>
<td>Credit score of customer</td>
<td>833; 514; ...</td>
</tr>
<tr>
<td><b>gender</b></td>
<td>Gender of customer <br> (Male or Female)</td>
<td>Female; Male; ...</td>
</tr>
<tr>
<td><b>marital_status</b></td>
<td>Marital Status of the customer <br> (Single, Married or Divorced)</td>
<td>Single; Divorced; ...</td>
</tr>
<tr>
<td><b>education_level</b></td>
<td>Education level of the customer <br> (High School, Bachelor, Master or PhD)</td>
<td>High School; PhD; ...</td>
</tr>
<tr>
<td><b>application_date</b></td>
<td>Date of the application from the customer</td>
<td>5/4/2018; 17/4/2021; ...</td>
</tr>
<tr>
<td><b>approval_date</b></td>
<td>Approval date from the company</td>
<td>23/4/2018; 24/4/2021; ...</td>
</tr>
<tr>
<td><b>disbursement_date</b></td>
<td>Disbursement date of loan</td>
<td>24/4/2018; 7/5/2021; ...</td>
</tr>
<tr>
<td><b>due_date</b></td>
<td>Due date for the loan</td>
<td>14/8/2018; 24/10/2021; ...</td>
</tr>
<tr>
<td><b>default_status</b></td>
<td>Default status (Target) <br> (TRUE or FALSE)</td>
<td>TRUE; FALSE; ...</td>
</tr>
</tbody>
</table>

---



# 2. Importing Libraries 📚
👉 **Importing libraries** that will be used in this notebook.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 3. Reading Data Set 👓
👉 After importing libraries, we will also **import the dataset** that will be used.

df_loan = pd.read_csv("loan.csv")

👉 Read the first 6 rows in the dataset.

df_loan.head()

👉 Data type and checking null in dataset.

print(df_loan.info())

👉 From the results above, **there are no missing/null value** in this dataset

# 4. Initial Dataset Exploration 🔍
👉 This section will explore raw dataset that has been imported.

## 4.1 Categorical Variables 📊

### 4.1.1 Loan Default

df_loan.default_status.value_counts()

👉 It can be seen that from results above, False status has more amount than True status of loan default.

👉 To simplify the analysis, a new column called Status was created in which the value will be Default if the value default_status contains True, it will be Non Default.

def change_status(x):
    if x:  # Check if x is True (default)
        return 'Default'
    else:
        return 'Non Default'

df_loan['Status'] = df_loan['default_status'].apply(change_status)
df_loan.head(7)

👉 Define a function encoding_status which converts Non Default values in Status to 0 and Default values to 1.

def encoding_status (Status):
    if Status == 'Non Default':
        return 0
    else:
        return 1


df_loan['Status_coded'] = df_loan['Status'].apply(encoding_status)
df_loan.head()

### 4.1.2 Loan Type

df_loan.loan_type.value_counts()

👉 Define a function encoding_status which converts Car Loan values in loan_type to 0, Education Loan values to 1, Home Loan values to 2 and Personal Loan values to 3.

def encoding_loan_type(loan_type):
    if loan_type == 'Car Loan':
        return 0
    elif loan_type == 'Education Loan':
        return 1
    elif loan_type == 'Home Loan':
        return 2
    elif loan_type == 'Personal Loan':
        return 3
    else:
        return -1  # Return -1 for unknown values or handle them as needed

df_loan['loan_type_coded'] = df_loan['loan_type'].apply(encoding_loan_type)
df_loan.head()

👉 The distribution of types of loan is balanced.

### 4.1.3 Employment Type

df_loan.employment_type.value_counts()

👉 The distribution of employment type is balanced.

👉 Define a function encoding_status which converts Full-time values in employment_type to 0, Part-time values to 1 and Self-employed values to 2.

def encoding_employment_type(employment_type):
    if employment_type == 'Full-time':
        return 0
    elif employment_type == 'Part-time':
        return 1
    elif employment_type == 'Self-employed':
        return 2
    else:
        return -1  # Return -1 for unknown values or handle them as needed

df_loan['employment_type_coded'] = df_loan['employment_type'].apply(encoding_employment_type)
df_loan.head()

### 4.1.4 Income Level

df_loan.income_level.value_counts()

👉 The distribution of income level is balanced.

👉 Define a function encoding_status which converts Low values in income_level to 0, Medium values to 1 and High values to 2.

def encoding_income_level(income_level):
    if income_level == 'Low':
        return 0
    elif income_level == 'Medium':
        return 1
    elif income_level == 'High':
        return 2
    else:
        return -1  # Return -1 for unknown values or handle them as needed

df_loan['income_level_coded'] = df_loan['income_level'].apply(encoding_income_level)
df_loan.head()

### 4.1.5 Gender

df_loan.gender.value_counts()

👉 The distribution of gender is balanced.

👉 Define a function encoding_status which converts Female values in gender to 0 and Male values to 1.

def encoding_gender (gender):
    if gender == 'Female':
        return 0
    else:
        return 1


df_loan['gender_coded'] = df_loan['gender'].apply(encoding_gender)
df_loan.head()

### 4.1.6 Marital Status

df_loan.marital_status.value_counts()

👉 The distribution of marital status is balanced.

👉 Define a function encoding_status which converts Singale values in marital_status to 0, Married values to 1 and Divorced values to 2.

def encoding_marital_status(marital_status):
    if marital_status == 'Single':
        return 0
    elif marital_status == 'Married':
        return 1
    elif marital_status == 'Divorced':
        return 2
    else:
        return -1  # Return -1 for unknown values or handle them as needed

df_loan['marital_status_coded'] = df_loan['marital_status'].apply(encoding_marital_status)
df_loan.head()

### 4.1.7 Education Level

df_loan.education_level.value_counts()

👉 The distribution of education level is balanced.

👉 Define a function encoding_status which converts High School values in education_level to 0, Bachelor values to 1, Master values to 2 and PhD values to 3.

def encoding_education_level(education_level):
    if education_level == 'High School':
        return 0
    elif education_level == 'Bachelor':
        return 1
    elif education_level == 'Master':
        return 2
    elif education_level == 'PhD':
        return 3
    else:
        return -1  # Return -1 for unknown values or handle them as needed

df_loan['education_level_coded'] = df_loan['education_level'].apply(encoding_education_level)
df_loan.head()

## 4.2 Numerical Variables 🔢
👉 This section will show mean, count, std, min, max and others using describe function. The skewness value for each numerical variables will also shown in this section.

df_loan.describe()

skewLoan = df_loan.loan_amount.skew(axis = 0, skipna = True)
print('Loan Amount skewness: ', skewLoan)

skewInterest = df_loan.interest_rate.skew(axis = 0, skipna = True)
print('Interest Rate skewness: ', skewInterest)

skewTerm = df_loan.loan_term.skew(axis = 0, skipna = True)
print('Loan Term skewness: ', skewTerm)

skewCredit = df_loan.credit_score.skew(axis = 0, skipna = True)
print('Credit Score skewness: ', skewCredit)

sns.distplot(df_loan['loan_amount']);

sns.distplot(df_loan['interest_rate']);

sns.distplot(df_loan['loan_term']);

sns.distplot(df_loan['credit_score']);

👉The distribution of **'Loan Amount', 'Interest Rate', 'Loan Term', 'Credit Score'** columns are **symetric**, since the skewness value  between -0.5 and 0.5.

# 5. EDA 📊
👉 This section will explore variables in the dataset using different various plots/charts.

## 5.1 Loan Default Distribution 💰

sns.set_theme(style="darkgrid")
sns.countplot(y="Status", data=df_loan, palette="flare")
plt.ylabel('Loan Default')
plt.xlabel('Total')
plt.show()

## 5.2 Loan Type Distribution 💵

sns.set_theme(style="darkgrid")
sns.countplot(y="loan_type", data=df_loan, palette="flare")
plt.ylabel('Loan Type')
plt.xlabel('Total')
plt.show()

## 5.3 Gender Distribution 👫

sns.set_theme(style="darkgrid")
sns.countplot(x="gender", data=df_loan, palette="rocket")
plt.xlabel('Gender (F=Female, M=Male)')
plt.ylabel('Total')
plt.show()

## 5.4 Marital Status Distribution 🤵🏼‍♂️👰🏼‍♀️

sns.set_theme(style="darkgrid")
sns.countplot(y="marital_status", data=df_loan, palette="crest")
plt.ylabel('Marital Status')
plt.xlabel('Total')
plt.show()

## 5.5 Education Level Distribution🧑🏼‍🎓

# Define the order of education levels
education_level_order = ['High School', 'Bachelor', 'Master', 'PhD']

sns.set_theme(style="darkgrid")
sns.countplot(x="education_level", data=df_loan, palette="magma", order=education_level_order)
plt.xlabel('Education Level')
plt.ylabel('Total')
plt.show()

## 5.6 Employment Type Distribution👩🏼‍💻

# Define the order of employment type
employment_type_order = ['Full-time', 'Part-time', 'Self-employed']
sns.set_theme(style="darkgrid")
sns.countplot(x="employment_type", data=df_loan, palette="magma", order=employment_type_order)
plt.xlabel('Employment Type')
plt.ylabel('Total')
plt.show()

## 5.7 Income Level Distribution📊

# Define the order of income levels
income_level_order = ['Low', 'Medium', 'High']

# Create the countplot with the specified order
sns.set_theme(style="darkgrid")
sns.countplot(x="income_level", data=df_loan, palette="magma", order=income_level_order)

plt.xlabel('Income Level')
plt.ylabel('Total')
plt.show()

## 5.8 Gender Distribution based on Loan Type👫💵

pd.crosstab(df_loan.gender,df_loan.loan_type).plot(kind="bar",figsize=(12,5),color=['#003f5c','#ffa600','#58508d','#bc5090','#ff6361'])
plt.title('Gender distribution based on Loan type')
plt.xlabel('Gender')
plt.xticks(rotation=0)
plt.ylabel('Frequency')
plt.show()

## 5.9 Income Level Distribution based on Education Level 📊🧑🏼‍🎓

# Define the order of income levels and education levels
income_level_order = ['Low', 'Medium', 'High']
education_level_order = ['High School', 'Bachelor', 'Master', 'PhD']

# Filter the DataFrame for only the relevant columns
filtered_df = df_loan[['income_level', 'education_level']]

# Create a crosstab to count the frequencies
crosstab_df = pd.crosstab(filtered_df['income_level'], filtered_df['education_level'])

# Plot the grouped bar chart with specified orders
crosstab_df.loc[income_level_order, education_level_order].plot(kind='bar', figsize=(15, 6), 
                                                              color=['#6929c4', '#1192e8', '#58508d', '#bc5090'])
plt.title('Income Level distribution based on Education Level')
plt.xlabel('Income Level')
plt.xticks(rotation=0)
plt.ylabel('Frequency')
plt.show()

## 5.10 Loan Amount Distribution based on Gender and Interest Rate 🧪👫👴

plt.scatter(x=df_loan.interest_rate[df_loan.gender=='Female'], y=df_loan.loan_amount[(df_loan.gender=='Female')], c="Blue")
plt.scatter(x=df_loan.interest_rate[df_loan.gender=='Male'], y=df_loan.loan_amount[(df_loan.gender=='Male')], c="Orange")
plt.legend(["Female", "Male"])
plt.xlabel("Interest Rate")
plt.ylabel("Loan Amount")
plt.show()

## 5.11 Pair plot and Correlation Heatmap 

# Select the numeric features you want to analyze
numeric_features = ["loan_amount", "interest_rate", "loan_term", "credit_score"]

# Create pair plots for selected numeric features
sns.pairplot(df_loan[numeric_features], diag_kind='kde')
plt.show()

# Create a correlation heatmap for all numeric features
correlation_matrix = df_loan[numeric_features].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# 6. Dataset Preparation ⚙
👉 This section will prepare the dataset before building the machine learning models.

## 6.1 Data Binning 🚮

### 6.1.1 Loan Amount 💵
👉 The age will be divided into **10 loan amount categories**:
*  Below 10,000
*  10,000 - 19,999
*  20,000 - 29,999
*  30,000 - 39,999
*  40,000 - 49,999
*  50,000 - 59,999
*  60,000 - 69,999
*  70,000 - 79,999
*  80,000 - 89,999
*  Above 90,000


bin_loan_amount = [0, 9999, 19999, 29999, 39999, 49999, 59999, 69999, 79999, 89999, 100000]
category_loan_amount = ['<10k', '10k-20k', '20k-30k', '30k-40k', '40k-50k', '50k-60k', '60k-70k','70k-80k','80k-90k','>90k']
df_loan['loan_amount_binned'] = pd.cut(df_loan['loan_amount'], bins=bin_loan_amount, labels=category_loan_amount)


### 6.1.2 Interest Rate
👉 The interest rate will be divided into **8 categories**:
*  Below 5%.
*  5% - 6%.
*  6% - 7%.
*  7% - 8%.
*  8% - 9%.
*  9% - 10%
*  10% - 11%
*  Above 11%.

# Define the bin edges and category labels
bin_interest_rate = [0, 0.049, 0.059, 0.069, 0.079, 0.089, 0.099, 0.109, 0.14]
category_interest_rate = ['<5%', '5%-6%', '6%-7%', '7%-8%', '8%-9%', '9%-10%', '10%-11%', '>=11%']

# Create the 'interest_rate_binned' column
df_loan['interest_rate_binned'] = pd.cut(df_loan['interest_rate'], bins=bin_interest_rate, labels=category_interest_rate)




### 6.1.3 Loan Term
👉 The loan term will be divided into **5 categories**:
*  Below 12.
*  12 - 24.
*  24 - 36.
*  36 - 48.
*  48 - 60.

# Define the bin edges and category labels
bin_loan_term = [0, 11, 23, 35, 47, 59]
category_loan_term = ['<12', '12-24', '24-36', '36-48', '48-60']

# Create the 'loan_term_binned' column
df_loan['loan_term_binned'] = pd.cut(df_loan['loan_term'], bins=bin_loan_term, labels=category_loan_term)




### 6.1.4 Credit Score
👉 The credit score will be divided into **5 categories**:
*  Below 400.
*  401 - 500.
*  501 - 600.
*  601 - 700.
*  Above 700.

# Define the bin edges and category labels
bin_credit_score = [0, 399, 499, 599, 699, 850]
category_credit_score = ['<400', '401-500', '501-600', '601-700', '>700']

# Create the 'credit_score_binned' column
df_loan['credit_score_binned'] = pd.cut(df_loan['credit_score'], bins=bin_credit_score, labels=category_credit_score)




print(df_loan.info())

We decided that only the below columns are import in determining default status hence, we want to use these only.

1.	`Status_coded`
2.	`loan_amount_binned`
3.	`interest_rate_binned`
4.	`loan_term_binned`
5.	`credit_score_binned`
6.	`loan_type_coded`
7.	`employment_type_coded`
8.	`income_level_coded`
9.	`gender_coded`
10.	`marital_status_coded`
11.	`education_level_coded`


Filter down the previous DataFrame to just these 11 columns. Name this new DataFrame as `df2`.

needed_columns = ['Status_coded','loan_amount_binned','interest_rate_binned',
              'loan_term_binned','credit_score_binned','loan_type_coded', 
               'employment_type_coded','income_level_coded','gender_coded',
              'marital_status_coded','education_level_coded']

df2 = df_loan[needed_columns]

df2.head()

## 6.2 Splitting the dataset 🪓
👉 The dataset will be split into **70% training and 30% testing**.

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Assuming df2 is your DataFrame
df2.to_csv('df2.csv', index=False)

X = df2.drop(["Status_coded"], axis=1)
y = df2["Status_coded"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

print(X_train)

print(y_train)

## 6.3 Feature Engineering 🔧
👉 The FE method that used is **one-hot encoding**, which is **transforming categorical variables into a form that could be provided to ML algorithms to do a better prediction**.

X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

X_train.head()

X_test.head()

## 6.4 Synthetic Minority Oversampling Technique (SMOTE Technique) ⚒

👉 Since the number of 'default_status = False' is more than 'default_status = True', **oversampling is carried out to avoid overfitting**.

pip install --upgrade scikit-learn imbalanced-learn

from imblearn.over_sampling import SMOTE
X_train, y_train = SMOTE().fit_resample(X_train, y_train)

sns.set_theme(style="darkgrid")
sns.countplot(y=y_train, palette="mako_r")
plt.ylabel('Default Status')
plt.xlabel('Total')
plt.show()

👉 As can be seen, the distrubtion of default status are now balanced.

# 7. Models 🛠

## 7.1 Logistic Regression

from sklearn.linear_model import LogisticRegression
import random
LRclassifier = LogisticRegression(solver='liblinear', max_iter=5000)
LRclassifier.fit(X_train, y_train)

y_pred = LRclassifier.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

from sklearn.metrics import accuracy_score
LRAcc = accuracy_score(y_pred,y_test)
print('Logistic Regression accuracy is: {:.2f}%'.format(LRAcc*100))

import os
print(os.getcwd())

# Save the trained model to a file
model_filename = 'C:\\Users\\Admin\\Machine Learning.pkl'  # Replace with the desired file path and name
joblib.dump(LRclassifier, model_filename)

filename = 'C:\\Users\\Admin\\Machine Learning.pkl'
model = joblib.load(filename)
# Get a random sample of 5 rows from the test set and their corresponding predictions
random.seed(42)  # Set a random seed for reproducibility
sample_indices = random.sample(range(len(X_test)), 5)

for i in sample_indices:
    print(f"Sample {i + 1}:")
    print(f"Actual: {y_test.iloc[i]}")
    print(f"Predicted: {y_pred[i]}")
    print()

## 7.2 K Neighbours

print(df2.dtypes)

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Convert your data to NumPy arrays to ensure contiguity
X_train_np = np.array(X_train)
y_train_np = np.array(y_train)
X_test_np = np.array(X_test)

# Create and fit the KNeighborsClassifier
KNclassifier = KNeighborsClassifier(n_neighbors=20)
KNclassifier.fit(X_train_np, y_train_np)

# Make predictions
y_pred = KNclassifier.predict(X_test_np)

# Evaluate the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

KNAcc = accuracy_score(y_test, y_pred)
print('K Neighbours accuracy is: {:.2f}%'.format(KNAcc * 100))

## 7.3 Support Vector Machine (SVM)

from sklearn.svm import SVC
SVCclassifier = SVC(kernel='linear', max_iter=251)
SVCclassifier.fit(X_train, y_train)

y_pred = SVCclassifier.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

from sklearn.metrics import accuracy_score
SVCAcc = accuracy_score(y_pred,y_test)
print('SVC accuracy is: {:.2f}%'.format(SVCAcc*100))

## 7.4 Naive Bayes
### 7.4.1 Categorical NB

from sklearn.naive_bayes import CategoricalNB
NBclassifier1 = CategoricalNB()
NBclassifier1.fit(X_train, y_train)

y_pred = NBclassifier1.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

from sklearn.metrics import accuracy_score
NBAcc1 = accuracy_score(y_pred,y_test)
print('Naive Bayes accuracy is: {:.2f}%'.format(NBAcc1*100))

### 7.4.2 Gaussian NB

from sklearn.naive_bayes import GaussianNB
NBclassifier2 = GaussianNB()
NBclassifier2.fit(X_train, y_train)

y_pred = NBclassifier2.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

from sklearn.metrics import accuracy_score
NBAcc2 = accuracy_score(y_pred,y_test)
print('Gaussian Naive Bayes accuracy is: {:.2f}%'.format(NBAcc2*100))

## 7.5 Decision Tree

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# Create and train the Decision Tree classifier
DTclassifier = DecisionTreeClassifier(max_leaf_nodes=20)
DTclassifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = DTclassifier.predict(X_test)

# Print classification report and confusion matrix
print(classification_report(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
classes = ['Non Default', 'Default']
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)
plt.xlabel('Predicted')
plt.ylabel('Actual')

for i in range(len(classes)):
    for j in range(len(classes)):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black")

plt.show()

# Calculate and print accuracy
DTAcc = accuracy_score(y_test, y_pred)
print('Decision Tree accuracy is: {:.2f}%'.format(DTAcc * 100))

scoreListDT = []
for i in range(2,50):
    DTclassifier = DecisionTreeClassifier(max_leaf_nodes=i)
    DTclassifier.fit(X_train, y_train)
    scoreListDT.append(DTclassifier.score(X_test, y_test))
    
plt.plot(range(2,50), scoreListDT)
plt.xticks(np.arange(2,50,5))
plt.xlabel("Leaf")
plt.ylabel("Score")
plt.show()
DTAccMax = max(scoreListDT)
print("DT Acc Max {:.2f}%".format(DTAccMax*100))

## 7.6 Random Forest

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Create and train the Random Forest classifier
RFclassifier = RandomForestClassifier(max_leaf_nodes=30)
RFclassifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = RFclassifier.predict(X_test)

# Print classification report and confusion matrix
print(classification_report(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
classes = ['Non Default', 'Default']
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)
plt.xlabel('Predicted')
plt.ylabel('Actual')

for i in range(len(classes)):
    for j in range(len(classes)):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black")

plt.show()

# Calculate and print accuracy
RFAcc = accuracy_score(y_test, y_pred)
print('Random Forest accuracy is: {:.2f}%'.format(RFAcc * 100))

scoreListRF = []
for i in range(2,50):
    RFclassifier = RandomForestClassifier(n_estimators = 1000, random_state = 1, max_leaf_nodes=i)
    RFclassifier.fit(X_train, y_train)
    scoreListRF.append(RFclassifier.score(X_test, y_test))
    
plt.plot(range(2,50), scoreListRF)
plt.xticks(np.arange(2,50,5))
plt.xlabel("RF Value")
plt.ylabel("Score")
plt.show()
RFAccMax = max(scoreListRF)
print("RF Acc Max {:.2f}%".format(RFAccMax*100))

# 8. Model Comparison 👀

compare = pd.DataFrame({'Model': ['Logistic Regression', 'K Neighbors', 'SVM', 'Categorical NB', 'Gaussian NB', 'Decision Tree', 'Decision Tree Max', 'Random Forest', 'Random Forest Max'], 
                        'Accuracy': [LRAcc*100, KNAcc*100, SVCAcc*100, NBAcc1*100, NBAcc2*100, DTAcc*100, DTAccMax*100, RFAcc*100, RFAccMax*100]})
compare.sort_values(by='Accuracy', ascending=False)

## 8.1 Model Evaluation
👉 Based on the provided accuracy scores for different machine learning models, here are some comments on their performance:

**Logistic Regression (81.00%) and Decision Tree (81.00%):**

Both Logistic Regression and Decision Tree models perform the best with an accuracy of 81.00%.
These models are showing strong predictive power for the task at hand.

**Random Forest (71.33%) and Random Forest Max (72.93%):**

Random Forest models are slightly outperformed by Logistic Regression and Decision Tree.
Random Forest Max performs better than the standard Random Forest model.
Random Forest models are known for their robustness and are often used for classification tasks.

**Categorical NB (67.40%):**

The Categorical Naive Bayes model performs reasonably well with an accuracy of 67.40%.
Naive Bayes models are simple and work well with categorical data.

**Gaussian NB (56.13%):**

The Gaussian Naive Bayes model has the lowest accuracy among the models evaluated, with 56.13%.
This model might not be well-suited for this dataset, which may contain non-Gaussian distributed features.

**SVM (49.53%):**

The Support Vector Machine (SVM) model has the lowest accuracy of 49.53%.
SVMs can perform well on certain datasets, but they may not be the best choice for this particular problem.

**K Neighbors (34.93%):**

The K Neighbors model has the lowest accuracy among all models, with 34.93%.
This suggests that the choice of k and the data distribution might not be suitable for this model.

#### 👉In summary, **Logistic Regression** and **Decision Tree models** stand out as the best performers in terms of accuracy. However, it's essential to consider other metrics such as precision, recall, and F1-score, as well as potentially fine-tuning hyperparameters and exploring different algorithms to optimize the model further. 

# 9. Actionable Insight

👉 Based on the performance evaluation of the machine learning models, here are some actionable insights:

## 9.1 Choose Logistic Regression or Decision Tree: 
Given their high accuracy of 81.00%, consider using Logistic Regression or Decision Tree as the primary model for predicting loan defaults. These models have shown strong predictive power for this task.

## 9.2 Consider Random Forest with Max Nodes: 
Random Forest models, especially the one with max leaf nodes (72.93%), can be a good alternative if you want to explore ensemble methods. They provide robustness and may offer improved performance with more tuning.

## 9.3 Feature Engineering: 
Evaluate the importance of features in the selected models to identify which factors are most influential in predicting loan defaults. Feature engineering, such as creating new features or transforming existing ones, can potentially enhance model performance.

## 9.4 Model Evaluation Metrics: 
Apart from accuracy, pay attention to other evaluation metrics like precision, recall, and F1-score. These metrics provide a more comprehensive view of model performance, especially in imbalanced datasets where defaults may be relatively rare.

## 9.5 Continuous Monitoring: 
Implement a system for continuous monitoring and updating of the model. As the dataset evolves over time, retraining the model with fresh data can help maintain its accuracy and relevance.

## 9.6 Explore Additional Data: 
Depending on the availability, consider incorporating additional relevant data sources (e.g. time factor, economic indicators, customer behavior) that could further enhance the model's predictive capabilities.

# 10. Future Prediction📤
👉 The next step will save trained model for future use.

# Verify the current working directory
import os
print(os.getcwd())

## 10.1 Logistic Regression Classifier Trained Model 🧹

### 10.1.1 Save the trained model to a file

model_filename = 'C:\\Users\\Admin\\Machine Learning.pkl'  # Replace with the desired file path and name
joblib.dump(LRclassifier, model_filename)

### 10.1.2 Load the saved model for prediction

import joblib
import random

# Load the saved model
filename = 'C:\\Users\\Admin\\Machine Learning.pkl'
model = joblib.load(filename)

# Get a random sample of 10 rows from the test set and their corresponding predictions
random.seed(42)  # Set a random seed for reproducibility
sample_indices = random.sample(range(len(X_train)), 10)

for i in sample_indices:
    print(f"Sample {i + 1}:")
    print(f"Actual: {y_train.iloc[i]}")  # Actual label from the train set
    # Make a prediction using the loaded model
    prediction = model.predict([X_train.iloc[i]])[0]
    print(f"Predicted: {prediction}")  # Model's prediction for the sample
    print()

## 10.2 Decision Tree Classifier Trained Model

### 10.2.1 Save the trained model to a file

model_filename = 'C:\\Users\\Admin\\Machine Learning.pkl'  # Replace with the desired file path and name
joblib.dump(DTclassifier, model_filename)

### 10.2.2 Load the saved DT Classifier model for prediction

import joblib
import random

# Load the saved model
filename = 'C:\\Users\\Admin\\Machine Learning.pkl'
model = joblib.load(filename)

# Get a random sample of 10 rows from the test set and their corresponding predictions
random.seed(50)  # Set a random seed for reproducibility
sample_indices = random.sample(range(len(X_train)), 10)

for i in sample_indices:
    print(f"Sample {i + 1}:")
    print(f"Actual: {y_train.iloc[i]}")  # Actual label from the train set
    # Make a prediction using the loaded model
    prediction = model.predict([X_train.iloc[i]])[0]
    print(f"Predicted: {prediction}")  # Model's prediction for the sample
    print()

# 11. References 🔗
📚 **Kaggle Notebook**:
*  [Drug Classification With Different Algorithms by Görkem Günay](https://www.kaggle.com/gorkemgunay/drug-classification-with-different-algorithms)
*  [Drug Classification - 100% Accuracy by Erin Ward](https://www.kaggle.com/eward96/drug-classification-100-accuracy)
*  [drug prediction with acc(100 %) by Sachin Sharma](https://www.kaggle.com/sachinsharma1123/drug-prediction-with-acc-100)

---
