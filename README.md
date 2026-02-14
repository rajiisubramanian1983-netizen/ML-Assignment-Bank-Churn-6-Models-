**1. Problem Statement**
The objective of this assignment is to predict bank customer churn using machine learning models.
Given historical customer data (demographics, account information, and behavior), the goal is to build models that classify whether a customer will exit the bank (churn) or stay.
This is a binary classification problem in the banking domain, where the target variable indicates whether a customer has exited the bank.
The models are trained on the provided dataset, evaluated using multiple metrics, and finally deployed as an interactive web application using Streamlit Community Cloud.

**2. Dataset Description**
Dataset Source: Kaggle – Bank Customer Churn Dataset
Domain: Banking
Problem Type: Binary Classification

**Dataset Characteristics:**

Number of instances: ~10,000
Number of features: 13+
Target Variable: Exited
1 → Customer churned
0 → Customer did not churn

Features Used:
CreditScore

Geography

Gender

Age

Tenure

Balance

NumOfProducts

HasCrCard

IsActiveMember

EstimatedSalary

The dataset is moderately imbalanced, hence robust metrics such as AUC and Matthews Correlation Coefficient (MCC) are used for evaluation.

Columns dropped before modeling:
RowNumber, CustomerId, Surname (ID-like columns that do not help prediction)

Preprocessing:
Missing values in Exited are removed.
Categorical columns (Geography, Gender) are label‑encoded using LabelEncoder.

**3. Machine Learning Models Used**
The following six classification models were implemented on the same dataset:
Logistic Regression
Decision Tree Classifier
K-Nearest Neighbors (KNN)
Naive Bayes (Gaussian)
Random Forest (Ensemble)
XGBoost (Ensemble)
Each trained model is saved as a .pkl file and loaded in the Streamlit app for evaluation on the test dataset

**4. Evaluation Metrics**
Each model was evaluated using the following metrics:
Accuracy
AUC Score
Precision
Recall
F1 Score
Matthews Correlation Coefficient (MCC)
**5. Model Comparison Table**

|ML Model|Accuracy|AUC|Precision|Recall|F1 Score|MCC|
|-|-|-|-|-|-|-|
|Logistic Regression|0.807|0.760|0.580|0.197|0.294|0.254|
|Decision Tree|0.774|0.651|0.444|0.442|0.443|0.302|
|KNN|0.825|0.710|0.640|0.314|0.422|0.361|
|Naive Bayes|0.831|0.805|0.735|0.265|0.390|0.372|
|Random Forest (Ensemble)|0.848|0.809|0.738|0.388|0.509|0.460|
|XGBoost (Ensemble)|0.856|0.838|0.754|0.430|0.548|0.496|

**6. Model Performance Observations**

|ML Model|Observation|
|-|-|
|Logistic Regression|Achieves moderate accuracy (0.807) and AUC (0.760), showing reasonable ranking ability but not the best.Precision is low‑medium (0.580) and recall is very low (0.197), meaning it predicts few churners and misses many actual churners.F1 (0.294) and MCC (0.254) are low, indicating overall weak performance on the positive (churn) class|
|Decision Tree|Captures non-linear relationships in the data and improves recall compared to Logistic Regression, but may overfit and shows moderate MCC.|
|KNN|Performance is sensitive to feature scaling. It achieves balanced precision and recall but does not significantly outperform tree-based models.|
|Naive Bayes|Assumes feature independence, resulting in lower precision but higher recall. Useful for identifying churners but less accurate overall.|
|Random Forest (Ensemble)|Demonstrates improved generalization and balanced performance across all metrics, with higher AUC and MCC compared to individual models.|
|XGBoost (Ensemble)|Achieves the best overall performance with the highest AUC, F1 score, and MCC, effectively handling class imbalance and complex feature interactions.|



LIVE DEMO 


https://ml-assignment-2-bank-churn-yehaq9itdm2appzgpoo7zcx.streamlit.app/



GitHub 


https://github.com/rajiisubramanian1983-netizen/ML-Assignment-2-Bank-Churn/tree/main

ScreenShot 

<img width="940" height="461" alt="image" src="https://github.com/user-attachments/assets/1964812a-77ec-4514-a9ff-210c2ade4c66" />


