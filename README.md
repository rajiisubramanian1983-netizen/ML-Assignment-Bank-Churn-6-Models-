Name :RAJALAKSHMI S
BITS ID :2025AA05262
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
|Decision Tree|Accuracy (0.774) and AUC (0.651) are lower than Logistic Regression and the ensemble models.Precision (0.444) and recall (0.442) are more balanced, so it catches more churners but with more false positives compared to LR.F1 (0.443) and MCC (0.302) so it is better than Logistic Regression at identifying churners, but overall still weaker than KNN, Naive Bayes, Random Forest, and XGBoost.|
|KNN|Accuracy (0.825) is better than LR and Decision Tree, but AUC (0.710) is still below the best models.​Precision is good (0.640), but recall is only 0.314, so it identifies churners more confidently but still misses many.F1 (0.422) and MCC (0.361) show a noticeable improvement over LR and Decision Tree, but not as strong as Naive Bayes or the ensembles..|
|Naive Bayes|Accuracy (0.831) and especially AUC (0.805) are strong, indicating good ranking between churn and non‑churn customers.​Precision is high (0.735), but recall is low (0.265), meaning it predicts churners with high confidence but finds only a small fraction of them.F1 (0.390) and MCC (0.372) are slightly better than kNN, suggesting Naive Bayes is more reliable overall despite low recall..|
|Random Forest (Ensemble)|Shows high accuracy (0.848) and AUC (0.809), clearly better than most non‑ensemble models.Precision is high (0.738) and recall (0.388) improves compared to LR, kNN, and Naive Bayes, so it balances catching more churners without too many false alarms.F1 (0.509) and MCC (0.460) are strong, indicating good overall classification performance and better handling of both classes.|
|XGBoost (Ensemble)|Delivers the best overall performance: highest accuracy (0.856) and AUC (0.838), showing excellent separation between churn and non‑churn customers.Precision (0.754) and recall (0.430) are both higher than for other models, so it captures more churners while keeping predictions precise.F1 (0.548) and MCC (0.496) are the highest, making XGBoost the most effective and balanced model for this dataset among all six|

Deployment on Streamlit Community Cloud

The trained models (.pkl files for all 6 models, scaler, and label encoders) are stored in the model/ folder.

The app is implemented in app.py using Streamlit, and loads the .pkl files using joblib.load.

The project is pushed to a GitHub repository, and the app is deployed on Streamlit Community Cloud by linking the GitHub repo and setting app.py as the main file.

The deployed app allows the user to:

Download a sample test CSV.

Upload a test CSV with the same schema.

Select any of the 6 models.

View all evaluation metrics, confusion matrix, and classification report directly in the browser.

LIVE DEMO 

https://vztbalhvmvynayrmwpkarm.streamlit.app/

GitHub 


https://github.com/rajiisubramanian1983-netizen/ML-Assignment-Bank-Churn-6-Models-

ScreenShot 

<img width="940" height="461" alt="image" src="https://github.com/user-attachments/assets/1964812a-77ec-4514-a9ff-210c2ade4c66" />


