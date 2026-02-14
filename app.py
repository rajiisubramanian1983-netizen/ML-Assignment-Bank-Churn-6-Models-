import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

st.title("Bank Customer Churn Prediction – 6 ML Models")

st.write(
    "1) Download the test data CSV. "
    "2) Upload this test CSV. "
    "3) Select a model to see metrics and confusion matrix."
)

# -----------------------
# Step 1: Download test data
# -----------------------
try:
    test_df = pd.read_csv("model/test_data.csv")
    csv_bytes = test_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download test data CSV",
        data=csv_bytes,
        file_name="test_data.csv",
        mime="text/csv",
    )
except Exception:
    st.error("model/test_data.csv not found. Run the training notebook first.")
    st.stop()

st.markdown("---")

# -----------------------
# Step 2: Upload test data
# -----------------------
uploaded_file = st.file_uploader(
    "Upload the test CSV (same format as downloaded)", type=["csv"]
)

# -----------------------
# Step 3: Select model
# -----------------------
model_name = st.selectbox(
    "Select a model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost",
    ],
)

# -----------------------
# Step 4: Evaluate
# -----------------------
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    if "Exited" not in data.columns:
        st.error("Uploaded CSV must contain the 'Exited' column.")
    else:
        # Separate features and target
        X_test = data.drop("Exited", axis=1)
        y_test = data["Exited"]

        # Load scaler and encoders saved from training notebook
        try:
            scaler = joblib.load("model/scaler.pkl")
            le_geo = joblib.load("model/label_encoder_geo.pkl")
            le_gender = joblib.load("model/label_encoder_gender.pkl")
        except Exception:
            st.error("Scaler / encoders not found in model/ folder.")
            st.stop()

        # Columns used during training (same order as notebook)
        feature_columns = [
            "CreditScore",
            "Geography",
            "Gender",
            "Age",
            "Tenure",
            "Balance",
            "NumOfProducts",
            "HasCrCard",
            "IsActiveMember",
            "EstimatedSalary",
        ]

        # Apply same encoding as training, but handle unseen categories
        if "Geography" in X_test.columns:
            known_geo = list(le_geo.classes_)
            default_geo = known_geo[0]
            X_test["Geography"] = X_test["Geography"].where(
                X_test["Geography"].isin(known_geo),
                other=default_geo,
            )
            X_test["Geography"] = le_geo.transform(X_test["Geography"])

        if "Gender" in X_test.columns:
            known_gender = list(le_gender.classes_)
            default_gender = known_gender[0]
            X_test["Gender"] = X_test["Gender"].where(
                X_test["Gender"].isin(known_gender),
                other=default_gender,
            )
            X_test["Gender"] = le_gender.transform(X_test["Gender"])

        # Keep only the training features, in the same order
        try:
            X_test = X_test[feature_columns]
        except KeyError:
            st.error(
                "Uploaded CSV does not have the exact feature columns used in training.\n"
                "Expected columns: "
                + ", ".join(feature_columns)
            )
            st.stop()

        # Scaled version
        X_test_scaled = scaler.transform(X_test)

        # Map model name to file and whether to use scaled features
        model_map = {
            "Logistic Regression": ("model/logistic_regression.pkl", True),
            "Decision Tree": ("model/decision_tree.pkl", False),
            "KNN": ("model/knn.pkl", True),
            "Naive Bayes": ("model/naive_bayes.pkl", True),
            "Random Forest": ("model/random_forest.pkl", False),
            "XGBoost": ("model/xgboost.pkl", False),
        }

        model_path, uses_scaled = model_map[model_name]

        try:
            model = joblib.load(model_path)
        except Exception:
            st.error(f"{model_name} file not found at {model_path}.")
            st.stop()

        X_input = X_test_scaled if uses_scaled else X_test

        # Predictions
        y_pred = model.predict(X_input)

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_input)[:, 1]
        else:
            st.warning("Model has no predict_proba; AUC may not be meaningful.")
            y_proba = np.zeros_like(y_pred, dtype=float)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)

        st.subheader("Evaluation Metrics")
        st.write(f"**Model**: {model_name}")
        st.write(f"Accuracy: {acc:.3f}")
        st.write(f"AUC: {auc:.3f}")
        st.write(f"Precision: {prec:.3f}")
        st.write(f"Recall: {rec:.3f}")
        st.write(f"F1 Score: {f1:.3f}")
        st.write(f"MCC: {mcc:.3f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(
            cm,
            index=["Actual 0", "Actual 1"],
            columns=["Predicted 0", "Predicted 1"],
        )

        st.subheader("Confusion Matrix")
        st.dataframe(cm_df)

        # Classification report
        st.subheader("Classification Report")
        report_str = classification_report(y_test, y_pred, zero_division=0)
        st.text(report_str)
else:
    st.info("Upload the test CSV to evaluate a model.")
