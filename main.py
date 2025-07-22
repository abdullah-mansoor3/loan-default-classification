# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
                             RocCurveDisplay)
from sklearn.tree import plot_tree

# Load model once
@st.cache_resource
def load_model():
    with open("./models/decision-tree-model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# Page navigation
st.sidebar.title("Model Dashboard")
page = st.sidebar.radio("Select Page", ["Model Comparison", "Model Evaluation", "Random Prediction"])

# --- Page 1: Comparison Report ---
if page == "Model Comparison":
    st.title("Model Comparison Report")

    st.markdown("""
    **Random Forest:**
    - Accuracy: ~87%
    - ROC-AUC: ~0.50 (class 1)
    - Precision (class 1): ~0%
    - Recall (class 1): ~0%
    - **Conclusion:** High accuracy overall, but completely fails on the minority class.

    **Decision Tree (Vanilla):**
    - Accuracy: ~65%
    - ROC-AUC: ~0.66
    - Precision (class 1): ~15%
    - Recall (class 1): ~68%
    - **Conclusion:** Decent recall on class 1, but poor precision. Needs better balance.

    **Decision Tree (Hyperparameter Tuned):**
    - Accuracy: ~64%
    - ROC-AUC: ~0.66
    - Precision (class 1): ~14%
    - Recall (class 1): ~67%
    - **Conclusion:** No significant improvement from hyperparameter tuning. Slight drop in precision and accuracy; likely due to overfitting or suboptimal grid.

    **Balanced Random Forest:**
    - Accuracy: ~78%
    - ROC-AUC: ~0.72
    - Precision (class 1): ~31%
    - Recall (class 1): ~67%
    - **Conclusion:** Much better balance between classes. Good overall performance.

    **Logistic Regression (with SMOTE):**
    - Accuracy: ~71%
    - ROC-AUC: ~0.67
    - Precision (class 1): ~22%
    - Recall (class 1): ~62%
    - **Conclusion:** Handles class imbalance better than RF/DT, but lower overall accuracy.

    **XGBoost:**
    - Accuracy: ~83%
    - ROC-AUC: ~0.76
    - Precision (class 1): ~40%
    - Recall (class 1): ~61%
    - **Conclusion:** Strongest performance across both classes. Best generalization.
    """)


# --- Page 2: Evaluation Metrics + Tree Visualization ---
elif page == "Model Evaluation":
    st.title("Decision Tree Evaluation")

    # Load full training set to evaluate
    df = pd.read_csv("./data/test-data.csv")
    y_true = df['TARGET']
    X = df.drop(columns=['TARGET'])
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    st.subheader("Classification Metrics")
    st.write(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    st.write(f"Precision: {precision_score(y_true, y_pred):.4f}")
    st.write(f"Recall: {recall_score(y_true, y_pred):.4f}")
    st.write(f"F1 Score: {f1_score(y_true, y_pred):.4f}")
    st.write(f"ROC-AUC: {roc_auc_score(y_true, y_prob):.4f}")

    st.subheader("Confusion Matrix")
    fig1, ax1 = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax1)
    st.pyplot(fig1)

    st.subheader("ROC Curve")
    fig2, ax2 = plt.subplots()
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax2)
    st.pyplot(fig2)

    st.subheader("Tree Visualization")
    fig3, ax3 = plt.subplots(figsize=(15, 8))
    plot_tree(model, filled=True, fontsize=6, max_depth=3, feature_names=[f"Feature {i}" for i in range(model.n_features_in_)])
    st.pyplot(fig3)

# --- Page 3: Load CSV and Predict ---
elif page == "Random Prediction":
    st.title("Predict on Test Data")
    df = pd.read_csv('./data/test-data.csv')
    if 'TARGET' in df.columns:
        df.drop(columns=['TARGET'], inplace = True)
    st.write("Data Preview:")
    st.dataframe(df.head())

    if st.button("Predict Random Sample"):
        sample = df.sample(1, random_state=np.random.randint(1000))
        pred = model.predict(sample)[0]
        proba = model.predict_proba(sample)[0]
        st.write("sample:", sample)
        st.write("Prediction:", pred)
        st.write("Probabilities:", {f"Class {i}": round(p, 3) for i, p in enumerate(proba)})
