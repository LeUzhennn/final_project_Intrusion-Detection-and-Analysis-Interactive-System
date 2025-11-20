"""此模組負責模型訓練、評估與預測。"""
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

@st.cache_data(show_spinner=False)
def train_and_evaluate(_X_train, _X_test, _y_train, _y_test, class_names):
    """
    訓練隨機森林模型並評估其效能。
    """
    with st.spinner("正在訓練隨機森林模型..."):
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(_X_train, _y_train)

    with st.spinner("正在評估模型效能..."):
        y_pred = model.predict(_X_test)

        metrics = {
            "accuracy": accuracy_score(_y_test, y_pred),
            "precision": precision_score(_y_test, y_pred, average='weighted'),
            "recall": recall_score(_y_test, y_pred, average='weighted'),
            "f1_score": f1_score(_y_test, y_pred, average='weighted')
        }

        # 計算混淆矩陣
        cm = confusion_matrix(_y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    return metrics, model, cm_df