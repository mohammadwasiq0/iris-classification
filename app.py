import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def load_data():
    data = load_iris()
    df = pd.DataFrame(data=np.c_[data.data, data.target], columns=data.feature_names + ['target'])
    return df

def preprocess_data(data):
    X = data.drop("target", axis=1)
    y = data["target"]
    return X, y

def get_model_names():
    return [
        "Logistic Regression", "Support Vector Machine (SVM)", 
        "Decision Tree Classifier", "Random Forest Classifier", 
        "Gradient Boosting Classifier", "AdaBoost Classifier", 
        "XGBoost Classifier", "K-Nearest Neighbors Classifier (KNN)", 
        "Neural Network Classifier"
    ]

def get_model(model_name):
    if model_name == "Logistic Regression":
        return LogisticRegression()
    elif model_name == "Support Vector Machine (SVM)":
        return SVC()
    elif model_name == "Decision Tree Classifier":
        return DecisionTreeClassifier()
    elif model_name == "Random Forest Classifier":
        return RandomForestClassifier()
    elif model_name == "Gradient Boosting Classifier":
        return GradientBoostingClassifier()
    elif model_name == "AdaBoost Classifier":
        return AdaBoostClassifier()
    elif model_name == "XGBoost Classifier":
        return XGBClassifier()
    elif model_name == "K-Nearest Neighbors Classifier (KNN)":
        return KNeighborsClassifier()
    elif model_name == "Neural Network Classifier":
        return MLPClassifier()
    else:
        return None

def get_hyperparameters(model_name):
    # Add hyperparameter options for each classifier as needed
    if model_name == "Logistic Regression":
        return {"C": [0.1, 1.0, 10.0]}
    elif model_name == "Support Vector Machine (SVM)":
        return {"C": [0.1, 1.0, 10.0], "kernel": ["linear", "rbf"]}
    elif model_name == "Decision Tree Classifier":
        return {"max_depth": [None, 5, 10]}
    elif model_name == "Random Forest Classifier":
        return {"n_estimators": [50, 100, 200]}
    elif model_name == "Gradient Boosting Classifier":
        return {"n_estimators": [50, 100, 200], "learning_rate": [0.1, 0.01]}
    elif model_name == "AdaBoost Classifier":
        return {"n_estimators": [50, 100, 200], "learning_rate": [0.1, 0.01]}
    elif model_name == "XGBoost Classifier":
        return {"n_estimators": [50, 100, 200], "learning_rate": [0.1, 0.01]}
    elif model_name == "K-Nearest Neighbors Classifier (KNN)":
        return {"n_neighbors": [3, 5, 7]}
    elif model_name == "Neural Network Classifier":
        return {"hidden_layer_sizes": [(10,), (20,), (30,)]}
    else:
        return {}

def display_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    st.write("Accuracy:", acc)
    st.write("Precision:", precision)
    st.write("Recall:", recall)
    st.write("F1 Score:", f1)

def plot_confusion_matrix(y_true, y_pred, target_names):
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=target_names, columns=target_names)
    plt.figure(figsize=(6, 4))
    plt.title("Confusion Matrix")
    sn.heatmap(df_cm, annot=True, cmap="Blues", fmt="d")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    st.pyplot()

def main():
    st.title("Classification Algorithms with Hyperparameter Tuning")
    
    # Load the Iris dataset
    data = load_data()
    
    # Sidebar options
    model_names = get_model_names()
    selected_model = st.sidebar.selectbox("Select Classification Model", model_names)
    
    if selected_model:
        st.subheader(f"Selected Model: {selected_model}")
        
        # Split the dataset into features and target variable
        X, y = preprocess_data(data)
        
        # Hyperparameter tuning
        params = get_hyperparameters(selected_model)
        model = GridSearchCV(get_model(selected_model), params, scoring='accuracy', cv=5)
        model.fit(X, y)
        
        # Display hyperparameters and best score
        st.write("Best Hyperparameters:")
        st.write(model.best_params_)
        st.write("Best Accuracy Score:", model.best_score_)
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Display evaluation metrics
        display_metrics(y, y_pred)
        
        # Plot confusion matrix
        target_names = data.target_names
        plot_confusion_matrix(y, y_pred, target_names)
