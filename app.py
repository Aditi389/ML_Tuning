import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(page_title="ML Model Evaluation", layout="wide")
st.title("ML Model Evaluation and Tuning (Celebal Assignment)")

# Let the user upload a CSV file
uploaded_file = st.file_uploader("Upload your health_monitoring.csv file", type=["csv"])

if uploaded_file:
    # Read CSV into a dataframe
    data = pd.read_csv(uploaded_file)
    st.subheader("Preview of Raw Data")
    st.write(data.head())  # Just showing the top few rows

    # Dropping columns that don't really help with modeling
    columns_to_drop = [
        'Device-ID/User-ID', 'Timestamp',
        'Heart Rate Below/Above Threshold (Yes/No)', 'Blood Pressure',
        'Blood Pressure Below/Above Threshold (Yes/No)',
        'Glucose Levels Below/Above Threshold (Yes/No)',
        'SpO\u2082 Below Threshold (Yes/No)', 'Caregiver Notified (Yes/No)'
    ]

    # Not all of these columns might be present so using errors='ignore'
    data.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    # Convert the target column to binary values
    data['Alert Triggered (Yes/No)'] = data['Alert Triggered (Yes/No)'].map({'Yes': 1, 'No': 0})

    # Separate the features and the label
    X = data.drop(columns=['Alert Triggered (Yes/No)'])
    y = data['Alert Triggered (Yes/No)']

    # Standard train-test split, nothing fancy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features (I almost forgot this step last time!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Models to try out
    model_options = {
        "Logistic Regression": LogisticRegression(max_iter=200),
        "Support Vector Machine": SVC(),
        "Random Forest": RandomForestClassifier()
    }

    st.subheader("Model Evaluation Results")

    eval_metrics = {}  # Collect metrics here
    cv_scores_dict = {}  # And cross-validation scores

    for model_name, model_instance in model_options.items():
        # Fit model
        model_instance.fit(X_train_scaled, y_train)

        # Predict and calculate test metrics
        predictions = model_instance.predict(X_test_scaled)

        # These could be looped through but being explicit is clearer here
        eval_metrics[model_name] = {
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions),
            "Recall": recall_score(y_test, predictions),
            "F1 Score": f1_score(y_test, predictions)
        }

        # Cross-validation just to see how it holds up on different splits
        cross_val = cross_val_score(model_instance, X_train_scaled, y_train, cv=5, scoring='accuracy')
        cv_scores_dict[model_name] = f"{cross_val.mean():.4f} ± {cross_val.std():.4f}"

    st.write("Test Set Performance Metrics")
    results_df = pd.DataFrame(eval_metrics).T
    st.dataframe(results_df.style.format("{:.4f}"))

    st.write("5-Fold Cross-Validation Accuracy")
    cv_df = pd.DataFrame.from_dict(cv_scores_dict, orient='index', columns=["CV Accuracy"])
    st.dataframe(cv_df)

else:
    st.warning("Please upload the health_monitoring.csv file to proceed.")
