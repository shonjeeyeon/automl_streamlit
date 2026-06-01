"""
AutoML Insight Studio
A human-centered Streamlit AutoML app for CSV dataset classification.

How to run:
1. Save this file as app.py
2. Install dependencies:
   pip install streamlit pandas numpy scikit-learn matplotlib seaborn
3. Run:
   1) streamlit run app.py, OR
   2) visit: https://automlapp-mqysgjr8wnc6cvk35tuyww.streamlit.app/
"""

# Import Libraries
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier


# Page setup: Will use wide to display the tables
st.set_page_config(
    page_title="AutoML Insight Studio",
    layout="wide"
)


# Custom styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #1f4e79;
        margin-bottom: 0px;
    }
    .subtitle {
        font-size: 18px;
        color: #555;
        margin-bottom: 24px;
    }
    .section-card {
        background-color: #f8f9fb;
        padding: 22px;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        margin-bottom: 18px;
    }
    .success-box {
        background-color: #e8f5e9;
        color: #1b5e20;
        padding: 14px;
        border-radius: 10px;
        border-left: 6px solid #2e7d32;
    }
    .warning-box {
        background-color: #fff8e1;
        color: #795548;
        padding: 14px;
        border-radius: 10px;
        border-left: 6px solid #ffb300;
    }
    .metric-explain {
        font-size: 14px;
        color: #666;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ==================
# Helper functions
# ==================

def make_sample_dataset():
    """Create a small sample probability based classification dataset. That will help users without a CSV file understand how to run the app."""
    rng = np.random.default_rng(42)
    n = 250
    age = rng.integers(18, 70, n)
    income = rng.normal(60000, 15000, n).round(2)
    website_visits = rng.integers(1, 30, n)
    membership = rng.choice(["Basic", "Silver", "Gold"], n)

    purchased_probability = (
        0.15
        + 0.25 * (income > 65000)
        + 0.20 * (website_visits > 12)
        + 0.20 * (membership == "Gold")
    )
    purchased = rng.binomial(1, np.clip(purchased_probability, 0, 0.95))

    return pd.DataFrame({
        "Age": age,
        "Income": income,
        "Website_Visits": website_visits,
        "Membership": membership,
        "Purchased": purchased
    })


def get_problem_warning(target_series):
    """Return a user-friendly warning if the target column is not ideal: Too few or too many classes"""
    unique_count = target_series.nunique(dropna=True)

    if unique_count < 2:
        return "The selected target has fewer than two classes. Please choose another target column."
    if unique_count > 20:
        return (
            "The selected target has many unique values. This app is designed for classification, "
            "so choose a categorical target with a smaller number of classes."
        )
    return None


def build_preprocessor(X):
    """Create preprocessing steps for numeric and categorical columns."""
    numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["int64", "float64", "int32", "float32"]).columns.tolist()

   # Pipelines for impute, scaling, and encoding
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor, numeric_features, categorical_features


def train_models(X_train, X_test, y_train, y_test):
    """Train three models and return results."""
    preprocessor, numeric_features, categorical_features = build_preprocessor(X_train)

   # Will use Logreg, RF, and NN (MLPClassifier)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=600, random_state=42)
    }

    results = []
    trained_pipelines = {}

    progress = st.progress(0)
    status = st.empty()

    for i, (model_name, model) in enumerate(models.items(), start=1):
        status.info(f"Training {model_name}...")

        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        results.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "Predictions": predictions
        })
        trained_pipelines[model_name] = pipeline
        progress.progress(i / len(models))

    status.success("All models finished running.")
    return results, trained_pipelines, numeric_features, categorical_features


def plot_accuracy(results_df):
    """Plot model accuracy comparison."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(results_df["Model"], results_df["Accuracy"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy Comparison")
    ax.tick_params(axis="x", rotation=20)

    for idx, value in enumerate(results_df["Accuracy"]):
        ax.text(idx, value + 0.02, f"{value:.2%}", ha="center", fontsize=10)

    st.pyplot(fig)


def plot_confusion_matrix(cm, labels, title):
    """Plot a confusion matrix using matplotlib."""
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    fig.colorbar(image)
    st.pyplot(fig)


def explain_accuracy(score):
   """Explain what the score means"""
    if score >= 0.90:
        return "Excellent performance. The model is making very accurate predictions on the test data."
    if score >= 0.80:
        return "Strong performance. The model is likely useful, but it may still make some mistakes."
    if score >= 0.70:
        return "Moderate performance. The model finds patterns, but improvement may be needed."
    return "Low performance. The dataset may need cleaning, better features, or a different modeling approach."


# ==================
# Visualization
# ==================

# App header
st.markdown('<p class="main-title">AutoML Insight Studio</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Upload a CSV dataset, choose a target column, and compare three machine learning models automatically.</p>',
    unsafe_allow_html=True
)


# Sidebar guidance: display steps to take
with st.sidebar:
    st.header("How to Use This App")
    st.write("Step 1. Upload a CSV file or use the sample dataset.")
    st.write("Step 2. Preview the data.")
    st.write("Step 3. Select a target column.")   
    st.write("Step 4. Run the models with one click.")
    st.write("Step 5. Review accuracy, charts, and recommendations.")

    st.divider()
    st.subheader("Human-Centered Design")
    st.write("This app uses simple language, guided steps, visual feedback, and automatic preprocessing to reduce cognitive load.")


# Step 1: Dataset input
st.header("Step 1: Upload Your Dataset")
st.write("Use a CSV file with rows as observations and columns as features. One column should be the value you want to predict.")

col_upload, col_sample = st.columns([2, 1])

# Form for drag and drop CSV upload
with col_upload:
    uploaded_file = st.file_uploader("Drag and drop your CSV file here", type=["csv"])

# Choose to use sample dataset
with col_sample:
    use_sample = st.button("Use Sample Dataset")

# Success message if CSV is valid; error message for invalid files
# Success message if sample CSV is made successfully
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.markdown('<div class="success-box">Dataset uploaded successfully.</div>', unsafe_allow_html=True)
    except Exception as error:
        st.error(f"Unable to read this file. Please upload a valid CSV. Error: {error}")
        st.stop()
elif use_sample:
    df = make_sample_dataset()
    st.markdown('<div class="success-box">Sample dataset loaded successfully.</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="warning-box">Upload a CSV file or click “Use Sample Dataset” to begin.</div>', unsafe_allow_html=True)
    st.stop()



# Step 2: Dataset preview

st.header("Step 2: Understand Your Data")

# Show shape of the dataset and number of NAs
summary_col1, summary_col2, summary_col3 = st.columns(3)
summary_col1.metric("Rows", df.shape[0])
summary_col2.metric("Columns", df.shape[1])
summary_col3.metric("Missing Values", int(df.isna().sum().sum()))

# Preview of the dataset (expanded)
with st.expander("Preview dataset", expanded=True):
    st.dataframe(df.head(20), use_container_width=True)

# Column summary of the dataset (expander, not expanded)
with st.expander("Column summary"):
    summary = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing Values": df.isna().sum().values,
        "Unique Values": df.nunique(dropna=True).values
    })
    st.dataframe(summary, use_container_width=True)



# Step 3: Select target

st.header("Step 3: Choose What You Want to Predict")

# Explain how to select the target column
st.write("Select the target column. For this version, the target should be a classification label such as Yes/No, Purchased/Not Purchased, or Category A/B/C.")

# Use dropdown to select
target_column = st.selectbox("Target column", df.columns)

# Warning if the target has too many or too few classes
target_warning = get_problem_warning(df[target_column])
if target_warning:
    st.error(target_warning)
    st.stop()

# Separate the target variable
X = df.drop(columns=[target_column])
y = df[target_column]

# Remove rows where the target is missing.
valid_target_rows = y.notna()
X = X.loc[valid_target_rows]
y = y.loc[valid_target_rows]

# Label encoding for classes
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y.astype(str))
class_labels = label_encoder.classes_

# Print success message and detected classes
st.success(f"Target selected: {target_column}")
st.write(f"Detected classes: {', '.join(class_labels.astype(str))}")



# Step 4: Model execution settings

st.header("Step 4: Run the AutoML Models")

# Explanation of the step and models to be used
st.write("The app will automatically preprocess the data and run three models:")
st.write("✓ Logistic Regression")
st.write("✓ Random Forest")
st.write("✓ Neural Network")

# Slider for setting the test set percentage
test_size = st.slider("Test data percentage", min_value=10, max_value=40, value=20, step=5)

# Button for running the model
run_models = st.button("Run Models")

if not run_models:
    st.info("Click “Run Models” when you are ready.")
    st.stop()

# Run training/test split, using the test_size input from the slider
try:
    stratify_value = y_encoded if len(np.unique(y_encoded)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=test_size / 100,
        random_state=42,
        stratify=stratify_value
    )

# Try again without stratification when stratification cannot be done
except ValueError:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=test_size / 100,
        random_state=42
    )

# Run the functions to run the models
try:
    results, trained_pipelines, numeric_features, categorical_features = train_models(X_train, X_test, y_train, y_test)
   
# Error message to explain potential reasons when the model could not run
except Exception as error:
    st.error("The models could not run successfully. Try checking for unusual columns, too little data, or an unsuitable target column.")
    st.exception(error)
    st.stop()



# Step 5: Results dashboard

st.header("Step 5: Results Dashboard")

# View the result of each model
results_df = pd.DataFrame([
    {"Model": item["Model"], "Accuracy": item["Accuracy"]}
    for item in results
]).sort_values(by="Accuracy", ascending=False)

# Find the best model
best_model_name = results_df.iloc[0]["Model"]
best_accuracy = results_df.iloc[0]["Accuracy"]
best_predictions = next(item["Predictions"] for item in results if item["Model"] == best_model_name)

# Display the recommendation
st.markdown(
    f'<div class="success-box"><strong>Recommended Model:</strong> {best_model_name} with {best_accuracy:.2%} accuracy.<br>{explain_accuracy(best_accuracy)}</div>',
    unsafe_allow_html=True
)

# Display the best model, accuracy, and number of test set rows
metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric("Best Model", best_model_name)
metric_col2.metric("Best Accuracy", f"{best_accuracy:.2%}")
metric_col3.metric("Test Rows", len(y_test))

# Compare the models
st.subheader("Model Comparison")
st.dataframe(
    results_df.assign(Accuracy=results_df["Accuracy"].map(lambda value: f"{value:.2%}")),
    use_container_width=True
)

# Bar plot to compare performances
plot_accuracy(results_df)

# Build a confusion matrix
st.subheader("Confusion Matrix for Recommended Model")
cm = confusion_matrix(y_test, best_predictions)
plot_confusion_matrix(cm, class_labels, f"Confusion Matrix: {best_model_name}")

# Classification report of each model in a tabluar form
st.subheader("Detailed Classification Report")
report = classification_report(
    y_test,
    best_predictions,
    target_names=class_labels.astype(str),
    output_dict=True,
    zero_division=0
)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df, use_container_width=True)



# Human-centered explanation of what the results mean and how the dataset was processed

st.header("What These Results Mean")
st.write(
    "Accuracy shows the percentage of test examples the model predicted correctly. "
    "The confusion matrix shows where the model was correct and where it made mistakes. "
    "The recommended model is the one with the highest accuracy on the test data."
)

with st.expander("Data processing details"):
    st.write("Numeric columns were filled with median values and scaled.")
    st.write("Categorical columns were filled with the most common value and converted using one-hot encoding.")
    st.write(f"Numeric columns detected: {numeric_features}")
    st.write(f"Categorical columns detected: {categorical_features}")

st.caption("AutoML Insight Studio is designed for clear guidance, minimal setup, and beginner-friendly interpretation.")
