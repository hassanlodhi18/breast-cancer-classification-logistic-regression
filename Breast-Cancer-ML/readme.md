# Breast Cancer Classification Using Logistic Regression

A machine learning project that classifies breast tumor observations as
**Benign** or **Malignant** using **Logistic Regression**. The notebook
covers data loading, data inspection, preprocessing, feature scaling,
model training, evaluation, model serialization, and a sample
prediction. It also includes an optional Google Gemini integration that
generates general physical precautions when the model predicts a
malignant result.

> **Medical Disclaimer:** This project is for educational and
> demonstration purposes only. A machine learning prediction is **not a
> confirmed cancer diagnosis** and must not be used as a substitute for
> evaluation by a qualified healthcare professional.

------------------------------------------------------------------------

## Project Overview

The project uses a tabular breast-cancer dataset containing **569
observations** and **33 columns** in the original file.

The workflow is:

1.  Load the dataset with Pandas.
2.  Inspect the data using `head()`, `shape`, `info()`, `describe()`,
    and missing-value checks.
3.  Remove the `id` column and the completely empty `Unnamed: 32`
    column.
4.  Convert the target variable:
    -   `M` → `1` (Malignant)
    -   `B` → `0` (Benign)
5.  Separate features (`X`) and target (`y`).
6.  Split the data into training and testing sets using an 80/20 split
    with stratification.
7.  Standardize the features using `StandardScaler`.
8.  Train a Logistic Regression classifier.
9.  Evaluate the model using multiple classification metrics.
10. Save the trained model and scaler using Joblib.
11. Run a prediction on a sample patient observation.
12. Optionally generate general physical precautions using Google Gemini
    when the prediction is malignant.

------------------------------------------------------------------------

## Dataset

The original dataset contains:

-   **569 rows**
-   **33 columns**
-   `diagnosis` as the target column
-   `id` as an identifier
-   `Unnamed: 32` as a completely empty column
-   Numeric tumor-related measurements used as model features

After preprocessing:

-   `id` is removed because it is an identifier rather than a predictive
    feature.
-   `Unnamed: 32` is removed because all 569 values are missing.
-   The model uses **30 numeric features**.
-   The target is binary:
    -   `0` = Benign
    -   `1` = Malignant

### Feature Groups

The notebook uses measurements belonging to three groups:

-   **Mean features:** measurements such as `radius_mean`,
    `texture_mean`, `area_mean`, and related characteristics.
-   **Standard error features:** measurements such as `radius_se`,
    `texture_se`, `area_se`, and related characteristics.
-   **Worst features:** measurements such as `radius_worst`,
    `texture_worst`, `area_worst`, and related characteristics.

------------------------------------------------------------------------

## Technologies Used

-   **Python**
-   **Pandas** --- data loading and manipulation
-   **Scikit-learn** --- preprocessing, Logistic Regression, train/test
    splitting, and evaluation
-   **Joblib** --- saving the trained model and scaler
-   **Google GenAI** --- optional generation of patient-friendly general
    precautions
-   **Jupyter Notebook** --- development and experimentation environment

------------------------------------------------------------------------

## Machine Learning Workflow

### 1. Data Loading

The notebook loads the CSV dataset using Pandas:

``` python
df = pd.read_csv("data.csv")
```

It then checks the dataset structure, data types, summary statistics,
and missing values.

### 2. Data Cleaning

The notebook identifies `Unnamed: 32` as completely missing and removes
it along with the `id` column:

``` python
df = df.drop(columns=["id", "Unnamed: 32"])
```

No rows are removed because the missing values are confined to the empty
`Unnamed: 32` column.

### 3. Target Encoding

The categorical diagnosis labels are converted into numerical values:

``` python
df["diagnosis"] = df["diagnosis"].map({
    "M": 1,
    "B": 0
})
```

This creates a binary classification problem.

### 4. Feature and Target Separation

``` python
X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]
```

-   `X` contains the 30 input features.
-   `y` contains the binary diagnosis target.

### 5. Train/Test Split

The data is split into:

-   **80% training data**
-   **20% testing data**

A `random_state` of `42` is used for reproducibility, and `stratify=y`
preserves the class distribution between the training and testing sets.

``` python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

The resulting test set contains **114 observations**.

### 6. Feature Scaling

Because the numerical features have different ranges, `StandardScaler`
is used:

``` python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The scaler is fitted only on the training data and then applied to the
test data.

### 7. Model Training

The project uses Logistic Regression:

``` python
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)
```

The model also produces class probabilities using:

``` python
model.predict_proba(X_test_scaled)
```

------------------------------------------------------------------------

## Model Performance

The notebook reports the following results on the test set:

  Metric         Score
  ----------- --------
  Accuracy      96.49%
  Precision     97.50%
  Recall        92.86%
  F1 Score      95.12%
  ROC-AUC       99.60%

### Classification Report

  Class                    Precision   Recall   F1-Score   Support
  ---------------------- ----------- -------- ---------- ---------
  Benign                        0.96     0.99       0.97        72
  Malignant                     0.97     0.93       0.95        42
  **Overall Accuracy**                          **0.96**   **114**

### Confusion Matrix

``` text
[[71,  1],
 [ 3, 39]]
```

Using the notebook's class order:

-   **71** Benign observations were correctly classified as Benign.
-   **1** Benign observation was classified as Malignant.
-   **3** Malignant observations were classified as Benign.
-   **39** Malignant observations were correctly classified as
    Malignant.

The model therefore achieved strong performance on this particular test
split. These results should not be interpreted as evidence that the
model is clinically validated.

------------------------------------------------------------------------

## Model Saving

The notebook uses Joblib to save the trained model and scaler:

``` python
joblib.dump(model, "breast_cancer_logistic_regression.pkl")
joblib.dump(scaler, "breast_cancer_scaler.pkl")
```

The two generated files are:

``` text
breast_cancer_logistic_regression.pkl
breast_cancer_scaler.pkl
```

The scaler must be retained because new observations need to be
transformed using the same scaling process used during training.

------------------------------------------------------------------------

## Sample Prediction

The notebook creates a sample patient observation containing the same 30
features used during model training.

The observation is scaled before prediction:

``` python
new_patient_scaled = scaler.transform(new_patient)
prediction = model.predict(new_patient_scaled)
```

The notebook's sample prediction is:

``` text
Prediction: Malignant
```

The sample values are taken from an observation present in the dataset.
The prediction demonstrates how the saved preprocessing and trained
classifier can be applied to a new input.

------------------------------------------------------------------------

## Google Gemini Integration

The notebook optionally connects the prediction workflow to Google
Gemini.

The integration:

1.  Checks the model prediction.
2.  If the prediction is `Malignant`, sends a predefined prompt to
    Gemini.
3.  Requests short, general physical precautions.
4.  Explicitly instructs the generated response not to prescribe
    medicines or cancer treatments.
5.  Reminds the user that the ML prediction is not a confirmed
    diagnosis.

The notebook imports the API key from a local configuration module:

``` python
from config import GEMINI_API_KEY
```

The API key should **not** be hard-coded into the notebook or committed
to a public GitHub repository.

A local `config.py` file can contain the key, while the file should be
excluded from version control.

------------------------------------------------------------------------

## Installation

Install the main Python dependencies:

``` bash
pip install pandas scikit-learn joblib jupyter
```

For the optional Gemini integration:

``` bash
pip install google-genai
```

------------------------------------------------------------------------

## How to Run

### 1. Clone or download the project

Place the notebook and dataset in your project directory.

### 2. Update the dataset path

The notebook currently uses a local Windows path:

``` python
df = pd.read_csv(r"C:\Users\Nexskill\Downloads\breast_cancer\data.csv")
```

Change this path to the location of your own `data.csv`.

For a project repository, a relative path is preferable, for example:

``` python
df = pd.read_csv("data/data.csv")
```

### 3. Open the notebook

Run:

``` bash
jupyter notebook
```

Then open:

``` text
breast_cancer.ipynb
```

### 4. Run the cells

Execute the notebook from top to bottom so that:

-   the data is loaded,
-   preprocessing is performed,
-   the model is trained,
-   evaluation metrics are calculated,
-   the model and scaler are saved,
-   and the sample prediction is generated.

### 5. Optional Gemini setup

If you want to use the Gemini section, create the required local
configuration containing `GEMINI_API_KEY` and keep that configuration
file out of source control.

------------------------------------------------------------------------

## Project Structure

A recommended repository structure is:

``` text
breast-cancer-classification/
│
├── data/
│   └── data.csv
│
├── breast_cancer.ipynb
├── breast_cancer_logistic_regression.pkl
├── breast_cancer_scaler.pkl
├── config.py              # Local only; do not commit API keys
├── .gitignore
└── README.md
```

If the dataset or generated model files are not intended to be
committed, they can also be excluded through `.gitignore`.

------------------------------------------------------------------------

## Important Notes and Limitations

-   This is an **educational machine learning project**, not a clinical
    diagnostic system.
-   The reported metrics come from the notebook's single 80/20
    stratified train/test split.
-   No cross-validation or external validation is implemented in the
    notebook.
-   No hyperparameter tuning is implemented.
-   The notebook does not establish clinical safety, generalization to
    other populations, or regulatory approval.
-   A false negative can be especially important in a medical
    classification problem; model performance should therefore be
    assessed carefully before any real-world use.
-   The Gemini-generated precautions are general informational guidance
    and are not a treatment recommendation.
-   API keys and other credentials should never be committed to GitHub.

------------------------------------------------------------------------

## Future Improvements

Potential improvements to make the project more robust include:

-   Add cross-validation.
-   Compare Logistic Regression with other classification algorithms.
-   Perform hyperparameter tuning.
-   Add ROC and Precision-Recall curves.
-   Visualize the confusion matrix.
-   Analyze feature importance or Logistic Regression coefficients.
-   Build a reusable prediction function.
-   Use a proper preprocessing/model pipeline.
-   Add input validation for new patient data.
-   Add a Streamlit interface for demonstration.
-   Add automated tests.
-   Evaluate the model on an independent external dataset.
-   Improve the handling of the medical-use disclaimer and user-facing
    safety messaging.

------------------------------------------------------------------------

## Conclusion

This project demonstrates an end-to-end binary classification workflow
using Logistic Regression. Starting from raw tabular data, it performs
data inspection, cleaning, target encoding, train/test splitting,
feature standardization, model training, evaluation, model
serialization, and sample prediction.

The notebook achieved **96.49% test accuracy** and a **99.60% ROC-AUC**
on its recorded test split. The project also demonstrates how a machine
learning prediction can be connected to a generative AI component for
general, patient-friendly informational output while explicitly stating
that the prediction is not a medical diagnosis.
