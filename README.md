# 🛡️ Machine Learning Based Network Intrusion Detection System

A machine-learning based Network Intrusion Detection System (NIDS) that classifies network traffic as **normal or malicious** using the **NSL-KDD dataset** and compares the performance of multiple machine-learning models.

## 🚀 Features

* 📊 **NSL-KDD Dataset Analysis** — Uses `KDDTrain+` and `KDDTest+`
* 🔍 **Exploratory Data Analysis** — Examines traffic, protocols, and attack distribution
* 🧹 **Data Preprocessing** — Handles categorical features, feature selection, encoding, and scaling
* 🏷️ **Binary Classification** — Classifies traffic as Normal or Attack
* 📈 **Model Comparison** — Compares three machine-learning approaches
* 📊 **Performance Evaluation** — Measures Accuracy, Precision, Recall, and F1 Score
* 🔲 **Confusion Matrices** — Visualizes classification results for each model
* 📉 **Metric Comparison** — Provides a visual comparison of model performance

## 🤖 Machine Learning Models

The project evaluates three classification models:

### 1. Logistic Regression

Used as a baseline classification model for detecting normal and malicious network traffic.

### 2. Random Forest

An ensemble-based classifier using multiple decision trees for intrusion detection.

### 3. Neural Network

An `MLPClassifier` with two hidden layers:

```text
128 neurons → 64 neurons
```

using the ReLU activation function.

## 🔄 Machine Learning Pipeline

```text
NSL-KDD Dataset
       ↓
Data Loading
       ↓
Exploratory Data Analysis
       ↓
Missing Value & Data Type Checks
       ↓
Binary Attack Label Creation
       ↓
Categorical Feature Encoding
       ↓
Train/Test Feature Alignment
       ↓
Feature Scaling
       ↓
┌───────────────────────┐
│ Logistic Regression   │
│ Random Forest         │
│ Neural Network        │
└───────────────────────┘
       ↓
Predictions
       ↓
Accuracy / Precision / Recall / F1
       ↓
Confusion Matrices
       ↓
Model Comparison
```

## 📂 Project Structure

```text
ML_IDS_project/
│
├── intrusion_detection.ipynb
├── notebook_script.py
├── KDDTrain+.txt
├── KDDTest+.txt
├── test_samples.txt
├── .gitignore
└── README.md
```

## 🧪 Dataset

This project uses the **NSL-KDD dataset**, with:

* `KDDTrain+.txt` — Training dataset
* `KDDTest+.txt` — Test dataset

The original attack information is converted into a binary classification problem:

```text
0 → Normal Traffic
1 → Attack Traffic
```

## ⚙️ Preprocessing

The dataset contains both numerical and categorical network traffic features.

The preprocessing pipeline includes:

1. Removing the `difficulty` column
2. Creating a binary attack label
3. Checking for missing values
4. Identifying categorical features
5. One-hot encoding:

   * `protocol_type`
   * `service`
   * `flag`
6. Aligning training and testing feature columns
7. Standardizing numerical features using `StandardScaler`

The scaler is fitted only on the training data and then applied to both training and test data.

## 📊 Evaluation Metrics

Each model is evaluated using:

* **Accuracy** — Overall percentage of correct predictions
* **Precision** — How many predicted attacks were actually attacks
* **Recall** — How many actual attacks were successfully detected
* **F1 Score** — Combined measure of Precision and Recall

Confusion matrices are also generated for each model.

## 🔐 Cybersecurity Perspective

Recall is particularly important in an intrusion detection system because failing to detect a real attack can be more serious than generating a false alert.

The project also highlights the challenge of detecting less common or previously underrepresented attack types in the test data, which reflects an important limitation of machine-learning based intrusion detection systems.

## 🛠️ Technologies Used

* **Python**
* **Jupyter Notebook**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**
* **Seaborn**

## ▶️ Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/Mannat29/ML_IDS_project.git
cd ML_IDS_project
```

### 2. Install the required libraries

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### 3. Run the notebook

```bash
jupyter notebook intrusion_detection.ipynb
```

Alternatively, the Python script containing the complete analysis can be executed directly:

```bash
python notebook_script.py
```

## 📌 Project Status

**Status: Completed ML-based intrusion detection project**

The project demonstrates the complete workflow of preparing network traffic data, training multiple classification models, evaluating their performance, and analyzing their suitability for intrusion detection.

## 👩‍💻 Author

**Mannat Tomar**

GitHub: [@Mannat29](https://github.com/Mannat29)
