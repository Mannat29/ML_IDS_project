# Cell 0 (code)
import pandas as pd

# Column names for the NSL-KDD dataset
columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type', 'difficulty'
]

# Load the datasets
train_df = pd.read_csv('KDDTrain+.txt', names=columns)
test_df = pd.read_csv('KDDTest+.txt', names=columns)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
train_df.head()

# Cell 1 (code)
# See all unique attack types in the dataset
print("Unique attack types:")
print(train_df['attack_type'].value_counts())

# Cell 2 (code)
# Drop the 'difficulty' column - it's not a real network feature
train_df = train_df.drop(columns=['difficulty'])
test_df = test_df.drop(columns=['difficulty'])

# Create binary label: 0 = normal, 1 = attack
train_df['label'] = train_df['attack_type'].apply(lambda x: 0 if x == 'normal' else 1)
test_df['label'] = test_df['attack_type'].apply(lambda x: 0 if x == 'normal' else 1)

print("Training label distribution:")
print(train_df['label'].value_counts())

# Cell 3 (code)
# Check for missing values
print("Missing values in training set:")
print(train_df.isnull().sum().sum())

print("\nMissing values in test set:")
print(test_df.isnull().sum().sum())

# Check data types
print("\nData types:")
print(train_df.dtypes.value_counts())

# Cell 4 (code)
# Find the categorical columns
categorical_cols = train_df.select_dtypes(include='object').columns.tolist()
print("Categorical columns:", categorical_cols)

# Cell 5 (code)
# See unique values in each categorical column
for col in ['protocol_type', 'service', 'flag']:
    print(f"{col}: {train_df[col].nunique()} unique values")
    print(train_df[col].unique())
    print()

# Cell 6 (code)
import matplotlib.pyplot as plt
import seaborn as sns

# Plot 1: Attack vs Normal distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='label', data=train_df, palette='Set2')
plt.title('Normal vs Attack Traffic')
plt.xticks([0, 1], ['Normal', 'Attack'])
plt.xlabel('Traffic Type')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Cell 7 (code)
# Plot 2: Protocol type distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='protocol_type', hue='protocol_type', data=train_df, palette='Set2', legend=False)
plt.title('Protocol Type Distribution')
plt.xlabel('Protocol Type')
plt.ylabel('Count')
plt.tight_layout()
plt.show()


# Cell 8 (code)
# Plot 3: Attack type by protocol
plt.figure(figsize=(6, 4))
sns.countplot(x='protocol_type', hue='label', data=train_df, palette='Set2')
plt.title('Normal vs Attack by Protocol Type')
plt.xlabel('Protocol Type')
plt.ylabel('Count')
plt.legend(['Normal', 'Attack'])
plt.tight_layout()
plt.show()

# Cell 9 (code)
# Drop attack_type column (we already have our binary label)
train_df = train_df.drop(columns=['attack_type'])
test_df = test_df.drop(columns=['attack_type'])

# Separate features (X) and label (y)
X_train = train_df.drop(columns=['label'])
y_train = train_df['label']

X_test = test_df.drop(columns=['label'])
y_test = test_df['label']

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Cell 10 (code)
# One-hot encode categorical columns
categorical_cols = ['protocol_type', 'service', 'flag']

X_train = pd.get_dummies(X_train, columns=categorical_cols)
X_test = pd.get_dummies(X_test, columns=categorical_cols)

# Align columns - make sure train and test have same columns
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

print("X_train shape after encoding:", X_train.shape)
print("X_test shape after encoding:", X_test.shape)

# Cell 11 (code)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Fit ONLY on training data, transform both train and test
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Scaling complete!")
print("X_train sample mean (should be ~0):", X_train.mean().round(2))
print("X_train sample std (should be ~1):", X_train.std().round(2))

# Cell 12 (code)
#First Model: Logistic Regression
from sklearn.linear_model import LogisticRegression
import time

print("Training Logistic Regression...")
start = time.time()

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

end = time.time()
print(f"Training complete! Time taken: {round(end - start, 2)} seconds")

# Cell 13 (code)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Make predictions
lr_predictions = lr_model.predict(X_test)

# Calculate metrics
lr_accuracy = accuracy_score(y_test, lr_predictions)
lr_precision = precision_score(y_test, lr_predictions)
lr_recall = recall_score(y_test, lr_predictions)
lr_f1 = f1_score(y_test, lr_predictions)

print("Logistic Regression Results:")
print(f"Accuracy:  {lr_accuracy:.4f}")
print(f"Precision: {lr_precision:.4f}")
print(f"Recall:    {lr_recall:.4f}")
print(f"F1 Score:  {lr_f1:.4f}")

#"Logistic Regression has high Precision but low Recall — meaning it's conservative in flagging attacks and misses many real ones. For an intrusion detection system, Recall is critical because missing an attack is more dangerous than a false alarm. This motivated me to try a more powerful model."

# Cell 14 (code)
# Plot confusion matrix
cm = confusion_matrix(y_test, lr_predictions)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Attack'],
            yticklabels=['Normal', 'Attack'])
plt.title('Logistic Regression - Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()

# Cell 15 (code)
#Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

print("Training Random Forest...")
start = time.time()

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

end = time.time()
print(f"Training complete! Time taken: {round(end - start, 2)} seconds")

# Cell 16 (code)
# Make predictions
rf_predictions = rf_model.predict(X_test)

# Calculate metrics
rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_precision = precision_score(y_test, rf_predictions)
rf_recall = recall_score(y_test, rf_predictions)
rf_f1 = f1_score(y_test, rf_predictions)

print("Random Forest Results:")
print(f"Accuracy:  {rf_accuracy:.4f}")
print(f"Precision: {rf_precision:.4f}")
print(f"Recall:    {rf_recall:.4f}")
print(f"F1 Score:  {rf_f1:.4f}")

#"Both models struggled with Recall on the test set because KDDTest+ was designed to be harder, containing rare attack types underrepresented in training. This reflects a real world challenge in cybersecurity — models trained on known attacks may miss novel ones."

# Cell 17 (code)
# Plot confusion matrix for Random Forest
cm_rf = confusion_matrix(y_test, rf_predictions)

plt.figure(figsize=(6, 4))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Normal', 'Attack'],
            yticklabels=['Normal', 'Attack'])
plt.title('Random Forest - Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()

# Cell 18 (code)
#Neural Network 

from sklearn.neural_network import MLPClassifier

print("Training Neural Network...")
start = time.time()

nn_model = MLPClassifier(
    hidden_layer_sizes=(128, 64),  # 2 hidden layers
    activation='relu',             # ReLU activation function
    max_iter=100,                  # maximum iterations
    random_state=42,
    verbose=True                   # shows progress
)
nn_model.fit(X_train, y_train)

end = time.time()
print(f"\nTraining complete! Time taken: {round(end - start, 2)} seconds")

# Cell 19 (code)
# Make predictions
nn_predictions = nn_model.predict(X_test)

# Calculate metrics
nn_accuracy = accuracy_score(y_test, nn_predictions)
nn_precision = precision_score(y_test, nn_predictions)
nn_recall = recall_score(y_test, nn_predictions)
nn_f1 = f1_score(y_test, nn_predictions)

print("Neural Network Results:")
print(f"Accuracy:  {nn_accuracy:.4f}")
print(f"Precision: {nn_precision:.4f}")
print(f"Recall:    {nn_recall:.4f}")
print(f"F1 Score:  {nn_f1:.4f}")

# Cell 20 (code)
# Compare all models visually
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
lr_scores = [lr_accuracy, lr_precision, lr_recall, lr_f1]
rf_scores = [rf_accuracy, rf_precision, rf_recall, rf_f1]
nn_scores = [nn_accuracy, nn_precision, nn_recall, nn_f1]

x = range(len(metrics))
width = 0.25

plt.figure(figsize=(10, 6))
plt.bar([i - width for i in x], lr_scores, width=width, label='Logistic Regression', color='steelblue')
plt.bar([i for i in x], rf_scores, width=width, label='Random Forest', color='seagreen')
plt.bar([i + width for i in x], nn_scores, width=width, label='Neural Network', color='tomato')

plt.xticks(x, metrics)
plt.ylim(0, 1.1)
plt.title('Model Comparison - All Metrics')
plt.ylabel('Score')
plt.legend()
plt.tight_layout()
plt.show()

# Cell 21 (code)
# Plot confusion matrix for Neural Network
cm_nn = confusion_matrix(y_test, nn_predictions)

plt.figure(figsize=(6, 4))
sns.heatmap(cm_nn, annot=True, fmt='d', cmap='Reds',
            xticklabels=['Normal', 'Attack'],
            yticklabels=['Normal', 'Attack'])
plt.title('Neural Network - Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()

# Cell 22 (code)
# Final summary table
results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'Neural Network'],
    'Accuracy': [lr_accuracy, rf_accuracy, nn_accuracy],
    'Precision': [lr_precision, rf_precision, nn_precision],
    'Recall': [lr_recall, rf_recall, nn_recall],
    'F1 Score': [lr_f1, rf_f1, nn_f1]
})

results = results.set_index('Model')
results = results.round(4)
print(results)

# Cell 23 (code)

