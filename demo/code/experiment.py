import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
import json

# Read the data
data = pd.read_csv('raisin.csv')

# Separate features and target
X = data.drop(columns=['Class'])
y = data['Class']

# Extract fold count
n_folds = 10

# Initialize StratifiedKFold with fixed parameters
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Store results
records = []

# Iterate over each fold
for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
    # Split data
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Train a pipeline with StandardScaler and SVC
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel="poly", C=1.0, degree=3, gamma="scale", coef0=0.0))
    ])
    
    # Fit on training data only
    pipeline.fit(X_train, y_train)
    
    # Predict on test data
    predicted_labels = pipeline.predict(X_test)
    
    # Print fold information
    print(f"FOLD {fold_idx}: train size={len(train_idx)}, test size={len(test_idx)}, correct={np.sum(predicted_labels == y_test.values)}")
    
    # Store records: (row_index, fold, prediction)
    for i, label in zip(test_idx, predicted_labels):
        records.append((int(i), fold_idx, str(label)))

# Convert records to DataFrame and save
result_df = pd.DataFrame(records, columns=["row_id", "fold", "prediction"])
result_df.to_csv("predictions.csv", index=False)

# Compute accuracy across all folds
total_correct = 0
total_samples = 0

# We need to compute accuracy over all test sets across folds
# Since each fold has 90 test samples, total samples = 10 * 90 = 900
total_samples = 900

# We must collect all true labels and predicted labels from all folds
true_labels_all = []
predicted_labels_all = []
for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel="poly", C=1.0, degree=3, gamma="scale", coef0=0.0))
    ])
    
    pipeline.fit(X_train, y_train)
    predicted_labels = pipeline.predict(X_test)
    
    # Append true and predicted labels for test set
    true_labels_all.extend(y_test.values)
    predicted_labels_all.extend(predicted_labels)

# Compute overall accuracy
correct = np.sum(np.array(predicted_labels_all) == np.array(true_labels_all))
accuracy = correct / total_samples

print("METRIC", json.dumps({"accuracy": accuracy}))
