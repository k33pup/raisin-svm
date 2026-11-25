from collections import Counter
import csv
from io import StringIO
import json
import math

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def dataset_labels(data):
    return [row["Class"] for row in csv.DictReader(data.splitlines())]


def reference_predictions(data, truth, folds):
    rows = csv.DictReader(data.splitlines())
    features = [name for name in rows.fieldnames if name != "Class"]
    X = np.array([[float(row[name]) for name in features] for row in rows])
    y = np.asarray(truth)
    result = np.empty_like(y)
    for train, test in folds:
        scale = StandardScaler().fit(X[train])
        estimator = SVC(kernel="poly", C=1.0, degree=3, gamma="scale", coef0=0.0)
        estimator.fit(scale.transform(X[train]), y[train])
        result[test] = estimator.predict(scale.transform(X[test]))
    return result


def check_predictions(data, predictions, stdout, recipe, expected):
    truth = dataset_labels(data)
    classes = sorted(set(truth))
    keys = ["row_id", "fold", "prediction"]
    reader = csv.DictReader(StringIO(predictions))
    if reader.fieldnames != keys:
        raise ValueError(f"predictions.csv header is {reader.fieldnames}. "
                         "Set DataFrame columns=['row_id', 'fold', 'prediction'] before to_csv(index=False).")
    labels = {}
    for row in reader:
        index, fold = int(row["row_id"]), int(row["fold"])
        if index in labels or row["prediction"] not in classes:
            raise ValueError("Duplicate row_id or an unknown class label")
        labels[index] = fold, row["prediction"]
    if set(labels) != set(range(len(truth))):
        raise ValueError("Expected one held-out prediction per original row_id (0-based)")

    split = StratifiedKFold(n_splits=recipe["folds"], shuffle=True, random_state=42)
    partitions = list(split.split(truth, truth))
    folds = []
    for number, (train, test) in enumerate(partitions, 1):
        if any(labels[int(i)][0] != number for i in test):
            raise ValueError("Fold membership differs from the fixed StratifiedKFold recipe")
        correct = sum(labels[int(i)][1] == truth[int(i)] for i in test)
        folds.append({"fold": number, "train": len(train), "test": len(test), "correct": correct,
                      "accuracy": correct / len(test)})
    confusion = Counter((label, labels[index][1]) for index, label in enumerate(truth))
    correct = sum(confusion[name, name] for name in classes)
    accuracy = correct / len(truth)
    lines = [line.removeprefix("METRIC ") for line in stdout.splitlines() if line.startswith("METRIC ")]
    if len(lines) != 1:
        raise ValueError("Print exactly one METRIC line: METRIC {\"accuracy\": number}")
    claimed = json.loads(lines[0])["accuracy"]
    if type(claimed) not in (int, float) or not math.isclose(claimed, accuracy, abs_tol=1e-12):
        raise ValueError(f"Printed accuracy does not match the predictions ({accuracy})")
    audited = reference_predictions(data, truth, partitions)
    different = sum(prediction != labels[index][1] for index, prediction in enumerate(audited))
    if different:
        raise ValueError(f"{different} predictions differ from a separate run of the fixed protocol. "
                         "Check the SVC parameters, fold indices and training-only scaling.")
    low, high = expected["min"], expected["max"]
    return {"accuracy": accuracy, "correct": correct, "total": len(truth),
            "classes": classes, "confusion": [[confusion[a, b] for b in classes] for a in classes],
            "folds": folds, "range": [low, high], "in_range": low <= accuracy <= high,
            "reference_disagreements": int(different)}
