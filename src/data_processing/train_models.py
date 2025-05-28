import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, recall_score

try:
    from xgboost import XGBClassifier
    xgboost_available = True
except ImportError:
    xgboost_available = False

def train_and_evaluate(model, name, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    report = classification_report(y_val, preds, output_dict=True)
    recall = recall_score(y_val, preds)
    return model, report, recall

def main():
    SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, "..", ".."))
    data_path = os.path.join(PROJECT_ROOT, "data", "processed_windows")
    model_path = os.path.join(PROJECT_ROOT, "models")
    metrics_path = os.path.join(model_path, "model_metrics.csv")
    os.makedirs(model_path, exist_ok=True)

    X = joblib.load(os.path.join(data_path, "X_features.pkl"))
    y = joblib.load(os.path.join(data_path, "y_labels.pkl"))

    # Split data: train_val (80%) and test (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # From train_val split into train (60%) and val (20%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, stratify=y_temp, test_size=0.25, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(model_path, "feature_scaler.pkl"))

    models = [
        (RandomForestClassifier(n_estimators=100, random_state=42), "Random Forest", X_train, X_val),
        (SVC(kernel='rbf', probability=True, random_state=42), "SVM", X_train_scaled, X_val_scaled),
        (LogisticRegression(max_iter=1000, class_weight='balanced'), "Logistic Regression", X_train_scaled, X_val_scaled),
        (KNeighborsClassifier(n_neighbors=5), "KNN", X_train_scaled, X_val_scaled),
        (AdaBoostClassifier(n_estimators=100, random_state=42), "AdaBoost", X_train, X_val)
    ]

    if xgboost_available:
        models.append((XGBClassifier(use_label_encoder=False, eval_metric='logloss'), "XGBoost", X_train, X_val))

    best_model = None
    best_name = ""
    best_recall = 0
    all_metrics = []

    for model, name, Xtr, Xva in models:
        trained_model, report, recall = train_and_evaluate(model, name, Xtr, y_train, Xva, y_val)
        all_metrics.append({"Model": name, "Recall_Seizure": recall, **report['1']})
        if recall > best_recall:
            best_recall = recall
            best_model = trained_model
            best_name = name

    # Retrain best model on full train + val
    X_full_train = np.vstack([X_train, X_val])
    y_full_train = np.concatenate([y_train, y_val])
    if best_name in ["SVM", "Logistic Regression", "KNN"]:
        X_full_train = scaler.transform(X_full_train)
        X_test_final = X_test_scaled
    else:
        X_test_final = X_test

    best_model.fit(X_full_train, y_full_train)
    test_preds = best_model.predict(X_test_final)
    test_report = classification_report(y_test, test_preds, output_dict=True)

    print(f"\nBest Model on Validation: {best_name}")
    print("\nFinal Test Results:")
    print(classification_report(y_test, test_preds))
    print(confusion_matrix(y_test, test_preds))

    # Save best model
    joblib.dump(best_model, os.path.join(model_path, f"best_model_{best_name.lower().replace(' ', '_')}.pkl"))

    # Save metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nModel metrics saved to {metrics_path}")

if __name__ == "__main__":
    main()
