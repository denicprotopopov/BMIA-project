import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import AdaBoostClassifier

try:
    from xgboost import XGBClassifier
    xgboost_available = True
except ImportError:
    xgboost_available = False

def train_and_evaluate(model, name, X_train, X_test, y_train, y_test, model_path):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"\n{name} Results:")
    print(classification_report(y_test, preds))
    print(confusion_matrix(y_test, preds))
    joblib.dump(model, os.path.join(model_path, f"{name.lower().replace(' ', '_')}.pkl"))

def main():
    SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, "..", ".."))
    data_path = os.path.join(PROJECT_ROOT, "data", "processed_windows")
    model_path = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(model_path, exist_ok=True)

    X = joblib.load(os.path.join(data_path, "X_features.pkl"))
    y = joblib.load(os.path.join(data_path, "y_labels.pkl"))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(model_path, "feature_scaler.pkl"))

    models = [
    (RandomForestClassifier(n_estimators=100, random_state=42), "Random Forest", X_train, X_test),
    (SVC(kernel='rbf', probability=True, random_state=42), "SVM", X_train_scaled, X_test_scaled),
    (LogisticRegression(max_iter=1000, class_weight='balanced'), "Logistic Regression", X_train_scaled, X_test_scaled),
    (KNeighborsClassifier(n_neighbors=5), "KNN", X_train_scaled, X_test_scaled),
    (AdaBoostClassifier(n_estimators=100, random_state=42), "AdaBoost", X_train, X_test)
    ]

    if xgboost_available:
        models.append((XGBClassifier(use_label_encoder=False, eval_metric='logloss'), "XGBoost", X_train, X_test))

    for model, name, Xtr, Xte in models:
        train_and_evaluate(model, name, Xtr, Xte, y_train, y_test, model_path)

if __name__ == "__main__":
    main()
