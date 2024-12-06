import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import optuna
import joblib
import shap
import matplotlib.pyplot as plt
import os
import json

# Funkcja wczytania danych i modelu
def load_data_and_model(feature_file, model_file=None):
    data = pd.read_csv(feature_file)
    X = data.drop(columns=['label'])
    y = data['label']
    model = None
    if model_file:
        model = joblib.load(model_file)
    return X, y, model

# Funkcja zapisu metryk do pliku JSON
def save_metrics_to_json(metrics, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metryki zapisane w: {output_file}")

# Funkcja oceny modelu
def evaluate_model(model, X_test, y_test, label_encoder, output_file):
    y_pred = model.predict(X_test)
    decoded_y_test = label_encoder.inverse_transform(y_test)
    decoded_y_pred = label_encoder.inverse_transform(y_pred)

    acc = accuracy_score(decoded_y_test, decoded_y_pred)
    prec = precision_score(decoded_y_test, decoded_y_pred, average='weighted', zero_division=0)
    rec = recall_score(decoded_y_test, decoded_y_pred, average='weighted', zero_division=0)
    f1 = f1_score(decoded_y_test, decoded_y_pred, average='weighted', zero_division=0)

    metrics = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-score": f1
    }
    save_metrics_to_json(metrics, output_file)

    print("Metryki modelu:")
    print(f"Accuracy: {acc:.2f}")
    print(f"Precision: {prec:.2f}")
    print(f"Recall: {rec:.2f}")
    print(f"F1-score: {f1:.2f}")
    print("\nRaport klasyfikacji:")
    print(classification_report(decoded_y_test, decoded_y_pred, zero_division=0))

# Funkcja celu dla Optuna
def objective(trial, X_train_resampled, y_train_resampled, X_test, y_test):
    param = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }

    model = XGBClassifier(**param, use_label_encoder=False, eval_metric="mlogloss", random_state=42)
    model.fit(X_train_resampled, y_train_resampled)

    y_pred = model.predict(X_test)
    recall_ftd = recall_score(y_test, y_pred, average=None)[2]
    return recall_ftd

# Funkcja analizy SHAP
def analyze_shap(model, X_test, feature_names):
    """
    Analyzes SHAP values for a given model and test dataset.
    Args:
        model: Trained model to analyze.
        X_test: Test dataset features.
        feature_names: List of feature names.
    """
    # Inicjalizacja narzędzia SHAP Explainer
    explainer = shap.Explainer(model, X_test)
    shap_values = explainer(X_test)

    #  Debugowanie wymiarów danych
    print(f"Rozmiar X_test: {X_test.shape}")
    print(f"Rozmiar SHAP values: {shap_values.values.shape}")

    # Obsługa przypadku wieloklasowego
    if len(shap_values.values.shape) == 3:  # Multiclass scenario
        class_index = 0  # Adjust this index to target a specific class
        shap_values_for_class = shap_values.values[:, :, class_index]
    else:  # Binary classification or regression
        shap_values_for_class = shap_values.values

    # Generowanie globalnego wykresu SHAP
    print("Generowanie wykresu globalnego (średnie wartości SHAP dla cech)...")
    shap.summary_plot(shap_values_for_class, X_test, feature_names=feature_names)

    # Generowanie wykresu zależności
    feature_index = 0  # Adjust index to select the desired feature
    print(f"Generowanie wykresu zależności dla cechy: {feature_names[feature_index]}...")
    shap.dependence_plot(
        feature_index,
        shap_values_for_class,
        X_test,
        feature_names=feature_names
    )

    # Generowanie lokalnej interpretacji
    print("Generowanie lokalnej interpretacji dla pierwszego rekordu...")
    shap.force_plot(
        explainer.expected_value[class_index] if len(shap_values.values.shape) == 3 else explainer.expected_value,
        shap_values_for_class[0],
        X_test.iloc[0],
        feature_names=feature_names
    )

# Główna funkcja
def main():
    feature_file = os.path.join(os.getcwd(), "data", "features.csv")
    X, y, _ = load_data_and_model(feature_file)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Ewaluacja przed optymalizacją
    print("\nEwaluacja modelu przed optymalizacją...")
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    model_before = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)
    model_before.fit(X_train, y_train)
    evaluate_model(model_before, X_test, y_test, label_encoder, "eda_outputs/metrics_before.json")

    # Przetwarzanie danych z SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print("Rozkład etykiet po SMOTE:", pd.Series(y_train_resampled).value_counts())

    # Optymalizacja za pomocą Optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X_train_resampled, y_train_resampled, X_test, y_test),
        n_trials=50
    )
    best_params = study.best_params
    print("Najlepsze parametry:", best_params)

    # Trening i ewaluacja modelu po optymalizacji
    model_after = XGBClassifier(**best_params, use_label_encoder=False, eval_metric="mlogloss", random_state=42)
    model_after.fit(X_train_resampled, y_train_resampled)
    evaluate_model(model_after, X_test, y_test, label_encoder, "eda_outputs/metrics_after.json")

    # Analiza SHAP
    print("\nAnaliza SHAP...")
    analyze_shap(
    model_after,
    pd.DataFrame(X_test, columns=X.columns),  # DataFrame z cechami
    feature_names=X.columns.tolist()  # Nazwy cech

)

if __name__ == "__main__":
    main()
