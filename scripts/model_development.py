import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt
import os


# Funkcja wczytania cech
def load_features(feature_file):
    """
    Wczytuje DataFrame z cechami i etykietami.
    """
    data = pd.read_csv(feature_file)
    X = data.drop(columns=['label'])
    y = data['label']
    return X, y

# Funkcja treningu i oceny modeli
def train_and_evaluate(X_train, X_test, y_train, y_test):
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    results = {}
    best_model = None
    best_score = 0

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        results[name] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1}
        print(f"Model: {name}")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        if acc > best_score:
            best_score = acc
            best_model = model

    return results, best_model

# Funkcja zapisu modelu
def save_model(model, output_path):
    joblib.dump(model, output_path)
    print(f"Model zapisany do: {output_path}")

# Wizualizacja wyników modeli
def plot_results(results):
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    plt.figure(figsize=(10, 6))
    for metric in metrics:
        values = [results[model][metric] for model in results]
        plt.plot(results.keys(), values, marker='o', label=metric)
    plt.title('Porównanie wyników modeli')
    plt.xlabel('Model')
    plt.ylabel('Wartość metryki')
    plt.legend()
    plt.grid()
    plt.show()

# Główna funkcja
def main():
    feature_file = os.path.join(os.getcwd(), "data", "features.csv")
    model_output_path = os.path.join(os.getcwd(), "best_model.pkl")

    # Wczytanie danych
    X, y = load_features(feature_file)

    # Kodowanie etykiet
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Podział danych
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Trening i ocena
    results, best_model = train_and_evaluate(X_train, X_test, y_train, y_test)

    # Zapis najlepszego modelu
    save_model(best_model, model_output_path)

    # Wizualizacja wyników
    plot_results(results)

if __name__ == "__main__":
    main()
