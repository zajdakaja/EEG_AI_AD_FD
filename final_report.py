import os
import json
import matplotlib.pyplot as plt
import shap
from openai import OpenAI  # Użycie nowej biblioteki OpenAI

# Funkcja wczytania klucza API z pliku
def load_api_key():
    """
    Wczytuje klucz API z pliku `api_key.json`.
    """
    api_key_path = os.path.join(os.getcwd(), "env", "api_key.json")
    try:
        with open(api_key_path, "r") as f:
            api_key_data = json.load(f)
            return api_key_data["api_key"]
    except FileNotFoundError:
        print(f"Błąd: Plik `api_key.json` nie został znaleziony pod ścieżką: {api_key_path}")
        exit()
    except KeyError:
        print(f"Błąd: Brak klucza API w pliku `api_key.json`. Sprawdź strukturę pliku.")
        exit()

# Funkcja analizy wyników za pomocą GenAI
def analyze_metrics_with_genai(api_key, metrics_before, metrics_after, shap_summary_path, output_file):
    """
    Analizuje metryki modelu oraz wyniki SHAP za pomocą GenAI.
    """
    # Tworzenie instancji klienta OpenAI
    client = OpenAI(api_key=api_key)

    # Treść zapytania
    comparison_query = f"""
    Oto wyniki klasyfikacji modelu przed i po optymalizacji:

    Wyniki przed optymalizacją:
    {json.dumps(metrics_before, indent=4)}

    Wyniki po optymalizacji:
    {json.dumps(metrics_after, indent=4)}

    Dodatkowo załączono globalny wykres SHAP, przedstawiający wpływ cech na predykcje modelu:
    Wykres znajduje się w: {shap_summary_path}

    Proszę o interpretację zmian, w tym które metryki uległy największej poprawie, jakie mogą być tego przyczyny oraz jakie dalsze kroki można podjąć.
    """
    try:
        print("Wysyłanie zapytania do GenAI...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Jesteś ekspertem ds. analizy danych i modeli AI."},
                {"role": "user", "content": comparison_query},
            ]
        )
        # Pobieranie wyników
        result = response.choices[0].message.content

        # Zapis wyników do pliku
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"Analiza wyników zapisana w: {output_file}")

    except Exception as e:
        print(f"Błąd podczas analizy metryk za pomocą GenAI:\n{e}")
# Funkcja generująca raport końcowy
def generate_results_report(metrics, shap_summary_path, shap_waterfall_path, confusion_matrix_path, genai_summary_path, output_path):
    """
    Tworzy raport końcowy uwzględniający metryki, wykresy SHAP i wnioski GenAI.
    """
    try:
        report_content = f"""
# Results and Discussion

## 1. Model Evaluation Metrics
- **Accuracy**: {metrics['Accuracy']:.2f}
- **Precision**: {metrics['Precision']:.2f}
- **Recall**: {metrics['Recall']:.2f}
- **F1-Score**: {metrics['F1-score']:.2f}

### Confusion Matrix
The following confusion matrix illustrates the classification results:
![Confusion Matrix]({confusion_matrix_path})

## 2. SHAP Analysis
### Global Feature Importance
The SHAP Summary Plot highlights the global importance of features:
![SHAP Summary Plot]({shap_summary_path})

### Local Explanation
The following SHAP Waterfall Plot explains the prediction for a specific sample:
![SHAP Waterfall Plot]({shap_waterfall_path})

## 3. Insights from GenAI
The following insights were generated using GenAI based on the feature statistics and classification results:
"""
        try:
            with open(genai_summary_path, "r", encoding="utf-8") as genai_file:
                genai_summary = genai_file.read()
            report_content += f"\n{genai_summary}\n"
        except FileNotFoundError:
            report_content += "\nBrak wniosków z GenAI. Sprawdź konfigurację lub przeprowadź analizę ponownie.\n"

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(report_content)

        print(f"Raport zapisany w: {output_path}")

    except Exception as e:
        print(f"Błąd podczas generowania raportu: {e}")

# Główna funkcja
def main():
    # Przykładowe metryki (zastąp odpowiednimi wynikami)
    metrics_before = {
        "Accuracy": 0.85,
        "Precision": 0.86,
        "Recall": 0.84,
        "F1-score": 0.85
    }
    metrics_after = {
        "Accuracy": 0.89,
        "Precision": 0.90,
        "Recall": 0.89,
        "F1-score": 0.89
    }

    # Ścieżki do plików wizualizacji i wniosków
    shap_summary_path = "plots/shap_summary_plot.png"
    shap_waterfall_path = "plots/shap_waterfall_sample_0.png"
    confusion_matrix_path = "plots/confusion_matrix_after.png"
    genai_summary_path = "eda_outputs/genai_analysis.txt"
    output_path = "eda_outputs/results_and_discussion.md"

    # Analiza wyników za pomocą GenAI
    api_key = load_api_key()
    analyze_metrics_with_genai(
        api_key,
        metrics_before,
        metrics_after,
        shap_summary_path="eda_outputs/shap_summary_plot.png",
        output_file="eda_outputs/genai_analysis.txt"
    )
    # Generowanie raportu końcowego
    generate_results_report(metrics_after, shap_summary_path, shap_waterfall_path, confusion_matrix_path, genai_summary_path, output_path)

if __name__ == "__main__":
    main()
