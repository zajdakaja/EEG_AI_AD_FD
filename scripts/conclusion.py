import os
import json

def load_metrics(metrics_path):
    """
    Wczytuje metryki z pliku JSON.
    """
    try:
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        print(f"Metryki wczytane z: {metrics_path}")
        return metrics
    except FileNotFoundError:
        print(f"Błąd: Plik {metrics_path} nie został znaleziony.")
        return {}

def generate_conclusion(genai_analysis_path, metrics_before_path, metrics_after_path, output_path):
    """
    Tworzy plik Markdown z podsumowaniem projektu.
    """
    try:
        # Wczytanie metryk
        metrics_before = load_metrics(metrics_before_path)
        metrics_after = load_metrics(metrics_after_path)

        # Wczytanie analizy GenAI
        with open(genai_analysis_path, "r", encoding="utf-8") as file:
            genai_summary = file.read()

        # Treść podsumowania
        conclusion_content = f"""
# Conclusion

## 1. Summary of Key Findings
- Model accuracy improved from {metrics_before.get('accuracy', 0):.2%} to {metrics_after.get('accuracy', 0):.2%}.
- Precision increased from {metrics_before.get('precision', 0):.2%} to {metrics_after.get('precision', 0):.2%}.
- Recall improved from {metrics_before.get('recall', 0):.2%} to {metrics_after.get('recall', 0):.2%}.
- F1-score rose from {metrics_before.get('f1_score', 0):.2%} to {metrics_after.get('f1_score', 0):.2%}.
- SHAP analysis provided interpretability:
  - Global feature importance and local explanations for individual predictions.

## 2. Significance of Results
- Early detection of neurodegenerative disorders such as AD and FTD is critical for timely intervention.
- Identifying the key EEG features contributing to the classification helps advance understanding of these diseases.
- SHAP and GenAI offered transparency and interpretability, building trust in AI-based diagnostic tools.

## 3. Future Directions
- Increase sample size to improve generalizability and robustness of the model.
- Incorporate multimodal data (e.g., MRI, genetic data) for more comprehensive predictions.
- Explore other algorithms like deep neural networks or ensemble methods to further improve performance.

## 4. Next Steps
- Publish findings in a relevant journal or present at a conference.
- Collaborate with neuroscience experts to validate findings.
- Integrate the model into a clinical setting for real-world testing.

## Appendix: GenAI Insights
{genai_summary}
"""

        # Zapisanie do pliku
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(conclusion_content)

        print(f"Podsumowanie zapisane w: {output_path}")

    except Exception as e:
        print(f"Błąd podczas generowania podsumowania: {e}")

# Ścieżki do plików
genai_analysis_path =os.path.join(os.getcwd(), "eda_outputs", "genai_analysis.txt")

metrics_before_path =os.path.join(os.getcwd(), "eda_outputs", "metrics_before.json")
metrics_after_path = os.path.join(os.getcwd(), "eda_outputs", "metrics_after.json")
output_path =os.path.join(os.getcwd(), "eda_outputs",  "conclusion.md")

# Wywołanie funkcji
generate_conclusion(genai_analysis_path, metrics_before_path, metrics_after_path, output_path)
