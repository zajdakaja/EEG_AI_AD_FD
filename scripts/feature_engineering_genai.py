import os
import mne
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from openai import OpenAI
import json

# Funkcja do ekstrakcji cech
def extract_features(raw, sfreq):
    psd, freqs = mne.time_frequency.psd_array_welch(
        raw.get_data(), sfreq=sfreq, fmin=0.5, fmax=45, n_fft=int(sfreq * 2), n_overlap=int(sfreq * 1))

    band_freqs = {
        'Delta_power': (0.5, 4),
        'Theta_power': (4, 8),
        'Alpha_power': (8, 13),
        'Beta_power': (13, 30),
        'Gamma_power': (30, 45)
    }

    features = {}
    for band, (fmin, fmax) in band_freqs.items():
        band_indices = np.where((freqs >= fmin) & (freqs <= fmax))[0]
        features[band] = np.mean(psd[:, band_indices], axis=1).tolist()

    return features

# Funkcja przetwarzania danych EEG
def process_subject(raw_file, group):
    raw = mne.io.read_raw_fif(raw_file, preload=True)
    sfreq = raw.info['sfreq']
    features = extract_features(raw, sfreq)
    features['label'] = group
    return features

# Funkcja zapisu cech
def save_features_and_labels(features_list, features_file):
    rows = []
    for entry in features_list:
        label = entry.pop("label")
        for i in range(len(next(iter(entry.values())))):
            row = {band: entry[band][i] for band in entry.keys()}
            row["label"] = label
            rows.append(row)

    features_df = pd.DataFrame(rows)
    features_df.to_csv(features_file, index=False)
    print(f"Cechy i etykiety zapisane do pliku: {features_file}")

# Wizualizacja cech
def visualize_features(features_df):
    os.makedirs("plots", exist_ok=True)

    for col in features_df.columns[:-1]:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=features_df, x='label', y=col)
        plt.title(f'Wykres pudełkowy cechy {col}', fontsize=14)
        plt.xlabel('Grupa', fontsize=12)
        plt.ylabel(f'{col} (jednostka)', fontsize=12)
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(f"plots/boxplot_{col}.png")
        plt.close()

    for col in features_df.columns[:-1]:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=features_df, x=col, hue='label', bins=30, kde=True)
        plt.title(f'Histogram cechy {col}', fontsize=14)
        plt.xlabel(f'{col} (jednostka)', fontsize=12)
        plt.ylabel('Liczba', fontsize=12)
        plt.tight_layout()
        plt.savefig(f"plots/histogram_{col}.png")
        plt.close()

# Analiza korelacji
def correlation_analysis(features_df):
    corr_matrix = features_df.iloc[:, :-1].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
    plt.title("Macierz korelacji cech", fontsize=14)
    plt.tight_layout()
    plt.savefig("plots/correlation_matrix.png")
    plt.close()
    print("Macierz korelacji zapisana w: plots/correlation_matrix.png")

# Funkcja do inicjalizacji klienta OpenAI
def init_openai_client():
    api_key_path = os.path.join(os.getcwd(), "env", "api_key.json")
    try:
        with open(api_key_path, "r") as f:
            api_key = json.load(f)["api_key"]
        return OpenAI(api_key=api_key)
    except FileNotFoundError:
        print("Nie znaleziono pliku z kluczem API.")
        exit()

# Generowanie wniosków z GenAI
def generate_insights(client, stats_summary):
    user_query = f"""
    Mam zbiór danych EEG zawierający cechy Delta, Theta, Alpha, Beta i Gamma dla różnych grup (np. AD, CN).
    Oto statystyki opisowe:
    {stats_summary}

    Oceń różnice między grupami, wskaż znaczące cechy i zaproponuj ich wykorzystanie w klasyfikacji chorób neurodegeneracyjnych.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": user_query}],
        )
        result = response.choices[0].message.content
        output_path = os.path.join(os.getcwd(), "eda_outputs", "eda_summary.txt")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"Podsumowanie zapisane w: {output_path}")
    except Exception as e:
        print(f"Błąd podczas generowania wniosków z GenAI: {e}")

# Główna funkcja
def main():
    data_path = "C:/Users/Dell/neuro_ai_project/data/preprocessed"
    output_features = "C:/Users/Dell/neuro_ai_project/data/features.csv"

    groups = {
        'AD': ["sub-001", "sub-002", "sub-003"],
        'FTD': ["sub-004", "sub-005"],
        'CN': ["sub-006", "sub-007"]
    }

    features_list = []

    for group, subjects in groups.items():
        for subject in subjects:
            subject_path = os.path.join(data_path, f"{subject}_preprocessed.fif")
            if os.path.exists(subject_path):
                print(f"Przetwarzanie: {subject_path}")
                features = process_subject(subject_path, group)
                features_list.append(features)
            else:
                print(f"Plik nie istnieje: {subject_path}")

    save_features_and_labels(features_list, output_features)

    # Analiza danych
    features_df = pd.read_csv(output_features)
    visualize_features(features_df)
    correlation_analysis(features_df)

    # Generowanie wniosków z GenAI
    client = init_openai_client()
    stats_summary = features_df.describe().to_string()
    generate_insights(client, stats_summary)

if __name__ == "__main__":
    main()
