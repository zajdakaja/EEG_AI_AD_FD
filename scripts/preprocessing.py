import os
import mne
import json
import pandas as pd

def load_data(subject_folder):
    """
    Wczytuje dane EEG, metadane i informacje o kanałach.
    
    Args:
        subject_folder (str): Ścieżka do folderu uczestnika (np. sub-001).
    
    Returns:
        raw (mne.io.Raw): Obiekt Raw EEG.
        metadata (dict): Metadane z pliku .json.
        channels (pd.DataFrame): Informacje o kanałach z pliku .tsv.
    """
    eeg_file = os.path.join(subject_folder, 'eeg', f'{os.path.basename(subject_folder)}_task-eyesclosed_eeg.set')
    json_file = os.path.join(subject_folder, 'eeg', f'{os.path.basename(subject_folder)}_task-eyesclosed_eeg.json')
    tsv_file = os.path.join(subject_folder, 'eeg', f'{os.path.basename(subject_folder)}_task-eyesclosed_channels.tsv')

    # Wczytywanie danych EEG
    raw = mne.io.read_raw_eeglab(eeg_file, preload=True)

    # Wczytywanie metadanych
    with open(json_file, 'r') as f:
        metadata = json.load(f)

    # Wczytywanie kanałów
    channels = pd.read_csv(tsv_file, sep='\t')

    return raw, metadata, channels

def preprocess_eeg(file_path, save_path):
    """
    Przetwarzanie sygnału EEG: filtrowanie, ICA, re-referencjonowanie i zapis.
    
    Args:
        file_path (str): Ścieżka do pliku EEG (.set).
        save_path (str): Ścieżka do zapisu przetworzonych danych.
    """
    # Wczytywanie surowego EEG
    raw = mne.io.read_raw_eeglab(file_path, preload=True)

    # Filtrowanie pasmowe
    raw.filter(0.5, 45, fir_design='firwin')
    
    # ICA do usuwania artefaktów
    ica = mne.preprocessing.ICA(n_components=15, random_state=97, max_iter=800)
    ica.fit(raw)
    raw = ica.apply(raw)

    # Re-referencjonowanie
    raw.set_eeg_reference('average', projection=True)

    # Zapis przetworzonych danych
    raw.save(save_path, overwrite=True)
    print(f"Dane przetworzone i zapisane do: {save_path}")

def main():
    # Definicja głównych folderów z użyciem os.getcwd()
    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, "data", "ds004504-download", "derivatives")
    output_dir = os.path.join(current_dir, "data", "preprocessed")
    os.makedirs(output_dir, exist_ok=True)

    # Iteracja przez uczestników
    for subject in os.listdir(data_dir):
        eeg_dir = os.path.join(data_dir, subject, "eeg")
        if not os.path.isdir(eeg_dir):
            continue

        for file in os.listdir(eeg_dir):
            if file.endswith(".set"):
                file_path = os.path.join(eeg_dir, file)
                save_path = os.path.join(output_dir, f"{subject}_preprocessed.fif")
                try:
                    preprocess_eeg(file_path, save_path)
                except Exception as e:
                    print(f"Błąd przetwarzania pliku {file_path}: {e}")

if __name__ == "__main__":
    main()
