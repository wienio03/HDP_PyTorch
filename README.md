# Klasyfikacja Choroby Serca - PyTorch

Projekt implementuje sieć neuronową do binarnej klasyfikacji choroby serca wykorzystując PyTorch. System przew### Optymalizacja i regularyzacja

- **AdamW**: Adam z dodatkową regularyzacją
- **Early Stopping**: Patience=15 epok, monitoring validation loss
- **Learning Rate Scheduler**: ReduceLROnPlateau (zmniejszanie przy plateau)
- **Regularyzacja**: Dropout, Batch Norm, Weight Decay, Early Stopping
- **Wagi klas**: BCEWithLogitsLoss z pos_weight dla niezbalansowanych danych binarnych

## Opis projektu

Model wykorzystuje sieć FNN z trzema ukrytymi warstwami do klasyfikacji binarnej (choroba/brak choroby). Projekt zawiera kompletny pipeline obejmujący pobieranie danych, preprocessing, trening z regularyzacją, ewaluację i wizualizację wyników.

### Architektura modelu

- **Typ**: FNN z batch normalization, dropout, early stopping, weight decay
- **Warstwy**: 256 → 128 → 64 → 1 neuron wyjściowy
- **Funkcja aktywacji**: ReLU w warstwach ukrytych, raw logits na wyjściu
- **Regularyzacja**: Dropout (0.5), Batch Normalization, Weight Decay
- **Funkcja straty**: BCEWithLogitsLoss z wagami klas dla niezbalansowanych danych
- **Optimizer**: AdamW z learning rate scheduler

### Dataset

Projekt wykorzystuje UCI Heart Disease Dataset:

- 303 próbki pacjentów
- 13 cech medycznych (wiek, płeć, typ bólu w klatce piersiowej, ciśnienie krwi, cholesterol, itp.)
- Target oryginalnie wieloklasowy (0-4) konwertowany na binarny: 0 (brak choroby), 1 (choroba)

## Struktura projektu

```
HDP_PyTorch/
├── main.py                       # Punkt wejścia aplikacji
├── requirements.txt              # Zależności projektu
├── src/
│   ├── data/
│   │   └── data_processing.py    # Preprocessing i DataLoader
│   ├── logger/
│   │   └── __init__.py           # Konfiguracja logowania
│   ├── model.py                  # Architektura sieci neuronowej
│   ├── train.py                  # Logika treningu i walidacji
│   ├── evaluate.py               # Ewaluacja i wizualizacja
│   ├── pipeline.py               # Główny pipeline projektu
│   ├── utils.py                  # Funkcje pomocnicze
│   └── download_data.py          # Pobieranie danych z UCI
├── models/                       # Zapisane modele
├── logs/                         # Logi treningu
└── README.md
```

## Instalacja i uruchomienie

### Wymagania systemowe

- Python 3.8+
- CUDA (opcjonalnie, dla GPU)

### Instalacja zależności

```bash
# Klonowanie repozytorium
git clone <repo-url>
cd HDP_PyTorch

# Instalacja zależności
pip install -r requirements.txt
```

### Główne zależności

```
torch>=2.0.0
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
```

## Użycie

### Podstawowe uruchomienie

```bash
python main.py
```

Program automatycznie:

1. Pobierze dane z UCI repository (jeśli nie istnieją jeszcze lokalnie)
2. Przeprowadzi preprocessing i konwersję na klasyfikację binarną
3. Podzieli dane na train/validation/test (70/15/15) ze stratyfikacją
4. Wytrenuje model z early stopping i wagami klas dla niezbalansowanych danych
5. Wygeneruje raport ewaluacji i wykresy z poprawionymi metrykami
6. Zapisze model w folderze `models/`

### Dostosowanie konfiguracji

Można zmodyfikować parametry w `src/pipeline.py` w funkcji `get_default_config()`:

```python
def get_default_config():
    return {
        'batch_size': 64,               # Rozmiar batcha
        'learning_rate': 0.001,         # Learning rate
        'epochs': 300,                  # Maksymalna liczba epok
        'hidden_dims': [256, 128, 64],  # Architektura warstw ukrytych
        'dropout_rate': 0.5,            # Współczynnik dropout
        'weight_decay': 1e-3,           # L2 regularization
        'test_size': 0.15,              # Procent danych na test
        'val_size': 0.15,               # Procent danych na walidację
    }
```

### Użycie pipeline

```python
from src.pipeline import HeartDiseasePipeline, get_default_config

# Tworzenie pipeline z domyślną konfiguracją
config = get_default_config()
pipeline = HeartDiseasePipeline(config)

# Uruchomienie kompletnego pipeline
results = pipeline.run_pipeline()

# Lub krok po kroku
train_loader, val_loader, test_loader, input_dim = pipeline.load_and_preprocess_data()
model = pipeline.create_model(input_dim)
history = pipeline.train_model(train_loader, val_loader)
results, predictions, probabilities, targets = pipeline.evaluate_model(test_loader)
```

## Szczegóły implementacji

### Preprocessing danych

- **Pobieranie**: Automatyczne pobieranie z UCI ML Repository
- **Konwersja target**: Z wieloklasowego (0,1,2,3,4) na binarny (0=brak choroby, 1-4=choroba)
- **Normalizacja**: StandardScaler dla wszystkich cech numerycznych
- **Podział danych**: Stratified split zachowujący proporcje klas
- **Obsługa niezbalansowanych danych**: Automatyczne wagi klas w BCEWithLogitsLoss

### Architektura modelu

```python
class HeartDiseaseNet(nn.Module):
    def __init__(self, input_dim=13, hidden_dims=[256, 128, 64], dropout_rate=0.5):
        # Warstwy: Linear -> BatchNorm -> ReLU -> Dropout
        # Ostatnia warstwa: Linear (raw logits dla BCEWithLogitsLoss)
```

### Optymalizacja i regularyzacja

- **AdamW**: Adam z dodatkową regularyzacją
- **Early Stopping**: Patience=15 epok, monitoring validation loss
- **Learning Rate Scheduler**: ReduceLROnPlateau (zmniejszanie przy plateau)
- **Regularizacja**: Dropout, Batch Norm, Weight Decay, Early Stopping

### Ewaluacja

Model generuje:

- **Metryki**: Accuracy, Precision, Recall dla klasyfikacji binarnej
- **Classification Report**: Precision, Recall, F1-Score dla każdej klasy
- **Confusion Matrix**: Macierz pomyłek
- **Wykresy**: Training curves, rozkład klas, confusion matrix

## Przykładowe wyniki

Typowe wyniki dla klasyfikacji binarnej:

```
==== EVALUATION REPORT (Binary Classification) ====
Classification Report:
              precision    recall  f1-score   support

   No Disease       0.87      0.90      0.88        30
      Disease       0.89      0.86      0.87        26

     accuracy                           0.88        56
    macro avg       0.88      0.88      0.88        56
 weighted avg       0.88      0.88      0.88        56

Final Test Accuracy: 0.8750
```
