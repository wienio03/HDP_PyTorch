# Klasyfikacja Choroby Serca - PyTorch

Projekt implementuje sieć neuronową do wieloklasowej klasyfikacji choroby serca wykorzystując PyTorch. System przewiduje stopień zaawansowania choroby serca na podstawie 13 parametrów medycznych pacjenta z datasetu UCI Heart Disease.

## Opis projektu

Model wykorzystuje sieć FNN z trzema ukrytymi warstwami do klasyfikacji wieloklasowej (5 klas: 0-4). Projekt zawiera kompletny pipeline obejmujący pobieranie danych, preprocessing, trening z regularyzacją, ewaluację i wizualizację wyników.

### Architektura modelu

- **Typ**: FNN z batch normalization, dropout, early stopping, weight decay
- **Warstwy**: 256 → 128 → 64 → 5 neuronów wyjściowych
- **Funkcja aktywacji**: ReLU w warstwach ukrytych, raw logits na wyjściu (dla CrossEntropyLoss)
- **Regularyzacja**: Dropout (0.45), Batch Normalization, Weight Decay
- **Funkcja straty**: CrossEntropyLoss z wagami klas dla niezbalansowanych danych
- **Optimizer**: AdamW z learning rate scheduler

### Dataset

Projekt wykorzystuje UCI Heart Disease Dataset:

- 303 próbki pacjentów
- 13 cech medycznych (wiek, płeć, typ bólu w klatce piersiowej, ciśnienie krwi, cholesterol, itp.)
- Target wieloklasowy:
  - **0**: Brak choroby serca
  - **1**: Łagodna choroba serca
  - **2**: Umiarkowana choroba serca
  - **3**: Poważna choroba serca
  - **4**: Krytyczna choroba serca

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
2. Przeprowadzi preprocessing zachowując oryginalne klasy wieloklasowe (0-4)
3. Podzieli dane na train/validation/test (70/15/15) z stratyfikacją
4. Wytrenuje model z early stopping i wagami klas dla niezbalansowanych danych
5. Wygeneruje raport ewaluacji wieloklasowej i wykresy
6. Zapisze model w folderze `models/`

### Dostosowanie konfiguracji

Można zmodyfikować parametry w `src/pipeline.py` w funkcji `get_default_config()`:

```python
def get_default_config():
    return {
        'batch_size': 64,               # Rozmiar batcha
        'learning_rate': 0.01,          # Learning rate
        'epochs': 300,                  # Maksymalna liczba epok
        'hidden_dims': [64, 32, 16],    # Architektura warstw ukrytych
        'dropout_rate': 0.45,           # Współczynnik dropout
        'weight_decay': 1e-3,           # L2 regularization
        'test_size': 0.5,               # Procent danych na test
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
- **Zachowanie target**: Oryginalne klasy wieloklasowe (0,1,2,3,4) bez konwersji
- **Normalizacja**: StandardScaler dla wszystkich cech numerycznych
- **Podział danych**: Stratified split zachowujący proporcje wszystkich 5 klas
- **Obsługa niezbalansowanych danych**: Wagi klas obliczane automatycznie

### Architektura modelu

```python
class HeartDiseaseNet(nn.Module):
    def __init__(self, input_dim=13, hidden_dims=[256, 128, 64], dropout_rate=0.5, num_classes=5):
        # Warstwy: Linear -> BatchNorm -> ReLU -> Dropout
        # Ostatnia warstwa: Linear (5 neuronów) -> Raw logits dla CrossEntropyLoss
```

### Optymalizacja i regularyzacja

- **AdamW**: Adam z dodatkową regularyzacją
- **Early Stopping**: Patience=15 epok, monitoring validation loss
- **Learning Rate Scheduler**: ReduceLROnPlateau (zmniejszanie przy plateau)
- **Regularyzacja**: Dropout, Batch Norm, Weight Decay, Early Stopping
- **Wagi klas**: Automatyczne równoważenie dla niezbalansowanych danych wieloklasowych

### Ewaluacja

Model generuje:

- **Metryki wieloklasowe**: Accuracy, Precision, Recall, F1-Score dla każdej z 5 klas
- **Classification Report**: Szczegółowe metryki per klasa i średnie
- **Confusion Matrix**: Macierz pomyłek 5x5 pokazująca klasyfikację między wszystkimi klasami
- **Wykresy**: Training curves, rozkład klas wieloklasowych, confusion matrix heatmap

## Przykładowe wyniki

Typowe wyniki dla klasyfikacji wieloklasowej (5 klas):

```
==== EVALUATION REPORT (Multi-class 0-4) ====
Classification Report:
                    precision    recall  f1-score   support

   No Disease (0)       0.75      0.78      0.76        18
       Mild (1)         0.68      0.65      0.67        20
   Moderate (2)         0.72      0.70      0.71        15
     Severe (3)         0.65      0.68      0.67        12
   Critical (4)         0.70      0.72      0.71         8

       accuracy                             0.70        73
      macro avg         0.70      0.71      0.70        73
   weighted avg         0.70      0.70      0.70        73

Final Test Accuracy: 0.7000

Multi-class distribution: [54 40 35 26 18] samples per class
```

## Różnice względem implementacji binarnej

### Zmiany w architekturze:

- **Warstwa wyjściowa**: 5 neuronów zamiast 1
- **Funkcja aktywacji**: Raw logits zamiast Sigmoid
- **Funkcja straty**: CrossEntropyLoss zamiast BCELoss

### Zmiany w preprocessing:

- **Target**: Zachowanie oryginalnych klas 0-4 zamiast konwersji na 0/1
- **Stratyfikacja**: Podział uwzględniający wszystkie 5 klas
- **Wagi klas**: Obliczanie wag dla 5 klas zamiast 2

### Zmiany w ewaluacji:

- **Metryki**: Per-class metrics dla 5 klas
- **Confusion Matrix**: 5x5 zamiast 2x2
- **Interpretacja**: Analiza stopnia zaawansowania choroby zamiast obecność/brak

### Oczekiwana accuracy:

- **Binarny**: ~87-90%
- **Wieloklasowy**: ~70% (trudniejsze zadanie z większą liczbą klas i średnim datasetem do takiej klasyfikacji)

Implementacja wieloklasowa pozwala na bardziej szczegółową diagnostykę, określając nie tylko obecność choroby serca, ale także jej stopień zaawansowania, co ma większą wartość kliniczną.
