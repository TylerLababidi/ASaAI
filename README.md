
# 🧠 ASaAI – Alzheimer Stadium Classification

Dieses Projekt trainiert ein Convolutional Neural Network (CNN), um verschiedene Alzheimer-Stadien anhand von MRT-Bildern zu klassifizieren.
Verwendet wird **EfficientNet-B0** aus PyTorch – optional mit vortrainierten ImageNet-Gewichten.
Trainingsergebnisse werden ausgewertet (Accuracy, Confusion Matrix, Classification Report) und visuell dargestellt.

---

## 🚀 Features

* **Datenaugmentation:**
  Rotation, horizontales Flip, Helligkeit-/Kontrastvariation
* **GPU-Unterstützung (CUDA)** – automatisch, falls verfügbar
* **Modellspeicherung:** `mini_model_pretrained.pth`
* **Evaluation:**

  * Accuracy
  * Confusion Matrix
  * Classification Report
* **Visualisierung:** Heatmap der Confusion Matrix

---

## 📦 Voraussetzungen

* **Python 3.10+**
* **Virtuelle Umgebung empfohlen**

### Abhängigkeiten installieren:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install matplotlib seaborn scikit-learn
```

---

## 📁 Ordnerstruktur

```
ASaAI/
│
├─ Data/
│   ├─ train/          # Trainingsbilder, Unterordner pro Klasse
│   └─ val/            # Validierungsbilder, Unterordner pro Klasse
│
├─ train_and_eval.py   # Trainings- und Evaluationsscript
├─ README.md           # Dieses Dokument
└─ mini_model_pretrained.pth  # Optional gespeichertes Modell
```

### Beispiel für Datenstruktur:

```
Data/train/MildDemented/img001.png
Data/train/ModerateDemented/img002.png
Data/val/VeryMildDemented/img003.png
Data/train/NonDemented/img004.png
```

---

## ▶️ Nutzung

### **1. Projekt klonen**

```bash
git clone <repo-url>
cd ASaAI
python -m venv venv
.\venv\Scripts\activate   # Windows
# oder
source venv/bin/activate  # Linux/Mac
```

### **2. Abhängigkeiten installieren**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install matplotlib seaborn scikit-learn
```

### **3. Training starten**

```bash
python train_and_eval.py
```

Was dann passiert:

* GPU wird automatisch erkannt
* EfficientNet-B0 wird geladen (mit oder ohne Pretrained Weights)
* Trainings- und Validierungsphase laufen
* Modell wird gespeichert
* Accuracy & Reports werden angezeigt
* Confusion-Matrix-Heatmap wird visualisiert

---

## 📥 Modell für spätere Nutzung laden

```python
import torch
from torchvision import models
from torch import nn

model = models.efficientnet_b0(
    weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
)

model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)
model.load_state_dict(torch.load("mini_model_pretrained.pth"))
model.eval()
```

---

## 🧩 Erklärung des Codes

### **1. Imports**

PyTorch, torchvision, sklearn, Matplotlib, Seaborn

### **2. Device-Auswahl**

CUDA, falls verfügbar → sonst CPU

### **3. Datenvorbereitung**

* Resize auf 224×224
* Normalisierung
* Data Augmentation für robustere Modelle

### **4. DataLoader**

Batchweise Bilder für Training & Validierung

### **5. Modell**

* EfficientNet-B0
* Vortrainierte Gewichte optional
* Klassifizierer wird für z. B. 4 Alzheimer-Klassen angepasst

### **6. Loss & Optimizer**

* CrossEntropyLoss
* Adam Optimizer

### **7. Training**

* Epochen durchlaufen
* Loss je Epoch ausgeben

### **8. Evaluation**

* Accuracy
* Confusion Matrix
* Classification Report

### **9. Visualisierung**

Heatmap der Confusion Matrix mit seaborn

---

## 📊 Hinweise & Tipps

### Gute Accuracy erreichen:

* Vortrainierte Gewichte nutzen (IMAGENET1K_V1)
* Data Augmentation erweitern
* Lernrate anpassen 
* Batch Size erhöhen ✅
* Mehr Bilder nutzen ✅

### Häufige Probleme:

| Problem               | Ursache                                    | Lösung                                          |
| --------------------- | ------------------------------------------ | ----------------------------------------------- |
| CUDA fehlt            | Torch ohne GPU installiert                 | Torch mit CUDA neu installieren                 |
| Niedrige Accuracy     | Wenig Daten, Overfitting oder Underfitting | Augmentieren, mehr Daten, LR ändern, Feintuning |
| Warnungen beim Report | Klasse fehlt im Testset                    | Mehr balanced Validation                        |

---

## 🔜 Nächste Schritte

* Komplettes Dataset trainieren
* Feintuning der EfficientNet-Basis
* Weitere Evaluationsmetriken wie ROC/PR-Kurven
* Modell exportieren für Web/Apps (z. B. ONNX)


