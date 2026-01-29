# 🧠 ASaAI – Alzheimer Stadium Classification

Dieses Projekt klassifiziert Alzheimer‑Stadien anhand von MRT‑Bildern mit **PyTorch**.

Im Gegensatz zur ersten Version mit EfficientNet wird nun **ein eigenes CNN (MyMRTModel bzw. MRI_LAD_e20)** verwendet. Zusätzlich enthält das Projekt **ein Gradio‑Webinterface**, mit dem einzelne MRT‑Bilder interaktiv ausgewertet und die Klassifikations‑Wahrscheinlichkeiten visualisiert werden können.

Anmerkung: MyMRTModel und MRI_LAD_e20 referenzieren dasselbe CNN, wobei MyMRTModel als Arbeitstitel zu verstehen ist. 

Wie das Projekt "einfach" zu nutzen ist, erfahren Sie ganz unten.
---

## 🚀 Features

* **Eigenes CNN (MyMRTModel/ MRI_LAD_e20)** statt EfficientNet
* **GPU‑Unterstützung (CUDA)**, automatisch falls verfügbar
* **Training, Evaluation & Inferenz getrennt**
* **Modellspeicherung:** `my_mrt_model.pth`
* **Evaluation:**

  * Accuracy
  * Confusion Matrix
  * Classification Report
* **Visualisierung:**

  * Confusion‑Matrix‑Heatmap
  * Balkendiagramm der Klassen‑Wahrscheinlichkeiten
* **Web‑UI mit Gradio** für Live‑Inference

---

## 📦 Voraussetzungen

* **Python 3.10+**
* **Virtuelle Umgebung empfohlen**

### Abhängigkeiten installieren

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install matplotlib seaborn scikit-learn gradio pillow
```

---

## 📁 Ordnerstruktur

```
ASaAI/
│
├─ Data/
│   ├─ train/          # Trainingsdaten (Unterordner = Klassen)
│   └─ val/            # Validierungsdaten
│
├─ train_model.py      # Training des CNN
├─ evaluate_model.py   # Evaluation & Confusion Matrix
├─ app.py              # Gradio Interface für Inferenz
├─ my_mrt_model.pth    # Gespeichertes Modell
└─ README.md
```

### Beispiel Datenstruktur

```
Data/train/Mild/img001.png
Data/train/Moderate/img002.png
Data/train/Non/img003.png
Data/train/VeryMild/img004.png
```

---

## 🧠 Klassen

```python
classes = ['Mild', 'Moderate', 'Non', 'VeryMild']
```

Die Klassen werden beim Training automatisch aus der Ordnerstruktur gelesen.

---

## ▶️ Nutzung

### 1️⃣ Projekt klonen & Umgebung aktivieren

```bash
git clone <repo-url>
cd ASaAI
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\\Scripts\\activate    # Windows
```

### 2️⃣ Training starten

```bash
python train_model.py
```

**Ablauf:**

* Bilder werden auf **224×224** skaliert
* Normalisierung auf Bereich **[-1, 1]**
* Training für **20 Epochen**
* Optimizer: **Adam (lr = 5e‑4)**
* Loss: **CrossEntropyLoss**
* Modell wird als `my_mrt_model.pth` gespeichert

---

### 3️⃣ Evaluation

```bash
python evaluate_model.py
```

Ausgabe:

* Accuracy in Prozent
* Classification Report (Precision, Recall, F1‑Score)
* Confusion‑Matrix‑Heatmap

---

### 4️⃣ Gradio Web‑Interface starten

```bash
python app.py
```

Funktionen:

* Upload eines MRT‑Bildes
* Anzeige der vorhergesagten Klasse
* Balkendiagramm mit Wahrscheinlichkeiten

---

## 🧩 Modellarchitektur (MyMRTModel)

**CNN‑Aufbau:**

* 4 Convolution‑Blöcke (3×3 Kernel)
* ReLU‑Aktivierung
* MaxPooling nach jedem Block
* Fully Connected Layer (256 Neuronen)
* Dropout (0.5)
* Output‑Layer: Anzahl Klassen

```text
Input (3×224×224)
→ Conv(16)
→ Conv(32)
→ Conv(64)
→ Conv(128)
→ FC(256)
→ Output (4 Klassen)
```

---

## 📊 Evaluation & Visualisierung

* **Accuracy** mit `sklearn.metrics.accuracy_score`
* **Classification Report** mit `classification_report`
* **Confusion Matrix** als Seaborn‑Heatmap
* **Inference‑Visualisierung:** Balkendiagramm der Softmax‑Wahrscheinlichkeiten

---

## ⚠️ Hinweise

* Architektur beim Laden **muss exakt** dem Trainingsmodell entsprechen
* Klassenreihenfolge ergibt sich aus `ImageFolder`
* Unbalancierte Datensätze beeinflussen Accuracy stark

---

## 🔜 Nächste Schritte

* Neuer Datensatz
* Klassen‑Balancing (Weighted Loss)

---

## ✅ Status

✔ Training
✔ Evaluation
✔ Web‑Interface
✔ Visualisierung

---

## 🖥️ Anwendung der App in der eigenen Umgebung

* Klonen des Main-Branches
* Öffnen mit einer IDE der Wahl (für das Projekt wurde PyCharm verwendet, daher wird dieses auch empfohlen)
* Es müssen zum Laden der App folgende Bibliotheken installiert sein:
** Gradio 6.3.0
** Matplotlib 3.10.8
** PyTorch 2.9.1
  Torchvision 0.25.0
** 



