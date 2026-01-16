# python
import torch
from collections import Counter
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import datasets, transforms
import torchvision.models as models
from torch.utils.data import DataLoader
# setzt voraus, dass train_data, val_data, train_loader, val_loader und model bereits geladen sind

batch_size = 32
num_workers = 0  # auf Windows standardmäßig 0; bei Bedarf erhöhen
pin_memory = True if torch.cuda.is_available() else False

# Bildtransformationen inkl. Data Augmentation
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Daten laden
train_data = datasets.ImageFolder('Data/train', transform=transform)
val_data = datasets.ImageFolder('Data/val', transform=transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=pin_memory)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=pin_memory)

num_classes = len(train_data.classes)
model = models.resnet18(pretrained=False)

# 1) Überlappende Dateipfade prüfen
train_paths = [p for p, _ in train_data.samples]
val_paths = [p for p, _ in val_data.samples]
overlap = set(train_paths) & set(val_paths)
print("Anzahl überlappender Dateien:", len(overlap))
if len(overlap) > 0:
    print("Beispiele:", list(overlap)[:5])

# 2) Klassenverteilung anzeigen
def print_class_counts(dataset, name="dataset"):
    counts = Counter([lab for _, lab in dataset.samples])
    idx_to_cls = {v:k for k,v in dataset.class_to_idx.items()}
    print(f"Klassenverteilung für {name}:")
    for idx, cnt in counts.items():
        print(f"  {idx_to_cls[idx]} ({idx}): {cnt}")
print_class_counts(train_data, "train")
print_class_counts(val_data, "val")
print("Train size:", len(train_data), "Val size:", len(val_data))

# 3) Validation: Vorhersagen sammeln und Report drucken
device = next(model.parameters()).device
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for imgs, labs in val_loader:
        imgs = imgs.to(device)
        outs = model(imgs)
        preds = torch.argmax(outs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labs.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
print("Validation accuracy:", (all_preds == all_labels).mean())
print("Confusion matrix:")
print(confusion_matrix(all_labels, all_preds))
print("Classification report:")
print(classification_report(all_labels, all_preds, target_names=train_data.classes, zero_division=0))

# Optional: Schnell-Check auf Trainings-Accuracy (zeigt Overfitting)
do_train_check = True
if do_train_check:
    t_preds = []
    t_labels = []
    with torch.no_grad():
        for imgs, labs in train_loader:
            imgs = imgs.to(device)
            outs = model(imgs)
            preds = torch.argmax(outs, dim=1).cpu().numpy()
            t_preds.extend(preds)
            t_labels.extend(labs.numpy())
    t_preds = np.array(t_preds); t_labels = np.array(t_labels)
    print("Train accuracy (quick):", (t_preds == t_labels).mean())
