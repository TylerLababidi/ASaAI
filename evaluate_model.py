import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

print("Script gestartet ✅")

# Gerät wählen: GPU (cuda) falls verfügbar, sonst CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Benutztes Gerät:", DEVICE)

# Bildtransformationen inkl. Data Augmentation
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

# Daten laden
train_data = datasets.ImageFolder('Data/train', transform=transform)
val_data = datasets.ImageFolder('Data/val', transform=transform)

# DataLoader erstellen
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16)

# Klassen automatisch ermitteln
classes = train_data.classes
print(f"Trainingsbilder: {len(train_data)}, Validierungsbilder: {len(val_data)}")
print("Klassen:", classes)

# Modell laden: EfficientNet-B0 mit vortrainierten ImageNet-Gewichten
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

# Basis einfrieren (Feature-Extractor)
for param in model.parameters():
    param.requires_grad = False

# Klassifizierer anpassen
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
model = model.to(DEVICE)

# Loss & Optimizer nur für Klassifizierer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

# Training
EPOCHS = 1  # Vortrainiertes Modell benötigt weniger Epochen
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader):.4f}")

# Modell speichern
torch.save(model.state_dict(), "mini_model_pretrained.pth")
print("Mini-Modell mit vortrainierten Gewichten gespeichert ✅")

# Evaluation
model.eval()
all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_labels.extend(labels.numpy())
        all_preds.extend(preds.cpu().numpy())

# Accuracy berechnen
accuracy = accuracy_score(all_labels, all_preds)
print(f"\nAccuracy: {accuracy*100:.2f}%")

# Confusion Matrix & Classification Report
cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(classes))))
report = classification_report(
    all_labels,
    all_preds,
    labels=list(range(len(classes))),
    target_names=classes,
    zero_division=0
)

print("\nClassification Report:")
print(report)

# Heatmap der Confusion Matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap="Blues")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.title("Confusion Matrix")
plt.show()
