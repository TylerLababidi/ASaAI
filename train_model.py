import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time

print("Script gestartet ✅")
num_blocks = len(models.efficientnet_b0().features)
print(f"Anzahl der Blöcke im EfficientNet-B0: {num_blocks}")

# Gerät wählen: GPU (cuda) falls verfügbar, sonst CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Benutztes Gerät:", DEVICE)

# Bildtransformationen inkl. Data Augmentation
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

# Daten laden
train_data = datasets.ImageFolder('Data/train', transform=transform)
val_data = datasets.ImageFolder('Data/val', transform=transform)

# DataLoader erstellen
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)

# Klassen automatisch ermitteln
classes = train_data.classes
print(f"Trainingsbilder: {len(train_data)}, Validierungsbilder: {len(val_data)}")
print("Klassen:", classes)

# Modell laden: EfficientNet-B0 mit vortrainierten ImageNet-Gewichten
# model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

# Eigenes Modell definieren
class MyMRTModel(nn.Module):
    def __init__(self, num_classes=4):
        super(MyMRTModel, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*14*14, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = MyMRTModel(num_classes=len(classes)).to(DEVICE)
print("Eigenes Modell erstellt ✅")


# Trainingsdauer berechnen
print("Starting training...")
train_start_time = time.time() 


# Loss-Funktion und Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-4)


# Training
EPOCHS = 20  #
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


train_end_time = time.time()
train_time = train_end_time - train_start_time
print(f"Training completed in {train_time:.2f} seconds ({train_time/60:.2f} minutes).")

#Modell speichern
torch.save(model.state_dict(), "my_mrt_model.pth")
print("Eigenes Modell gespeichert ✅")
print("Script beendet ✅")