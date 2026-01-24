import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time

#Benennung und Ressourcenzuweisung
modelName = "mri_model_LD_e20.pth"
print("Training model: " + modelName)
DEVICE = torch.device("cpu")
# num_blocks = len(models.efficientnet_b0().features)
# print(f"Anzahl der Blöcke im EfficientNet-B0: {num_blocks}")
#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Hardware used: ", DEVICE)

# Bildtransformationen inkl. Data Augmentation
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Daten laden
train_data = datasets.ImageFolder('data2/train', transform=transform)
val_data = datasets.ImageFolder('data2/val', transform=transform)

# DataLoader erstellen
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)

# Klassen automatisch ermitteln
classes = train_data.classes
print(f"Images for training: {len(train_data)}, images for validation: {len(val_data)}")
print("Classes: ", classes)

# Modell laden: EfficientNet-B0 mit vortrainierten ImageNet-Gewichten
# model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

# Eigenes Modell definieren
class DefineModel(nn.Module):
    def __init__(self, num_classes=4):
        super(DefineModel, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
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

model = DefineModel(num_classes=len(classes)).to(DEVICE)
print("Model created")


# Trainingsdauer berechnen
print("Starting training...")
train_start_time = time.time() 


# Loss-Funktion und Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-4)

train_losses = []
val_losses = []

# Training
EPOCHS = 20
for epoch in range(EPOCHS):
    # Trainingsphase
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} "
          #f", Loss: {running_loss/len(train_loader):.4f}"
    )
    train_losses.append(running_loss/len(train_loader))

    # Validierungsphase
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    val_accuracy = 100 * correct / total
#Ausgabe
    #print(f"Validation loss: {avg_val_loss:.4f} \n\t"
        #f"Validation accuracy: {val_accuracy:.2f}%")


train_end_time = time.time()
train_time = train_end_time - train_start_time
print(f"Training completed in {train_time:.2f} seconds ({train_time/60:.2f} minutes).")

#Modell speichern
torch.save(model.state_dict(), modelName)
print("New model saved as " + modelName)
print("Script finished.")

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('LD_e20_loss_plot.png')
plt.show()