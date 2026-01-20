import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

modelName = "smallDataset_mri_model_DL_e10.pth"
print("Evaluating model: " + modelName)

# Gerät wählen
DEVICE = torch.device("cpu")
print("Hardware used: ", DEVICE)

# Bildtransformationen
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Daten laden
val_data = datasets.ImageFolder('Data/val', transform=transform)
val_loader = DataLoader(val_data, batch_size=32)
classes = val_data.classes
print(f"Images for validation: {len(val_data)}")
print("Classes:", classes)

# Eigenes Modell definieren (muss identisch zur Training-Architektur sein)
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

# Modell laden
model = DefineModel(num_classes=len(classes)).to(DEVICE)
model.load_state_dict(torch.load(modelName, map_location=DEVICE))
model.eval()
print("Modell loaded. Starting evaluation...")

# Evaluation
all_labels, all_preds = [], []
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_labels.extend(labels.numpy())
        all_preds.extend(preds.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f"\nAccuracy: {accuracy*100:.2f}%")

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

# Confusion Matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap="Blues")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.title("Confusion Matrix")
plt.show()
print("Evaluation finished.")