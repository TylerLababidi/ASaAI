import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Gerät
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Klassen
classes = ['Mild', 'Moderate', 'Non', 'VeryMild']

# Eigenes Modell (muss identisch zum Training sein)
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

# Modell laden
model = MyMRTModel(num_classes=4).to(DEVICE)
model.load_state_dict(torch.load("my_mrt_model.pth", weights_only=True))
model.eval()
print("Modell geladen ✅")

# Transformation
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

def predict(image_path):
    # Bild laden
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Vorhersage
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred_idx = np.argmax(probs)
        pred_class = classes[pred_idx]

    # Ausgabe
    print(f"Bild: {image_path}")
    print(f"Vorhersage: {pred_class}")
    print("Wahrscheinlichkeiten:")
    for cls, p in zip(classes, probs):
        print(f"  {cls}: {p*100:.2f}%")

    # Bild mit Vorhersage anzeigen
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Vorhersage: {pred_class}")
    plt.show()

    return pred_class, probs

# Kommandozeile
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Bitte Bildpfad angeben, z.B.: python predict.py 'Data/test/1 (10).jpg'")
    else:
        predict(sys.argv[1])
