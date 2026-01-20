import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO

# Gerät wählen
DEVICE = torch.device("cpu")

# Klassen
classes = ['Mild', 'Moderate', 'Non', 'VeryMild']

# Eigenes Modell (identisch zum Training)
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


model = DefineModel(num_classes=4).to(DEVICE)
model.load_state_dict(torch.load("basti_mri_model_e15.pth", weights_only=True))
model.eval()
print("Model loaded")

# Transformation
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Vorhersagefunktion
def predict(image):
    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    # Vorhersage
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred_idx = np.argmax(probs)
        pred_class = classes[pred_idx]

    # Balkendiagramm erstellen
    fig, ax = plt.subplots(figsize=(5,3))
    ax.bar(classes, probs*100, color="skyblue")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Wahrscheinlichkeit (%)")
    ax.set_title(f"Vorhersage: {pred_class}")
    
    fig.tight_layout()
    # In PIL Image umwandeln
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    barchart_image = Image.open(buf).convert("RGB")
    buf.close()
    plt.close(fig)
    
    return image, barchart_image

# Gradio Interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Image(type="pil", label="Eingabebild"), 
             gr.Image(type="pil", label="Vorhersage Wahrscheinlichkeiten")],
    title="MRT Klassifikation",
    description="Lade ein MRT-Bild hoch. Das Modell zeigt die Vorhersage und die Wahrscheinlichkeiten."
)

iface.launch()
print("Gradio Interface started")