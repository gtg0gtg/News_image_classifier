import sys
from pathlib import Path
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

IMG_SIZE = 224

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225)),
])

Class_names = ['disaster','economy','health','politics','sports']


def load_model(model_path: str, num_classes: int = 5) -> nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device: ", device)

    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model, device

def predict_image(model: nn.Module, device, image_path:str):
    img = Image.open(image_path).convert('RGB')

    tensor = val_transform(img)
    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs =torch.softmax(outputs, dim=1)
        top_prob, top_idx = torch.max(probs, dim=1)

    class_idx = top_idx.item()

    class_name = Class_names[class_idx]
    confidence = top_prob.item()

    return class_name, confidence

def main():
    if len(sys.argv) < 2:
        print("usage: python3 predict.py /paht/to/image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    if not Path(image_path).exists():
        print("image does not exist")
        sys.exit(1)

    model_path ="../model/resnet18_model.pth"
    model, device = load_model(model_path, num_classes=len(Class_names))

    class_name, confidence = predict_image(model, device, image_path)

    print(f"predicted class: {class_name}")
    print(f"confidence: {confidence:.4f}")

if __name__ == "__main__":
    main()