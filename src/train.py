import os
import torch
from torch import nn, optim
from torchvision.models import resnet18, ResNet18_Weights
from dataset import create_dataloader
from sklearn.metrics import confusion_matrix, classification_report


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("useing device:", device)

    train_dir = "../data/train"
    val_dir = "../data/val"

    batch_size = 16
    image_size = 224

    train_loader, val_loader, classes_names = create_dataloader(
        train_dir = train_dir,
        val_dir = val_dir,
        batch_size= batch_size,
        img_size= image_size)

    num_classes = len(classes_names)
    print("number of classes: ", num_classes)
    print(classes_names)

    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    model= model.to(device)

    # loss + optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 5



    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 20)

        # ========TRAIN========

        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            preds = outputs.argmax(dim=1)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels).item()
            total += images.size(0)

        epoch_loss = running_loss / total
        epoch_acc = running_corrects / total
        print(f"Train loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # ========= validation ==============

        model.eval()
        val_loss = 0
        val_corrects = 0
        val_total = 0

        y_true = []
        y_pred = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                preds = outputs.argmax(dim=1)

                val_loss += loss.item() * images.size(0)
                val_corrects += torch.sum(preds == labels).item()
                val_total += images.size(0)

                y_true.extend(labels.cpu().tolist())
                y_pred.extend(preds.cpu().tolist())

        val_epoch_loss = val_loss / val_total
        val_epoch_acc = val_corrects / val_total
        print(f"val loss : {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")

        print("\nvalidation classification report:")
        print(classification_report(y_true,y_pred, target_names=classes_names))

    os.makedirs("../model", exist_ok=True)
    torch.save(model.state_dict(), "../model/resnet18_model.pth")
    print("model saved")


if __name__ == "__main__":
    main()

