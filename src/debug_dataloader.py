from dataset import create_dataloader

train_loader, val_loader, classes_names = create_dataloader(
    train_dir="../data/train",
    val_dir="../data/val",
    batch_size=16,
    img_size=224,
)
print("Classes names:", classes_names)
for images, labels in train_loader:
    print(images.shape, labels.shape)
    break