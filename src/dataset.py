import os
from torchvision import datasets, transforms
from torch.utils.data import  DataLoader

def create_dataloader(train_dir, val_dir, batch_size =32, img_size =224):



     train_transforms = transforms.Compose([

            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


     val_transforms = transforms.Compose([

            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
        )
    ])

     train_data = datasets.ImageFolder(root=train_dir, transform=train_transforms)
     val_data = datasets.ImageFolder(root=val_dir, transform=val_transforms)

     train_loader = DataLoader(train_data,batch_size=batch_size, shuffle=True, num_workers=4)
     val_loader = DataLoader(val_data,batch_size=batch_size, shuffle=False, num_workers=4)

     classes_names = train_data.classes

     return train_loader, val_loader, classes_names


