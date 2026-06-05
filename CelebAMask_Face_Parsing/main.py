import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import pickle
import time
import argparse
import os
import cv2
import pickle
from PIL import Image
from mobilenet_unet import MobileNet_UNet  # Import the model
from mobilenet_unet_cbam import MobileNet_UNet_CBAM
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts

# from accelerate import Accelerator

def compute_multiclass_fscore(mask_gt, mask_pred, beta=1):
    f_scores = []

    for class_id in np.unique(mask_gt):
        tp = np.sum((mask_gt == class_id) & (mask_pred == class_id))
        fp = np.sum((mask_gt != class_id) & (mask_pred == class_id))
        fn = np.sum((mask_gt == class_id) & (mask_pred != class_id))

        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f_score = (
            (1 + beta**2)
            * (precision * recall)
            / ((beta**2 * precision) + recall + 1e-7)
        )

        f_scores.append(f_score)

    return np.mean(f_scores)

class FaceParsingDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_filenames = sorted(os.listdir(img_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))
        self.palette = Image.open(os.path.join(mask_dir, self.mask_filenames[0])).getpalette()

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (512, 512))

        mask = Image.open(mask_path).convert("P")  # Load palette mask
        mask = mask.resize((512, 512), Image.NEAREST)
        mask = np.array(mask, dtype=np.int64)
        mask = torch.tensor(mask, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, mask

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Compute training accuracy
        preds = torch.argmax(outputs, dim=1)
        correct_train += (preds == masks).sum().item()
        total_train += masks.numel()
    
    train_acc = correct_train / total_train  # Training accuracy

    model.eval()
    val_loss = 0.0
    val_fscore = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            loss = criterion(outputs, masks)
            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1).cpu()
            masks = masks.cpu()
            correct_val += (preds == masks).sum().item()
            total_val += masks.numel()

            val_fscore += compute_multiclass_fscore(masks.numpy(), preds.numpy())

    val_acc = correct_val / total_val  # Validation accuracy
    scheduler.step(val_loss)

    return running_loss, train_acc, val_loss, val_acc, val_fscore


# Function to add Gaussian noise
def add_gaussian_noise(img, mean=0, std=0.1):
    noise = torch.randn_like(img) * std + mean  # Generate noise
    return torch.clamp(img + noise, 0, 1)  # Ensure pixel values remain in valid range

def main():

    torch.cuda.empty_cache()
    start_time = time.time()  # Start tracking time

    parser = argparse.ArgumentParser(description="Train MobileNet on CIFAR-100")
    parser.add_argument('--lr', type=float, default=0.2, help='Initial learning rate')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--weight_decay', type=float, default=None, help='Weight Decay Coefficient')

    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    data_root = "/CelebAMask_Face_Parsing/dev-public/train"
    train_images = os.path.join(data_root, "images")
    train_masks = os.path.join(data_root, "masks")
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = FaceParsingDataset(train_images, train_masks, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = MobileNet_UNet_CBAM(num_classes=19).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, threshold=1e-4, min_lr=1e-6)

    log_output = []  # List to store logs
    log_output.append(f"Total trainable parameters: {sum(p.numel() for p in model.parameters())}\n")
    log_output.append(f"Learning Rate: {args.lr}, Epochs: {args.epochs}, Batch Size: {args.batch_size}, Weight decay: {args.weight_decay}n")
    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters())}")
    print("Learning Rate:", args.lr, ", Epochs: ", args.epochs, ", Batch Size: ", args.batch_size, ", Weight decay: ", args.weight_decay)
    print(model)
    
    model, optimizer, train_loader, val_loader = (
        model, optimizer, train_loader, val_loader
    )

    torch.cuda.empty_cache()

    for epoch in range(args.epochs):
        running_loss, train_acc, val_loss, val_acc, val_fscore = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device)
        print(f"Epoch [{epoch+1}/{args.epochs}], Training Loss: {running_loss/len(train_loader):.4f}, Training Accuracy: {train_acc:.4f},"
            f"Val Loss: {val_loss/len(val_loader):.4f}, Val Accuracy: {val_acc:.4f}, Val F-score: {val_fscore/len(val_loader):.4f}")
        
        log_entry = f"Epoch [{epoch+1}/{args.epochs}], Training Loss: {running_loss/len(train_loader):.4f}, Training Accuracy: {train_acc:.4f},Val Loss: {val_loss/len(val_loader):.4f}, Val Accuracy: {val_acc:.4f}, Val F-score: {val_fscore/len(val_loader):.4f}"
        log_output.append(log_entry)

        del running_loss, val_fscore, train_acc, val_loss, val_acc
        torch.cuda.empty_cache()
    
    torch.save({"state_dict": model.state_dict()}, f'./models/mobilenet_unet_epochs_{args.epochs}_lr{args.lr}_wd{args.weight_decay}_bs{args.batch_size}_leakyrelu_cbam_plateau_upsample_labelsmoothing_newnormalize_dilatedconv.pth')
    log_output.append("Training complete, model saved!")

    del model, criterion, optimizer, scheduler
    torch.cuda.empty_cache()

    end_time = time.time()  # End tracking time
    total_time = end_time - start_time  # Calculate elapsed time
    print(f"\nTotal Training Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    log_entry = f"\nTotal Training Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)"
    log_output.append(log_entry)

    with open(f"training_logs_epochs{args.epochs}_lr{args.lr}_wd{args.weight_decay}_bs{args.batch_size}_leakyrelu_cbam_plateau_upsample_labelsmoothing_newnormalize_dilatedconv", "wb") as f:
        pickle.dump("\n".join(log_output), f)
    
if __name__ == "__main__":
    main()