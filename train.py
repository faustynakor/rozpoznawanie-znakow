#ten plik uczy model i zapisuje go jako model_ w formacie .pth

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
from dataset import TrafficSignDataset
import os

def load_class_map(path):
    with open(path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return {cls: idx+1 for idx, cls in enumerate(classes)}  # +1, bo 0 to background

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def collate_fn(batch):
    return tuple(zip(*batch))

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_map = load_class_map("detecion/labels/classes.txt")

    dataset = TrafficSignDataset(
        image_dir="detecion/images/train",
        label_dir="detecion/labels/train",
        class_map=label_map
    )

    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    model = get_model(num_classes=len(label_map) + 1)
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}", leave=True)
        for i, (images, targets) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch+1}")):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            total_loss += losses.item()
            progress_bar.set_postfix(loss=losses.item())
            
        print(f"Epoch {epoch+1} - Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")


if __name__ == "__main__":
    train()
