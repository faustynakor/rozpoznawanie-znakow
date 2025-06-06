import os
import json
import torch
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import Dataset

class TrafficSignDataset(Dataset):
    def __init__(self, image_dir, label_dir, class_map, transforms=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.class_map = class_map
        self.image_files = sorted(os.listdir(image_dir))

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + '.json')

        # Wymuszenie RGB i konwersja do tensora
        img = Image.open(img_path).convert("RGB")
        image = transforms.ToTensor()(img)  # ToTensor dzieli przez 255 automatycznie

        with open(label_path, 'r') as f:
            data = json.load(f)[0]
            annotations = data["annotations"]

        boxes = []
        labels = []

        for ann in annotations:
            label = ann["label"]
            coords = ann["coordinates"]
            x1 = coords["x"] - coords["width"] / 2
            y1 = coords["y"] - coords["height"] / 2
            x2 = coords["x"] + coords["width"] / 2
            y2 = coords["y"] + coords["height"] / 2

            boxes.append([x1, y1, x2, y2])
            labels.append(self.class_map[label])

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }

        return image, target

    def __len__(self):
        return len(self.image_files)
    
def load_class_map(filepath):
    with open(filepath, "r") as f:
        lines = f.read().splitlines()
    return {label: idx + 1 for idx, label in enumerate(lines)}

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model