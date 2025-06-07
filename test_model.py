import os
import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from dataset import TrafficSignDataset
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def load_class_map(path):
    with open(path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return {cls: idx+1 for idx, cls in enumerate(classes)}

def get_model(num_classes, model_path):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    return model

def evaluate_model(model, data_loader, device, iou_threshold=0.5, score_threshold=0.5):
    model.eval()
    all_true_positives = 0
    all_false_positives = 0
    all_false_negatives = 0
    all_true_negatives = 0  # dodane, ale raczej nieużywane
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for output, target in zip(outputs, targets):
                pred_boxes = output['boxes'][output['scores'] > score_threshold]
                pred_labels = output['labels'][output['scores'] > score_threshold]

                true_boxes = target['boxes'].to(device)
                true_labels = target['labels'].to(device)

                if len(pred_boxes) == 0:
                    all_false_negatives += len(true_boxes)
                    continue

                ious = box_iou(pred_boxes, true_boxes)

                tp = 0
                matched_true = set()
                for pred_idx in range(ious.size(0)):
                    candidate_true_indices = [i for i in range(ious.size(1))
                                              if ious[pred_idx, i] > iou_threshold and pred_labels[pred_idx] == true_labels[i]]
                    if candidate_true_indices:
                        for true_idx in candidate_true_indices:
                            if true_idx not in matched_true:
                                matched_true.add(true_idx)
                                tp += 1
                                break

                fp = len(pred_boxes) - tp
                fn = len(true_boxes) - len(matched_true)

                all_true_positives += tp
                all_false_positives += fp
                all_false_negatives += fn

    precision = all_true_positives / (all_true_positives + all_false_positives + 1e-6)
    recall = all_true_positives / (all_true_positives + all_false_negatives + 1e-6)
    return precision, recall , all_true_positives, all_false_positives, all_false_negatives

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_map = load_class_map("detection/labels/classes.txt")
    num_classes = len(label_map) + 1

    model_path = "model_epoch_10.pth"

    dataset_test = TrafficSignDataset(
        image_dir="detection/images/test",
        label_dir="detection/labels/test",
        class_map=label_map
    )

    data_loader_test = DataLoader(dataset_test, batch_size=4, shuffle=False, collate_fn=collate_fn)

    model = get_model(num_classes, model_path)
    model.to(device)

    precision, recall, tp, fp, fn = evaluate_model(model, data_loader_test, device)

    os.makedirs("./evaluate_model", exist_ok=True)
    with open("./evaluate_model/evaluate.txt", "w") as f:
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"TP: {tp:.4f}\n")
        f.write(f"FP: {fp:.4f}\n")
        f.write(f"FN: {fn:.4f}\n")

    print("Evaluation complete. Results saved to ./evaluate_model/evaluate.txt")

if __name__ == "__main__":
    main()
