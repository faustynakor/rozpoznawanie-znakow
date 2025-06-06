import cv2
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from dataset import load_class_map, get_model
import matplotlib.pyplot as plt


def detect_from_image(image_path, model_path, class_map_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Wczytanie klasyfikacji
    label_map = load_class_map(class_map_path)
    rev_label_map = {v: k for k, v in label_map.items()}

    # Wczytanie modelu
    model = get_model(len(label_map) + 1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)

    # Wczytanie obrazu
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # Predykcja
    with torch.no_grad():
        prediction = model(img_tensor)[0]

    for box, label, score in zip(prediction["boxes"], prediction["labels"], prediction["scores"]):
        if score > 0.5:
            x1, y1, x2, y2 = map(int, box)
            class_name = rev_label_map.get(label.item(), "???")
            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_bgr, f"{class_name} ({score:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Pokaż obraz z predykcjami
    cv2.imshow("Detected Signs", image_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Podmień na rzeczywistą ścieżkę do jednego z testowych obrazów
    detect_from_image(
        image_path="images/test/Screenshot_27.png",
        model_path="model_epoch_10.pth",
        class_map_path="labels/classes.txt"
    )
