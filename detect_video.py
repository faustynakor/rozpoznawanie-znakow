import cv2
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from dataset import load_class_map, get_model
import os

def detect_from_video(video_path, model_path, class_map_path, output_path="output/result.avi"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_map = load_class_map(class_map_path)
    rev_label_map = {v: k for k, v in label_map.items()}

    model = get_model(len(label_map) + 1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Błąd przy otwieraniu pliku wideo.")
        return

    # Pobranie rozmiaru i FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Możesz też użyć 'MP4V' dla .mp4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = model(img_tensor)[0]

        for box, label, score in zip(prediction["boxes"], prediction["labels"], prediction["scores"]):
            if score > 0.7:
                x1, y1, x2, y2 = box.int().tolist()
                label_text = rev_label_map[label.item()]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"{label_text} ({score:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        out.write(frame)  # Zapisz klatkę do pliku wyjściowego
        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_from_video("video/video_Torun.mp4", "model_epoch_10.pth", "detection/labels/classes.txt", output_path="output/result.mp4")
