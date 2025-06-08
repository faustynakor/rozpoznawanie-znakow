import os
import json
from PIL import Image

# 📁 Ścieżki
txt_folder = "detection/labels/train"
image_folder = "detection/images/train"

# 📋 Twoja lista labeli
labels = [
    "A-1", "A-11", "A-11a", "A-12a", "A-14", "A-15", "A-16", "A-17", "A-18b", "A-2",
    "A-20", "A-21", "A-24", "A-29", "A-3", "A-30", "A-32", "A-4", "A-6a", "A-6b",
    "A-6c", "A-6d", "A-6e", "A-7", "A-8", "B-1", "B-18", "B-2", "B-20", "B-21",
    "B-22", "B-25", "B-26", "B-27", "B-33", "B-34", "B-36", "B-41", "B-42", "B-43",
    "B-44", "B-5", "B-6-B-8-B-9", "B-8", "B-9", "C-10", "C-12", "C-13", "C-13-C-16",
    "C-13a", "C-13a-C-16a", "C-16", "C-2", "C-4", "C-5", "C-6", "C-7", "C-9", "D-1",
    "D-14", "D-15", "D-18", "D-18b", "D-2", "D-21", "D-23", "D-23a", "D-24", "D-26",
    "D-26b", "D-26c", "D-27", "D-28", "D-29", "D-3", "D-40", "D-41", "D-42", "D-43",
    "D-4a", "D-4b", "D-51", "D-52", "D-53", "D-6", "D-6b", "D-7", "D-8", "D-9",
    "D-tablica", "G-1a", "G-3"
]

index_to_label = {idx: label for idx, label in enumerate(labels)}

def convert_coords_back(x_center, y_center, w, h, image_width, image_height):
    x = x_center * image_width
    y = y_center * image_height
    width = w * image_width
    height = h * image_height
    return x, y, width, height

for filename in os.listdir(txt_folder):
    if filename.startswith("Screenshot") and filename.endswith(".txt"):
        txt_path = os.path.join(txt_folder, filename)

        # Szukamy obrazka
        image_filename = filename.replace(".txt", ".png")
        image_path = os.path.join(image_folder, image_filename)

        if not os.path.exists(image_path):
            print(f"⚠️  Brak obrazu dla {filename}, pomijam.")
            continue

        # Pobierz wymiary obrazu
        with Image.open(image_path) as img:
            image_width, image_height = img.size

        annotations = []

        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                class_id = int(parts[0])
                x_center, y_center, w, h = map(float, parts[1:])

                label = index_to_label.get(class_id, "A-1")  # domyślny label
                x, y, width, height = convert_coords_back(x_center, y_center, w, h, image_width, image_height)

                annotation = {
                    "label": label,
                    "coordinates": {
                        "x": x,
                        "y": y,
                        "width": width,
                        "height": height
                    }
                }
                annotations.append(annotation)

        json_data = [{"annotations": annotations}]

        json_filename = filename.replace(".txt", ".json")
        json_path = os.path.join(txt_folder, json_filename)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=4)

print("✅ Konwersja z .txt do .json zakończona.")
