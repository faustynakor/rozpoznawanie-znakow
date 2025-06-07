# rozpoznawanie-znakow

Struktura plików:

rozpoznawanie-znakow/
├── detection/
│   ├── images/
│   │   ├── train/       # Zbiór obrazów treningowych
│   │   └── test/        # Zbiór obrazów testowych
│   ├── labels/
│   │   ├── train/       # Etykiety treningowe 
│   │   └── test/        # Etykiety testowe
│   ├── classes.txt      # Lista klas znaków (po jednej klasie na wiersz)
│   └── video/
│       └── video_1.mp4  # Przykładowe wideo do detekcji
├── dataset.py           # Skrypt do tworzenia i ładowania datasetu
├── train.py             # Skrypt do trenowania sieci
├── test_model.py        # Testowanie jakości modelu
├── test_torch.py        # Testy kompatybilności PyTorch
├── detect_image.py      # Detekcja na wybranym obrazie
├── detect_video.py      # Detekcja na wideo
└── model.txt            # Plik linkiem do pobrania modelu (za duży żeby był na githubie) 