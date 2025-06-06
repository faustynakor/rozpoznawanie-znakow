# ten plik sprawdza jaka jest dostępna wersja pytorch i czy jest opcja CUDA, czyli czy korzystamy z karty graficznej
#byłoby szybciej, czy z procesora

import torch
print(torch.__version__)
print("CUDA dostępna:", torch.cuda.is_available())
