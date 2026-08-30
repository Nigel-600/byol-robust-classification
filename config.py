import torch
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
NUM_WORKERS = 4
CLASS_NAMES = ["BAD", "GOOD"]