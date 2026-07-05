"""
配置参数
"""

import os
import torch

BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 20
NUM_WORKERS = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
LOG_DIR = os.path.join(ROOT_DIR, "logs")
RESULT_DIR = os.path.join(ROOT_DIR, "results")
EVAL_IMAGE_DIR = os.path.join(DATA_DIR, "eval_images")

MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "mnist_model.pth")
EVAL_RESULT_PATH = os.path.join(RESULT_DIR, "evaluation_results.txt")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(EVAL_IMAGE_DIR, exist_ok=True)
