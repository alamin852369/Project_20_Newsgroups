# config.py
import torch

SEED = 42

GLOVE_PATH = "/home/alamin/Downloads/glove.6B/glove.6B.300d.txt"
EMBED_DIM = 300

HIDDEN_DIM = 128
DROPOUT = 0.5
FREEZE_EMBEDDINGS = False

BATCH_SIZE = 64
EPOCHS = 50
LR = 0.001
WEIGHT_DECAY = 0.0
VAL_SPLIT = 0.10

MAX_VOCAB = 30000
MIN_FREQ = 2
MAX_LEN = 400

HISTORY_CSV = "history_train_val.csv"
TEST_CSV = "test_metrics.csv"
CONFUSION_CSV = "confusion_matrix_test.csv"
BEST_MODEL_PATH = "best_model_by_val_loss.pt"

REMOVE_PARTS = ("headers", "footers", "quotes")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
