# plot_history.py
import csv
import matplotlib.pyplot as plt

CSV_PATH = "history_train_val.csv"   # change if needed

def read_history(csv_path):
    epochs, tr_loss, tr_acc, va_loss, va_acc = [], [], [], [], []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            tr_loss.append(float(row["train_loss"]))
            tr_acc.append(float(row["train_acc"]))
            va_loss.append(float(row["val_loss"]))
            va_acc.append(float(row["val_acc"]))

    return epochs, tr_loss, tr_acc, va_loss, va_acc

def main():
    epochs, tr_loss, tr_acc, va_loss, va_acc = read_history(CSV_PATH)

    # ---- Loss plot ----
    plt.figure()
    plt.plot(epochs, tr_loss, label="train_loss")
    plt.plot(epochs, va_loss, label="val_loss")
    plt.title("Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=160)

    # ---- Accuracy plot ----
    plt.figure()
    plt.plot(epochs, tr_acc, label="train_acc")
    plt.plot(epochs, va_acc, label="val_acc")
    plt.title("Accuracy vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("acc_curve.png", dpi=160)

    print("Saved: loss_curve.png, acc_curve.png")





if __name__ == "__main__":
    main()
