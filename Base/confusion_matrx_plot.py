import numpy as np
import matplotlib.pyplot as plt

def plot_cm(cm, labels=None, normalize=False, out_name="cm.png", title="Confusion Matrix"):
    cm = cm.astype(float)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm_plot = cm / row_sums
    else:
        cm_plot = cm

    n = cm.shape[0]
    plt.figure(figsize=(10, 8))
    plt.imshow(cm_plot, interpolation="nearest")
    plt.title(title + (" (Normalized)" if normalize else ""))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()

    ticks = np.arange(n)
    if labels is None:
        plt.xticks(ticks)
        plt.yticks(ticks)
    else:
        plt.xticks(ticks, labels, rotation=60, ha="right")
        plt.yticks(ticks, labels)

    # annotate
    for i in range(n):
        for j in range(n):
            v = cm_plot[i, j]
            text = f"{v:.2f}" if normalize else f"{int(round(v))}"
            plt.text(j, i, text, ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_name, dpi=200, bbox_inches="tight")  # ✅ saves figure
    plt.close()  # ✅ frees memory

# ---- Example usage ----
cm = np.loadtxt("confusion_matrix_test.csv", delimiter=",")   # change path if needed
plot_cm(cm, normalize=False, out_name="cm_raw.png")
plot_cm(cm, normalize=True,  out_name="cm_norm.png")

print("Saved: cm_raw.png, cm_norm.png")
