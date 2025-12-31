
import numpy as np
import matplotlib.pyplot as plt
from data_pipeline import load_splits, tokenize


def main():
    REMOVE_PARTS = ("headers", "footers", "quotes")
    SEED = 42
    VAL_SPLIT = 0.1

    CAP = 2000
    BINS = 50


    Xtr, Xv, Xt, ytr, yv, yt, target_names, num_classes = load_splits(
        REMOVE_PARTS, SEED, VAL_SPLIT
    )

    # make sure labels are simple int arrays
    ytr = np.asarray(ytr, dtype=np.int64)
    yv = np.asarray(yv, dtype=np.int64)
    yt = np.asarray(yt, dtype=np.int64)

    # counts per class id (0..num_classes-1)
    tr_counts = np.bincount(ytr, minlength=num_classes)
    va_counts = np.bincount(yv, minlength=num_classes)
    te_counts = np.bincount(yt, minlength=num_classes)

    print("Class distribution (train):")
    for i in range(num_classes):
        print(f"{i:2d} {tr_counts[i]:5d}  {target_names[i]}")

    labels = target_names
    x = np.arange(len(labels))

    # Train plot with class NAMES
    plt.figure(figsize=(14, 5))
    plt.bar(x, tr_counts)
    plt.title("Train class distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(x, labels, rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig("class_dist_train_names.png", dpi=160)

    # Combined Train/Val/Test with class NAMES
    plt.figure(figsize=(14, 5))
    w = 0.25
    plt.bar(x - w, tr_counts, width=w, label="train")
    plt.bar(x, va_counts, width=w, label="val")
    plt.bar(x + w, te_counts, width=w, label="test")
    plt.title("Class distribution (train/val/test)")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(x, labels, rotation=60, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig("class_dist_all_names.png", dpi=160)

    print("Saved: class_dist_train_names.png, class_dist_all_names.png")

    print("\nSaved plots: class_dist_train.png, class_dist_val.png, class_dist_test.png, class_dist_all.png")


    lengths = np.array([len(tokenize(t)) for t in Xtr], dtype=np.int32)

    mn = int(lengths.min())
    mx = int(lengths.max())
    avg = float(lengths.mean())

    print(f"Train docs: {len(Xtr)}")
    print(f"Train length stats: min={mn}, max={mx}, avg={avg:.2f}")
    print(f"Empty docs (len==0): {int((lengths==0).sum())}")


    ps = [50, 75, 90, 95, 99]
    pvals = [float(np.percentile(lengths, p)) for p in ps]
    for p, v in zip(ps, pvals):
        print(f"p{p}: {v:.0f} tokens")


    def round50(x): return int(50 * round(x / 50))
    print("\nSuggested MAX_LEN:")
    print("  Fast (~p90):", round50(pvals[2]))
    print("  Balanced (~p95):", round50(pvals[3]))
    print("  High coverage (~p99):", round50(pvals[4]))


    topk = np.argsort(-lengths)[:10]
    print("\nTop 10 longest (index, tokens):")
    for i in topk:
        print(int(i), int(lengths[i]))


    #bar: min/avg/max
    plt.figure()
    plt.bar(["min", "avg", "max"], [mn, avg, mx])
    plt.title("Train lengths: min / avg / max")
    plt.ylabel("Tokens")
    plt.tight_layout()
    plt.savefig("min_avg_max.png", dpi=160)

    #10 ranges (clipped)
    lengths_clip = np.clip(lengths, 0, CAP)
    counts, edges = np.histogram(lengths_clip, bins=10, range=(0, CAP))
    labels = [f"{int(edges[i])}-{int(edges[i+1])}" for i in range(len(edges)-1)]

    plt.figure(figsize=(10, 4))
    plt.bar(labels, counts)
    plt.title(f"Train length counts in 10 ranges (clipped at {CAP})")
    plt.xlabel("Token length range")
    plt.ylabel("Number of documents")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("ten_ranges_clipped.png", dpi=160)

    #histogram full
    plt.figure()
    plt.hist(lengths, bins=BINS)
    plt.title("Train length distribution (full)")
    plt.xlabel("Tokens")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("hist_full.png", dpi=160)

    #histogram clipped
    plt.figure()
    plt.hist(lengths_clip, bins=BINS)
    plt.title(f"Train length distribution (clipped at {CAP})")
    plt.xlabel("Tokens (clipped)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("hist_clipped.png", dpi=160)

    #percentile bars
    plt.figure()
    plt.bar([f"p{p}" for p in ps], pvals)
    plt.title("Train length percentiles")
    plt.ylabel("Tokens")
    plt.tight_layout()
    plt.savefig("percentiles.png", dpi=160)

    print("\nSaved plots:")
    print("  min_avg_max.png")
    print("  ten_ranges_clipped.png")
    print("  hist_full.png")
    print("  hist_clipped.png")
    print("  percentiles.png")






if __name__ == "__main__":
    main()
