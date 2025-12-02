import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 讀取 CSV
df = pd.read_csv("results.csv")

# 共同參數：字體大小
TITLE_FS = 50
LABEL_FS = 45
TICK_FS  = 35
LEGEND_FS = 25

def plot_curve(x, ys, labels, ylabel, title, save_name,
               ylim=None, loc="upper right"):
    """繪製通用曲線圖"""
    plt.figure(figsize=(60,12), dpi=300)
    for y, lab in zip(ys, labels):
        plt.plot(df[x], df[y], label=lab)

    if ylim:
        plt.ylim(*ylim)

    plt.xlabel("Epoch", fontsize=LABEL_FS)
    plt.ylabel(ylabel, fontsize=LABEL_FS)
    plt.title(title, fontsize=TITLE_FS)
    plt.legend(fontsize=LEGEND_FS, loc=loc)
    plt.grid(True)

    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.tick_params(axis="x", labelsize=TICK_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.close()

# --------- Loss 曲線 ---------
plot_curve(
    x="epoch",
    ys=["train/box_loss","train/cls_loss","train/dfl_loss",
        "val/box_loss","val/cls_loss","val/dfl_loss"],
    labels=["Train Box Loss","Train Cls Loss","Train DFL Loss",
            "Val Box Loss","Val Cls Loss","Val DFL Loss"],
    ylabel="Loss",
    title="Training & Validation Loss",
    save_name="loss_curve.png",
    ylim=(0.25, 2),
    loc="upper right"
)

# --------- Metrics 曲線 ---------
plot_curve(
    x="epoch",
    ys=["metrics/precision(B)","metrics/recall(B)",
        "metrics/mAP50(B)","metrics/mAP50-95(B)"],
    labels=["Precision","Recall","mAP@50","mAP@50-95"],
    ylabel="Value",
    title="Validation Metrics",
    save_name="metrics_curve.png",
    loc="upper right"
)

# --------- Learning Rate ---------
plot_curve(
    x="epoch",
    ys=["lr/pg0","lr/pg1","lr/pg2"],
    labels=["lr/pg0","lr/pg1","lr/pg2"],
    ylabel="Learning Rate",
    title="Learning Rate Schedule",
    save_name="lr_curve.png",
    loc="upper right"
)