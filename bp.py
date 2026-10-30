from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = ImageDraw = None

import mnist
import nn
from nn import Array


def build_model(n_in: int = 784, n_hidden: int = 256, n_out: int = 10,
                rng=None) -> nn.Sequential:
    if rng is None:
        rng = np.random.default_rng()

    return nn.Sequential(nn.Linear(n_in, n_hidden,init = "he",rng = rng),
                         nn.ReLU(),
                         nn.Linear(n_hidden, n_out,init = "xavier",rng = rng),
                         )


def train_step(model: nn.Sequential, opt: nn.Adam, x: Array, y: Array) -> dict[str, float]:
    #前向
    logits = model.forward(x)

    #损失+上游梯度
    loss,dlogits = nn.softmax_cross_entropy(logits, y)

    #清零
    model.zero_grad()

    #反向
    model.backward(dlogits)

    #原地更新
    opt.step(model.grads())

    #日志
    acc = float((logits.argmax(axis= 1)==y).mean())

    return {"loss": float(loss), "acc": acc}

def evaluate(model: nn.Sequential, x: Array, y: Array,
             batch: int = 1000) -> dict[str, float]:
    N = len(x)
    total_loss = 0.0
    total_correct = 0

    for i in range(0,N,batch):
        xb = x[i:i+batch]
        yb = y[i:i+batch]

        logits = model.forward(xb)
        loss,_ = nn.softmax_cross_entropy(logits, yb)

        total_loss += loss*len(xb)
        total_correct += (logits.argmax(axis=1) == yb).sum()

    return {"loss": float(total_loss)/N, "acc": float(total_correct)/N}

def confusion_matrix(y_true: Array, y_pred: Array, n_class: int = 10) -> Array:
    idx = y_true * n_class + y_pred
    return np.bincount(idx, minlength=n_class * n_class).reshape((n_class, n_class))

def predict(model: nn.Sequential, x: Array, batch: int = 1000) -> Array:
    out = []
    for start in range(0, len(x), batch):
        logits = model.forward(x[start:start + batch])
        out.append(logits.argmax(axis=1))
    return np.concatenate(out)

def train_epoch(model: nn.Sequential, opt: nn.Adam, x: Array, y: Array,
                batch: int, rng: np.random.Generator,
                log_every: int = 0) -> dict[str, float]:
    order = rng.permutation(len(x))
    logs: dict[str, list[float]] = defaultdict(list)

    n_step = 0
    for start in range(0, len(x) - batch + 1, batch):
        idx = order[start:start + batch]
        stats = train_step(model, opt, x[idx], y[idx])
        n_step += 1
        for k, v in stats.items():
            logs[k].append(float(v))
        if log_every and n_step % log_every == 0:
            print(f"      step {n_step:4d}  loss {logs['loss'][-1]:.4f}"
                  f"  acc {logs['acc'][-1]:.4f}")

    return {k: float(np.mean(v)) for k, v in logs.items()}


def save_curves(history: list[dict], path: str) -> None:
    if plt is None or not history:
        return
    ep = np.arange(1, len(history) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(ep, [h["train_loss"] for h in history], label="train")
    axes[0].plot(ep, [h["test_loss"] for h in history], label="test")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("cross entropy")
    axes[0].legend(); axes[0].set_title("loss")
    axes[1].plot(ep, [h["train_acc"] for h in history], label="train")
    axes[1].plot(ep, [h["test_acc"] for h in history], label="test")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy")
    axes[1].legend(); axes[1].set_title("accuracy")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_confusion(conf: Array, path: str) -> None:
    if plt is None:
        np.savetxt(path + ".txt", conf, fmt="%d")
        return
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(conf, cmap="Blues")
    n = conf.shape[0]
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title("confusion matrix (test set)")
    thresh = conf.max() / 2.0
    for i in range(n):
        for j in range(n):
            if conf[i, j]:
                ax.text(j, i, str(int(conf[i, j])), ha="center", va="center",
                        fontsize=7, color="white" if conf[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_misclassified(model: nn.Sequential, x: Array, y: Array, path: str,
                       n: int = 24, nrow: int = 6) -> None:
    if Image is None:
        print("  [skip] 未安装 Pillow（pip install Pillow），跳过错例图生成")
        return

    pred = predict(model, x)
    wrong = np.flatnonzero(pred != y)[:n]
    if len(wrong) == 0:
        print("  （测试集上全部预测正确，没有错例可画）")
        return

    cell, pad, text_h = 28, 2, 9
    h_cell, w_cell = cell + text_h + pad, cell + pad
    ncol = int(np.ceil(len(wrong) / nrow))
    canvas = np.full((ncol * h_cell + pad, nrow * w_cell + pad), 255, dtype=np.uint8)
    img = Image.fromarray(canvas).convert("RGB")
    draw = ImageDraw.Draw(img)

    for k, i in enumerate(wrong):
        r, c = divmod(k, nrow)
        y0, x0 = pad + r * h_cell, pad + c * w_cell
        tile = mnist.to_image(x[i])
        img.paste(Image.fromarray(tile).convert("RGB"), (x0, y0))
        draw.text((x0, y0 + cell), f"{int(y[i])}->{int(pred[i])}", fill=(200, 0, 0))

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    img.save(path)


def save_checkpoint(path: str, model: nn.Sequential, meta: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {k: v for k, v in model.state().items()}
    payload["meta"] = np.array([str(meta)], dtype=object)
    np.savez(path, **payload)


def load_checkpoint(path: str, model: nn.Sequential) -> dict:
    with np.load(path, allow_pickle=True) as f:
        keys = list(f.files)
        model.load_state({k: f[k] for k in keys if k != "meta"})
        meta = eval(str(f["meta"][0])) if "meta" in keys else {}
    return meta


def train(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)

    x_train, y_train, x_test, y_test = mnist.load_mnist(args.data, scale=args.scale)
    if args.limit:
        x_train, y_train = x_train[:args.limit], y_train[:args.limit]
    print(f"train {x_train.shape}  test {x_test.shape}  像素范围 "
          f"[{x_train.min():.2f}, {x_train.max():.2f}]  类别数 {len(np.unique(y_train))}")

    model = build_model(x_train.shape[1], args.hidden, 10, rng=rng)
    opt = nn.Adam(model.params(), lr=args.lr, beta1=args.beta1)
    print(f"参数总量 {model.num_params():,}   batch {args.batch}   lr {args.lr}")

    history: list[dict] = []
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = train_epoch(model, opt, x_train, y_train, args.batch, rng,
                         log_every=args.log_every)
        te = evaluate(model, x_test, y_test)
        dt = time.time() - t0

        history.append({"train_loss": tr["loss"], "train_acc": tr["acc"],
                        "test_loss": te["loss"], "test_acc": te["acc"]})
        print(f"epoch {epoch:3d}/{args.epochs}  {dt:5.2f}s  "
              f"train loss {tr['loss']:.4f} acc {tr['acc']:.4f}  |  "
              f"test loss {te['loss']:.4f} acc {te['acc']:.4f}")

        if te["acc"] > best_acc:
            best_acc = te["acc"]
            save_checkpoint(os.path.join(args.out, "best.npz"), model,
                            {"epoch": epoch, "test_acc": te["acc"]})

        if len(history) >= 3:
            gap = history[-1]["train_acc"] - history[-1]["test_acc"]
            if gap > 0.02:
                print(f"      [!] 训练/测试准确率差 {gap:.3f}：开始过拟合了。"
                      " → 早停 / 加 L2 / 加 Dropout / 减小隐藏层")
            if te["acc"] < 0.5 and epoch >= 3:
                print("      [!] 准确率还在 50% 以下：回去跑 check_grad.py，"
                      "多半是反向传播写错了")

    save_curves(history, os.path.join(args.out, "curves.png"))
    conf = confusion_matrix(y_test, predict(model, x_test))
    save_confusion(conf, os.path.join(args.out, "confusion.png"))
    save_misclassified(model, x_test, y_test, os.path.join(args.out, "misclassified.png"))

    per_class = conf.diagonal() / np.maximum(conf.sum(axis=1), 1)
    print("\n各类准确率：" + "  ".join(f"{i}:{a:.3f}" for i, a in enumerate(per_class)))
    worst = int(np.argmin(per_class))
    print(f"最差的类：{worst}（{per_class[worst]:.3f}）。最常见的混淆：", end="")
    off = conf.copy(); np.fill_diagonal(off, 0)
    for _ in range(3):
        i, j = np.unravel_index(np.argmax(off), off.shape)
        if off[i, j] == 0:
            break
        print(f"{i}→{j}({off[i, j]}) ", end="")
        off[i, j] = 0
    print(f"\n最佳测试准确率 {best_acc:.4f}，输出在 {os.path.abspath(args.out)}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="纯 NumPy BP 神经网络 on MNIST")
    p.add_argument("--data", default="data/mnist")
    p.add_argument("--out", default="runs/bp")
    p.add_argument("--scale", default="01", choices=["01", "tanh"],
                   help="输入像素归一化范围；'01' 是分类任务常规做法")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="只用前 N 个训练样本（冒烟测试用）")
    p.add_argument("--log-every", type=int, default=100)
    return p


if __name__ == "__main__":
    train(build_argparser().parse_args())
