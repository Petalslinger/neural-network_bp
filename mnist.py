from __future__ import annotations

import gzip
import os
import struct
import urllib.request

import numpy as np

MIRRORS = (
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
)

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _download_one(url: str, dest: str) -> None:
    tmp = dest + ".part"
    with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as fh:
        total = int(resp.headers.get("content-length") or 0)
        got = 0
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            fh.write(chunk)
            got += len(chunk)
            if total:
                print(f"\r  {os.path.basename(dest)}: {100.0 * got / total:5.1f}%",
                      end="", flush=True)
    print()
    os.replace(tmp, dest)


def download(root: str = "data/mnist") -> None:
    os.makedirs(root, exist_ok=True)
    for name in FILES.values():
        dest = os.path.join(root, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f"  [cache] {name}")
            continue
        last_err: Exception | None = None
        for base in MIRRORS:
            try:
                print(f"  [get  ] {base}{name}")
                _download_one(base + name, dest)
                last_err = None
                break
            except Exception as exc:
                last_err = exc
                print(f"  [fail ] {type(exc).__name__}: {exc}")
        if last_err is not None:
            raise RuntimeError(f"无法下载 {name}，请手动放到 {root}/") from last_err


def _read_idx(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as fh:
        blob = fh.read()
    magic, = struct.unpack(">I", blob[:4])
    ndim = magic & 0xFF
    dims = struct.unpack(f">{ndim}I", blob[4:4 + 4 * ndim])
    data = np.frombuffer(blob, dtype=np.uint8, offset=4 + 4 * ndim)
    return data.reshape(dims)


def load_mnist(root: str = "data/mnist", scale: str = "01"):
    download(root)
    x_tr = _read_idx(os.path.join(root, FILES["train_images"]))
    y_tr = _read_idx(os.path.join(root, FILES["train_labels"]))
    x_te = _read_idx(os.path.join(root, FILES["test_images"]))
    y_te = _read_idx(os.path.join(root, FILES["test_labels"]))

    if scale in ("01", "tanh"):
        x_tr = x_tr.astype(np.float32) / 255.0
        x_te = x_te.astype(np.float32) / 255.0
        if scale == "tanh":
            x_tr = x_tr * 2.0 - 1.0
            x_te = x_te * 2.0 - 1.0

    return (x_tr.reshape(len(x_tr), -1), y_tr.astype(np.int64),
            x_te.reshape(len(x_te), -1), y_te.astype(np.int64))


def one_hot(y: np.ndarray, n_class: int = 10) -> np.ndarray:
    out = np.zeros((len(y), n_class), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out


def to_image(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(28, 28)
    if x.min() < -0.05:
        x = (x + 1.0) / 2.0
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    xtr, ytr, xte, yte = load_mnist()
    print("train", xtr.shape, xtr.dtype, xtr.min(), xtr.max())
    print("test ", xte.shape, yte.shape, yte.dtype)
    print("label counts", np.bincount(ytr))
