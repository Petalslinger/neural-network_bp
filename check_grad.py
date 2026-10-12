from __future__ import annotations

from hmac import compare_digest
from string import digits

import numpy as np
from sklearn._loss import loss

import nn
from nn import Array

TOL = 1e-6   # 相对误差阈值
ATOL = 1e-8  # 绝对误差阈值：梯度本身≈0 的项只能靠它判定


def finite_diff(f, arr: Array, eps: float = 1e-5) -> Array:
    g = np.zeros_like(arr)
    flat = arr.reshape(-1)
    gflat = g.reshape(-1)
    for i in range(flat.size):
        old = flat[i]
        flat[i] = old + eps
        fp = float(f())
        flat[i] = old - eps
        fm = float(f())
        flat[i] = old
        gflat[i] = (fp - fm) / (2.0 * eps)
    return g


def rel_err(a: Array, b: Array) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = max(1e-8, float(np.max(np.abs(a) + np.abs(b))))
    return float(np.max(np.abs(a - b)) / denom)


def abs_err(a: Array, b: Array) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b)))

def check_layer(layer: nn.Module, x: Array, seed: int = 0) -> list[tuple[str, float, float]]:
    """[TODO] 对任意层（包括 Sequential）做完整梯度检查。

    返回 [(检查项名字, 相对误差, 绝对误差), ...]，检查项包括每个参数数组和输入 x。

    步骤：
      1) 构造一个"随机投影"损失，让它有确定的、非退化的上游梯度：
             rng = np.random.default_rng(seed)
             w_out = rng.standard_normal(layer.forward(x).shape)   # 固定不变
             L() = float((layer.forward(x) * w_out).sum())
             dout = w_out                                          # dL/d(输出)
         注意 w_out 只生成一次；每次调用 L() 内部都会重新 forward。

      2) 解析梯度：
             layer.zero_grad()                # ★ 必须先清零，backward 是累加的
             dx = layer.backward(dout.copy()) # 输入梯度
             ana = [g.copy() for g in layer.grads()]   # ★ 必须 copy

      3) 数值梯度：
             num_p = [finite_diff(L, p) for p in layer.params()]   # 直接传模型里的数组
             num_x = finite_diff(L, x)
         finite_diff 内部会原地扰动再还原，所以必须传 layer.params() 里的**那个**数组，
         不能传 copy（否则扰动不到模型，数值梯度恒为 0）。

      4) 用 rel_err / abs_err 逐项比对，名字里带上形状（如 'param0(784, 256)'），
         出错时容易定位。返回 [(名字, 相对误差, 绝对误差), ...]。

    两个坑：
      * `layer.grads()` 返回的是**引用**，下一次 backward 会覆盖它 —— 先 copy。
      * ReLU 在 0 点不可导；输入恰好落在 0 附近会失真（标准正态输入基本碰不到）。
    """
    rng = np.random.default_rng(seed)

    # 固定随机投影损失
    out = layer.forward(x)
    w_out = rng.standard_normal(out.shape)

    def L():
        return float((layer.forward(x) * w_out).sum())

    dout = w_out

    #解析梯度
    layer.zero_grad()
    dx = layer.backward(dout.copy())
    ana_params = [g.copy() for g in layer.grads()]
    ana_x = dx.copy()

    #数值梯度
    num_params = [finite_diff(L, p) for p in layer.params()]
    num_x = finite_diff(L, x)

    #逐项比对
    results=[]

    for i,(p,ana,num) in enumerate(zip(layer.params(), ana_params, num_params)):
        name = f"param{i}{tuple(p.shape)}"
        results.append(compare_grad(name, ana, num))

    results.append(compare_grad(f"input{tuple(x.shape)}",ana_x,num_x))
    return results

def compare_grad(name:str, ana:Array, num:Array) -> list[tuple[str, float, float]]:
    #计算解析梯度与数值梯度的相对误差与绝对误差
    ana = np.asarray(ana, dtype=np.float64)
    num = np.asarray(num, dtype=np.float64)

    abs_err = float(np.max(np.abs(ana - num)))
    denom = max(float(np.max(np.abs(ana)+np.abs(num))),1e-12 )

    return (name, abs_err/denom, abs_err)

def check_softmax_ce(seed: int = 0, n: int = 7, n_class: int = 5) -> list[tuple[str, float, float]]:
    rng = np.random.default_rng(seed)
    logits = (rng.standard_normal((n, n_class)) * 2).astype(nn.DTYPE)
    y = rng.integers(0, n_class, size=n)

    loss, dlogits = nn.softmax_cross_entropy(logits, y)
    num = finite_diff(lambda: nn.softmax_cross_entropy(logits, y)[0], logits)

    p = nn.softmax(logits)[np.arange(n), y]
    manual = float(-np.mean(np.log(p)))
    diff = abs(loss-manual)

    return [(f"dlogits{tuple(logits.shape)}", rel_err(num,dlogits),abs_err(num,dlogits)),
            ("loss vs -log softmax",diff,diff)]

def check_model(model: nn.Sequential, x: Array, y: Array,
                seed: int = 0) -> list[tuple[str, float, float]]:

   #损失函数
    def L() -> float:
        logits = model.forward(x)
        loss,_ = nn.softmax_cross_entropy(logits,y)
        return float(loss)

    #解析梯度
    logits = model.forward(x)
    loss,digits = nn.softmax_cross_entropy(logits,y)

    model.zero_grad()
    model.backward(digits)
    ana = [g.copy() for g in model.grads()]

    #数值梯度
    num = [finite_diff(L, p) for p in model.params()]

    results = []
    for i,(p,a,n) in enumerate(zip(model.params(), ana, num)):
        name = f"param{i}{tuple(p.shape)}"
        results.append(compare_grad(name, a, n))

    return results

def _build_cases(rng: np.random.Generator) -> list[tuple[str, tuple[nn.Module, Array]]]:
    return [
        ("Linear(6->5)", (nn.Linear(6, 5, "he", rng=rng), rng.standard_normal((8, 6)))),
        ("ReLU", (nn.ReLU(), rng.standard_normal((8, 5)))),
        ("Sequential 全链路", (nn.Sequential(
            nn.Linear(6, 8, "he", rng=rng),
            nn.ReLU(),
            nn.Linear(8, 4, "xavier", rng=rng),
        ), rng.standard_normal((8, 6)))),
    ]


def _report(results: list[tuple[str, float, float]]) -> int:
    n_fail = 0
    for pname, rel, abserr in results:
        ok = (rel < TOL) or (abserr < ATOL)
        n_fail += 0 if ok else 1
        note = "   (梯度≈0，按绝对误差通过)" if (ok and rel >= TOL) else ""
        print(f"  [{'OK' if ok else 'FAIL'}] {pname:<22} "
              f"相对 {rel:.3e}  绝对 {abserr:.3e}{note}")
    return n_fail


def main() -> int:

    nn.DTYPE = np.float64
    rng = np.random.default_rng(0)

    n_fail = 0
    n_total = 0
    dtype_warned = False

    print("=== 第 1 级：单层 / Sequential ===")
    try:
        cases = _build_cases(rng)
    except NotImplementedError as exc:
        print(f"  还没实现：{exc}")
        print("  先把 nn.py 里的 Linear / ReLU 写好，再回来跑这个脚本。")
        return 2

    for name, (layer, x) in cases:
        print(f"\n--- {name} ---")
        try:
            results = check_layer(layer, x)
        except NotImplementedError as exc:
            print(f"  还没实现：{exc}")
            return 2
        except Exception as exc:
            print(f"  [FAIL] 运行失败：{type(exc).__name__}: {exc}")
            n_fail += 1
            continue
        n_fail += _report(results)
        n_total += len(results)
        if not dtype_warned:
            params = layer.params()
            if params and params[0].dtype != np.float64:
                dtype_warned = True
                print(f"  [!] 参数 dtype 是 {params[0].dtype}，不是 float64。"
                      "说明某层把 dtype 写死了 —— 请改用模块常量 nn.DTYPE 建数组。")

    print("\n=== 第 2 级：softmax_cross_entropy（损失函数本身）===")
    try:
        results = check_softmax_ce()
        n_fail += _report(results)
        n_total += len(results)
    except NotImplementedError as exc:
        print(f"  还没实现：{exc}")
        return 2

    print("\n=== 第 3 级：端到端 Sequential + softmax_cross_entropy ===")
    try:
        model = nn.Sequential(nn.Linear(6, 8, "he", rng=rng), nn.ReLU(),
                              nn.Linear(8, 4, "xavier", rng=rng))
        x_e = rng.standard_normal((8, 6))
        y_e = rng.integers(0, 4, size=8)
        results = check_model(model, x_e, y_e)
        n_fail += _report(results)
        n_total += len(results)
    except NotImplementedError as exc:
        print(f"  还没实现：{exc}")
        return 2

    print("\n" + "=" * 60)
    if n_fail == 0 and n_total > 0:
        print(f"全部通过（{n_total} 项，阈值 {TOL:g}）。可以开始训练了："
              "python bp.py --epochs 2 --limit 2000")
        return 0
    print(f"{n_fail} / {n_total} 项未通过。按这个顺序排查：")
    print("  1. softmax_cross_entropy 的最后是不是忘了 /N 或写成 /C")
    print("  2. Linear.backward 是不是写成了 dout.T @ x（应该是 x.T @ dout）")
    print("  3. backward 里用了 `=` 覆盖而不是 `+=` 累加")
    print("  4. ReLU 的 mask 是不是在 backward 里重算成了 `dout * (dout > 0)`")
    print("  5. Sequential.backward 有没有逆序 `reversed(self.layers)`")
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
