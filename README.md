# 手搓 BP 神经网络（纯 NumPy · MNIST 十分类）

单隐藏层 784 → 256 → 10，ReLU + softmax 交叉熵 + Adam。**不需要 torch，不需要 GPU。**
数据已下好（`data/mnist/`，60000 训练 + 10000 测试，已验证可读）。

理论、公式推导、初始化数学、超参数、诊断表在 **[BP手搓笔记.md](BP手搓笔记.md)**。本文件只讲代码：怎么填、按什么顺序填、填完怎么验证。

---

## 三分钟上手

```powershell
python check_grad.py                     # ① 先让这个全绿（现在是红的，TODO 还没填）
python bp.py --epochs 2 --limit 2000     # ② 冒烟测试，几秒钟
python bp.py --epochs 20                 # ③ 正式训练，约 100 秒，测试准确率应到 97.5%+
```

---

## 文件

```
nn.py           层库骨架：Linear / ReLU / Sequential / Adam / softmax / 交叉熵   <- 12 个 TODO
bp.py           训练骨架：建模型 / 一步训练 / 评估 / 混淆矩阵                    <- 4 个 TODO
check_grad.py   数值梯度检查（三级）                                            <- 3 个 TODO
mnist.py        数据加载，完整可用，不用动
data/mnist/     IDX 文件，已下载
runs/bp/        训练输出（跑完才生成）
BP手搓笔记.md    理论、逐层公式、超参、诊断表
GAN手搓笔记.md   上一轮留下的 GAN 文档，与本次无关，可留可删
```

---

## 你要实现的 19 个函数

**建议严格按这个顺序**，每完成一步都能立刻验证：

| # | 文件 | 内容 | 难度 | 验证方式 |
|---|---|---|---|---|
| 1 | `nn.py` | `softmax` | 低 | 手算一行对照：每行和为 1 |
| 2 | `nn.py` | `softmax_cross_entropy` | 中 | `check_grad.py` 第 2 级 |
| 3 | `nn.py` | `Linear`（6 个方法） | 低 | `check_grad.py` 第 1 级 |
| 4 | `nn.py` | `ReLU`（3 个方法） | 低 | `check_grad.py` 第 1 级 |
| 5 | `nn.py` | `Adam.step` | 低 | 打印一步前后 `W1[0,0]` 看有没有变 |
| 6 | `check_grad.py` | `check_layer` / `check_softmax_ce` / `check_model` | 中 | 自己检查自己 |
| 7 | `bp.py` | `build_model` | 低 | 打印 `model.state().keys()` 和参数总量 |
| 8 | `bp.py` | `train_step` | 中 | 冒烟训练看 loss 降不降 |
| 9 | `bp.py` | `evaluate` / `confusion_matrix` | 低 | 测试准确率 |

★ **第 6 步是分水岭。** 在 `check_grad.py` 全绿之前不要启动训练 —— 反向传播写错时训练**不会报错**，只会安静地收敛到 90% 或者卡在 10%，你会去怀疑学习率、怀疑数据、怀疑人生。

参考：本骨架的完整实现实测 **15 项检查全绿（误差 1e-10 量级）**，5 轮达到 **97.6%**。

---

## 五条不能违反的接口约定

违反任何一条，症状都是"训练不收敛"而不是报错：

1. **参数数组必须用 `nn.DTYPE` 建**，不要写死 `np.float32`。`check_grad.py` 会把它临时改成 `float64`；写死了精度只有 1e-3，你会把浮点噪声当成写错了。
2. **`backward` 里参数梯度用 `+=` 累加。**
3. **`params()` / `grads()` 每次返回同一批数组对象**（不要 copy、不要重新分配）。优化器按引用持有参数并原地 `-=`。
4. **`Adam.step` 必须原地更新**（`p -= ...`，不是 `p = p - ...`）。写成后者参数永远不变，而 loss 曲线看起来还在动（动量在变）。
5. **`ReLU` 缓存 mask，`Linear` 缓存输入 `x`。** `backward(dout)` 只拿到上游梯度，其余信息必须由 `forward` 存进 `self`。

---

## 最容易写错的两处

### 1. `softmax_cross_entropy` 的最后一行

```
dlogits = (softmax(logits) - one_hot(y)) / N        # (N,C)
```

- 漏掉 `/N` → 梯度大 128 倍 → loss 一开始就炸或震荡不降
- 写成 `/C`（除以 10）→ 慢 12.8 倍 → "能学但很慢"

**两种都不会报错。** 这就是为什么 `check_grad.py` 第 2 级要单独检查这个函数 —— 它是全网络梯度的起点，错一个倍数，所有层的梯度都错同一个倍数，而 loss 依然在下降，只是慢。

推导（只有两行，值得自己推一遍）：`L = -z_y + log Σ_j e^{z_j}` ⇒ `∂L/∂z_k = p_k - δ_ky`。那个 C×C 的 softmax 雅可比矩阵**不需要**构造，链式相乘后会塌缩成这个 O(C) 的式子。

### 2. `zero_grad()` 的位置

```
logits = model.forward(xb)
loss, dlogits = nn.softmax_cross_entropy(logits, yb)
model.zero_grad()          # ★ 必须在 backward 之前（backward 是累加的）
model.backward(dlogits)
opt.step(model.grads())
```

放在 `backward` 之后（或者忘了放）会把上一批的梯度一起带上。

---

## 已给的部分（不用动）

- `mnist.py` —— 完整实现：自动下载、IDX 解析、`[0,1]` 归一化、`one_hot`、`to_image`。
- `nn.py` 的 `Module` 基类与 `Sequential`：参数收集、`zero_grad`、checkpoint 的 `state()` / `load_state()`（原地写入，保证优化器持有的引用依然有效）。
- `bp.py` 的：`predict`、`train_epoch`（打乱 + 分批的编排）、`save_curves`、`save_confusion`、`save_misclassified`、`save_checkpoint` / `load_checkpoint`、`train`、命令行。
- `check_grad.py` 的：`finite_diff`（中心差分）、`rel_err` / `abs_err`、主程序与结果表格。

`train_epoch` 里这段编排是刻意写死的，改之前先读笔记 §3：

```python
for start in range(0, len(x) - batch + 1, batch):
    idx = order[start:start + batch]
    stats = train_step(model, opt, x[idx], y[idx])   # 一个 batch 的全部动作在这里
```

---

## 实测数据（这台机器：Python 3.13 / numpy 2.3.5 / 无 CUDA）

| 项目 | 数值 |
|---|---|
| 参数总量 | 203,530（784×256 + 256 + 256×10 + 10） |
| 一个 epoch（60000 样本，469 步） | ≈ 4.9 s |
| 20 轮全量 | ≈ 100 s |
| 梯度检查 | 15 项，误差 1e-11 ~ 1e-13 |

```
epoch   1/5   4.90s  train loss 0.3246 acc 0.9113  |  test loss 0.1651 acc 0.9528
epoch   2/5   4.82s  train loss 0.1360 acc 0.9609  |  test loss 0.1136 acc 0.9666
epoch   3/5   4.91s  train loss 0.0935 acc 0.9728  |  test loss 0.0894 acc 0.9730
epoch   4/5   4.93s  train loss 0.0697 acc 0.9802  |  test loss 0.0824 acc 0.9760
epoch   5/5   5.08s  train loss 0.0538 acc 0.9844  |  test loss 0.0748 acc 0.9760
```

**你的实现跑出这个量级（5 轮 ≥ 96%、20 轮 97.5~98.3%）就说明反向传播是对的。** 低于 90% 或卡在 10%/50%，看笔记 §6。

---

## 输出文件与怎么看

| 文件 | 内容 |
|---|---|
| `runs/bp/curves.png` | 训练/测试的 loss 与准确率曲线 |
| `runs/bp/confusion.png` | 混淆矩阵热力图（带计数） |
| `runs/bp/misclassified.png` | 预测错的样本，每格标"真值→预测值" |
| `runs/bp/best.npz` | 测试准确率最高那一轮的参数 |

日志里每轮末尾脚本会自动判断：

- 训练/测试准确率差 > 0.02 → 提示**过拟合**（该加正则化了）
- 第 3 轮后测试准确率仍 < 0.5 → 提示**回 `check_grad.py`**
- 跑完打印各类准确率、最差的类、最常见的三种混淆

---

## 常见报错对照

| 报错 / 现象 | 原因 |
|---|---|
| `NotImplementedError: TODO: xxx` | 那一步还没实现 |
| `AttributeError: 'Linear' object has no attribute 'w'` | `Linear.__init__` 的参数初始化没写 |
| 准确率卡在 ~10% | `Adam.step` 用了 `p = p - ...`；或 `argmax(axis=0)` |
| 准确率卡在 ~50% | `softmax_cross_entropy` 的 `/N` 漏了；或某层忘了返回 `dx` |
| loss = NaN / inf | `softmax` 没减 `max`（exp 溢出）；或 lr 太大 |
| loss 降一点就震荡 | 忘了 `zero_grad()`；或 lr 太大 |
| `RuntimeWarning: overflow encountered in exp` | 同 NaN 那一条 |
| 训练很慢 | 检查 `dtype` 是不是 float64（应全程 float32） |
| 图里中文显示成方框 | matplotlib 默认字体没有中文字形；骨架里的图已全部用英文，别改成中文 |
