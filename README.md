# 纯 NumPy 手写 BP 神经网络

**不用 PyTorch / TensorFlow，只用 numpy 实现一个能认手写数字的神经网络。** 前向传播、反向传播、softmax 交叉熵、Adam 优化器全部手写，并配有一套数值梯度检查来验证反向传播的正确性。

单隐藏层：`784 → 256(ReLU) → 10(logits)`，在 MNIST 上测试准确率 **98.37%**。

---

## 实测结果

| 项目 | 数值 |
|---|---|
| 梯度检查（三级，15 项） | **全部通过**，误差 1e-11 量级 |
| 测试准确率 | 5 轮 **97.57%**，35 轮峰值 **98.37%** |
| 参数量 | 203,530（784×256 + 256 + 256×10 + 10） |
| 速度 | ≈ 2 秒 / 轮，40 轮约 80 秒（CPU，float32） |

第 200 轮时 train acc 达到 100%、test acc 98.21% —— 峰值其实出现在第 35 轮，之后是过拟合的平台期。

---

## 快速开始

```powershell
python check_grad.py          # ① 15 项梯度检查，验证反向传播
python bp.py --epochs 5       # ② 约 10 秒，测试准确率 97.6%
python bp.py --epochs 40      # ③ 约 80 秒，98.3% 左右
```

首次运行会自动下载 MNIST（约 11 MB）到 `data/mnist/`。依赖只有 `numpy`（画图需要 `matplotlib`，缺失时曲线图会降级为 `.txt`）。

---

## 文件结构

```
nn.py           手写层库：Linear / ReLU / Sequential / Adam / softmax / 交叉熵
bp.py           训练流程：建模型 / 一步训练 / 评估 / 混淆矩阵 / 可视化
check_grad.py   三级数值梯度检查（单层、损失函数、端到端）
mnist.py        IDX 数据加载（纯 urllib，不依赖 torchvision）
BP手搓笔记.md    理论：公式推导、初始化数学、超参、失败诊断表
```

被 `.gitignore` 排除（不进仓库）：`data/`（自动下载）、`runs/`（训练产物）、`.idea/`。

---

## 核心公式

网络结构（`x` 是一批图像，形状 `(N, 784)`）：

```
z1 = x @ W1 + b1            # (N, 256)
a1 = relu(z1)
z2 = a1 @ W2 + b2           # (N, 10)  ← 输出 logits，最后一层不加 softmax
p  = softmax(z2)
L  = -mean( log p[i, y_i] )
```

反向传播（严格逆序）：

```
dz2 = (p - onehot(y)) / N       # ★ softmax + 交叉熵合并求导的结果，全网络梯度的起点
dW2 = a1.T @ dz2 ; db2 = dz2.sum(0)
da1 = dz2 @ W2.T
dz1 = da1 * (z1 > 0)            # ★ 用 forward 缓存的 mask
dW1 = x.T  @ dz1 ; db1 = dz1.sum(0)
```

Adam 更新：

```
m = β1·m + (1-β1)·g
v = β2·v + (1-β2)·g²
w -= lr · (m/(1-β1^t)) / (√(v/(1-β2^t)) + ε)
```

训练一个 batch 的五步：

```python
logits = model.forward(x)                            # 前向
loss, dlogits = nn.softmax_cross_entropy(logits, y)  # 损失 + 起点梯度
model.zero_grad()                                    # ★ 必须在 backward 之前
model.backward(dlogits)                              # 反向（梯度累加）
opt.step(model.grads())                              # 原地更新参数
```

---

## 为什么要有 `check_grad.py`

手写的反向传播**推错了不会报错** —— 训练只会安静地收敛得慢一点，或者卡在某个准确率。所以需要一个独立的标准答案：数值微分。

```
∂L/∂w ≈ (L(w+ε) - L(w-ε)) / (2ε)
```

它不需要任何推导，慢但绝对可靠。检查分三级：

| 级别 | 检查对象 | 能抓到的错误 |
|---|---|---|
| 1 | 单层 + Sequential | 该层的梯度公式写错 |
| 2 | `softmax_cross_entropy` 本身 | **起点梯度错倍数**（最难发现的一类） |
| 3 | 整个模型端到端 | 链式连接顺序、某层忘了返回 `dx` |

判定用"相对误差 **或** 绝对误差"：梯度本身接近 0 的项（例如紧跟 BatchNorm 的 Linear 的 bias）只能靠绝对误差判定。

---

## 三个最容易写错的地方（都是这个项目实际踩过的）

### 1. `ReLU` 的 mask 必须存在 `self` 上

```python
def forward(self, x):
    self.mask = x > 0        # 存成局部变量 mask 的话，forward 返回就没了
    return np.where(self.mask, x, 0)

def backward(self, dout):
    return dout * self.mask  # backward 只拿到 dout，手上没有 x
```

### 2. `softmax_cross_entropy` 的最后一步要除以 **batch size**

```python
dlogits = (probs - onehot) / N     # N 是样本数（128），不是类别数（10）
```

漏掉 → 梯度大 128 倍；写成 `/C` → 慢 12.8 倍。**两种都不会报错**，只会让 loss 曲线看起来"怪"。第 2 级梯度检查就是专门抓这个的。

### 3. Adam 的分母里必须有二阶矩 `v`

```python
w -= lr * (m/bias1) / (np.sqrt(v/bias2) + eps)     # 正确
w -= lr * (m/bias1) / (np.sqrt(bias2)  + eps)      # 写成这样 v 就完全没被用到
```

这是本项目真实发生过的 bug。错误版本的实测代价：

| | 5 轮 test acc | 20 轮 test acc |
|---|---|---|
| 漏掉 `v` | 83.6% | 88.6% |
| 修正后 | **97.6%** | —（5 轮就够） |

优化器的错误**梯度检查抓不到**（它只验反向传播）。唯一可靠的验证是拿准确率对基准：这个规模的网络在 MNIST 上，Adam、lr=1e-3、5 轮必须上 97%。

---

## 输出文件

`runs/bp/` 下会生成：

| 文件 | 内容 |
|---|---|
| `best.npz` | 测试准确率最高那一轮的参数 |
| `confusion.png` | 混淆矩阵热力图（经典混淆：7→2、6→5、9→4） |
| `misclassified.png` | 预测错的样本，每格标注"真值→预测值" |
| `curves.png` | 训练/测试的 loss 与准确率曲线 |

`best.npz` 的 `meta` 里记录了最佳轮次和准确率。

---

## 想往下走

- **理论细节**（公式推导、初始化数学、超参、失败症状诊断表）：见 [BP手搓笔记.md](BP手搓笔记.md)
- **抑制过拟合**：L2 正则 / Dropout / 早停（当前 train 100% vs test 98.2%，间隙 1.8 个点）
- **再往上提精度**：数据增强（平移旋转，MLP 上最大的杠杆，+0.3~0.5）、卷积网络（99.2%+）
- **理解初始化**：把 `W1` 分别按 `0.01` / `Xavier` / `He` / `0.5` 初始化各跑 3 轮，看 loss 曲线的差别 —— 改一个数字就能看懂，性价比最高

单隐藏层 MLP 在这个数据集上的天花板约为 98.4~98.5%，本项目已经跑到 98.37%。
