# BP 神经网络手搓笔记（MNIST 十分类 · 纯 NumPy）

> 精简版：784 → 256 → 10，ReLU + softmax 交叉熵 + Adam。没有 BatchNorm、没有 Dropout。
> 你写代码，这份文档负责给公式、顺序、超参、诊断表。骨架在 `nn.py` / `bp.py` / `check_grad.py`。

---

## 0. 一页速查

```
结构     : x(784) -> Linear(W1,b1) -> ReLU -> Linear(W2,b2) -> logits(10)
前向     : z1 = x·W1 + b1 ; a1 = relu(z1) ; z2 = a1·W2 + b2 ; p = softmax(z2)
损失     : L = -mean( log p[i, y_i] )
起点梯度 : dz2 = (p - onehot(y)) / N            ← ★ 全网络最关键的一行
反向     : dW2 = a1ᵀ·dz2 ; db2 = Σ_batch dz2 ; da1 = dz2·W2ᵀ
           dz1 = da1 ⊙ (z1 > 0) ; dW1 = xᵀ·dz1 ; db1 = Σ_batch dz1
优化器   : Adam lr=1e-3, β1=0.9, β2=0.999
一轮     : 打乱 -> 分批 -> forward -> loss -> zero_grad -> backward -> step
初始化   : W1 用 He(sqrt(2/784))，W2 用 Xavier(sqrt(1/256))，b 全零
超参     : batch 128，输入归一化到 [0,1]，10~20 轮
预期     : 5 轮 97.6%，20 轮 98% 左右（本机实测 4.9 秒/轮）
```

---

## 1. 前向传播

| 步骤 | 形状 | 说明 |
|---|---|---|
| `x` | (N, 784) | 像素归一化到 `[0,1]`，float32 |
| `z1 = x @ W1 + b1` | (N, 256) | `W1` (784, 256)，`b1` (256,) |
| `a1 = relu(z1)` | (N, 256) | 逐元素 `max(0, ·)` |
| `z2 = a1 @ W2 + b2` | (N, 10) | `W2` (256, 10)，`b2` (10,) |
| `p = softmax(z2)` | (N, 10) | 每行和为 1；**只在算 loss 和看置信度时用** |

**最后一层不加 softmax，输出原始 logits。** 两个理由：

1. `softmax_cross_entropy` 内部用 log-sum-exp 稳定写法，比"先 softmax 再取 log"数值上好得多（后者会遇到 `log(0) = -inf`）。
2. 前向不需要 softmax，反向公式里 softmax 也已经被解析地消掉了（见 §2.1）。

要预测类别时用 `argmax(z2, axis=1)` 就够了 —— softmax 是单调的，不改变 argmax。**注意 `axis=1`**，写成 `axis=0` 会得到全 0，准确率永远 10%。

---

## 2. 反向传播

### 2.1 从损失开始：softmax + 交叉熵的合并梯度

这是整个 BP 里最值得慢慢看的一步。记 `z = z2 ∈ R^C`（C=10），真实类别 `y`：

$$ p_k = \frac{e^{z_k}}{\sum_j e^{z_j}}, \qquad L = -\log p_y = -z_y + \log \sum_j e^{z_j} $$

对 `z_k` 求导：

$$ \frac{\partial L}{\partial z_k} = -\delta_{ky} + \frac{e^{z_k}}{\sum_j e^{z_j}} = p_k - \delta_{ky} $$

**这就是全部。** batch 平均之后：

$$ \frac{\partial L_{batch}}{\partial z} = \frac{p - \text{onehot}(y)}{N} $$

三点值得注意：

- **不需要显式构造 C×C 的 softmax 雅可比矩阵** `∂p_k/∂z_j`。虽然它确实存在，但因为 `L` 只依赖 `p` 的一个分量，链式相乘后那个矩阵会塌缩成上面这个 O(C) 的式子。手写 BP 最大的收益就在这里：你知道哪些中间量根本不用算。
- **两个指数项成对消掉**了（softmax 的分子和 log 的导数），所以这个梯度既简单又数值稳定。
- **`/N` 来自"loss 是 batch 均值"**。漏掉它梯度就大 128 倍，表现是 loss 一开始就炸或者震荡不降；写成 `/C`（除以类别数 10）则会慢 12.8 倍，表现是"能学但学得很慢"。这两种错误都不会报错，只能靠 §5 的梯度检查抓出来。

### 2.2 逐层公式

**Linear**：`y = x @ W + b`

```
dW += x.T @ dout          # (in,out)
db += dout.sum(axis=0)    # (out,)，沿 batch 维求和
dx  = dout @ W.T          # (N,in)，传给上一层
```

`db` 的 `axis=0` 是沿 batch 求和 —— 因为每个样本都往同一个偏置上加了同样一份梯度。写成 `sum()` 不带 axis 会把整个矩阵压成一个标量，维度不匹配会立刻报错，还算幸运。

**ReLU**：`a = max(0, z)`

```
dz = dout * (z > 0)       # z 必须是 forward 时缓存的输入
```

**softmax + 交叉熵**（网络的起点）：见 §2.1。

### 2.3 完整的一次反向传播

```
# ---- 前向 ----
z1 = x  @ W1 + b1          # (N,256)
a1 = relu(z1)
z2 = a1 @ W2 + b2          # (N,10)
loss, dz2 = softmax_cross_entropy(z2, y)      # dz2 = (p - onehot)/N

# ---- 反向（严格逆序）----
dW2 = a1.T @ dz2           # (256,10)
db2 = dz2.sum(axis=0)      # (10,)
da1 = dz2 @ W2.T           # (N,256)
dz1 = da1 * (z1 > 0)       # (N,256)
dW1 = x.T  @ dz1           # (784,256)
db1 = dz1.sum(axis=0)      # (256,)

# ---- 更新 ----
W1 -= Adam(dW1) ; b1 -= Adam(db1) ; W2 -= Adam(dW2) ; b2 -= Adam(db2)
```

注意 `x.T @ dz1` 里 `x` 是**这一批的输入**，所以 `Linear` 层必须在 `forward` 时把输入缓存下来（`self.x = x`）。同理 ReLU 要缓存 mask。这就是为什么 `Module` 的 `backward(dout)` 只需要一个上游梯度就够 —— 其余信息层自己藏着。

---

## 3. 训练循环：精确顺序

```python
for epoch in range(E):
    order = rng.permutation(len(x_train))          # ★ 每轮重新打乱
    for start in range(0, len(x_train), B):
        idx = order[start:start+B]
        xb, yb = x_train[idx], y_train[idx]

        logits = model.forward(xb)                  # 1. 前向
        loss, dlogits = softmax_cross_entropy(logits, yb)   # 2. 损失 + 起点梯度
        model.zero_grad()                           # 3. ★ 清零，必须在 backward 之前
        model.backward(dlogits)                     # 4. 反向（累加进各层 buffer）
        opt.step(model.grads())                     # 5. 原地更新参数
```

三个顺序问题：

- **`zero_grad()` 必须在 `backward()` 之前**。因为反向传播是 `+=` 累加的（这个设计是为了支持梯度累积和多分支），忘了清零就会把上一批的梯度一起带上。症状：loss 一开始降得飞快然后卡住或者震荡。
- **打乱每一轮都要重新做**。`permutation` 放在 epoch 循环里，不是外面。
- **`step` 用 `model.grads()` 拿到的引用**，优化器内部持有的是同一批数组对象并原地 `-=`。如果某层的 `own_params()` 每次返回新数组，参数就永远不会变 —— 而 loss 曲线可能因为别的原因看起来在动。

---

## 4. 初始化与超参数

### 4.1 初始化的数学

反向传播把每层的梯度乘上 `W`，前向传播把每层的激活乘上 `W`。如果初始 `W` 太大，信号逐层放大 → 饱和 / 溢出；太小则逐层衰减 → 前面几层几乎收不到梯度。所以初始化要控制**方差**。

设输入维度 `fan_in`，希望每层输出方差与输入方差相当：

| 方案 | std | 适用 |
|---|---|---|
| Xavier / Glorot | `sqrt(1/fan_in)` | 线性或 tanh/sigmoid 激活 |
| He / Kaiming | `sqrt(2/fan_in)` | ReLU（补偿被掐掉的一半） |

本网络：`W1` 用 He（后面跟 ReLU），`W2` 用 Xavier。

**为什么输出层不用 He**：He 的 `sqrt(2)` 因子是为 ReLU 补偿的，输出层后面没有 ReLU，再用 He 初始 logits 的尺度会大 `sqrt(2) ≈ 1.41` 倍，softmax 一开始就过于尖锐，梯度偏小、起步慢。数值上：He(256) 给 0.0884，Xavier(256) 给 0.0625。

`b` 全零即可（对称性由 `W` 的随机性打破，偏置不需要随机）。

### 4.2 超参数

| 参数 | 推荐值 | 说明 |
|---|---|---|
| batch size | 128 | 太小梯度噪声大，太大收敛慢且泛化略差 |
| 隐藏层宽度 | 256 | 本例参数总量 203,530 |
| Adam lr | 1e-3 | 分类任务的标准起点；不收敛先试 3e-4 和 3e-3 |
| β1 / β2 | 0.9 / 0.999 | **普通监督学习用 0.9**（GAN 里那个 0.5 是特例） |
| epochs | 10~20 | 5 轮就 97.6%，之后收益递减 |
| 输入归一化 | `[0,1]` | 用 0..255 原始像素会直接不收敛 |
| 权重衰减 | 无 | 精简版不加；要加就 1e-4 起 |

### 4.3 实测数据（这台机器：Python 3.13 / numpy 2.3.5 / 无 CUDA）

```
epoch   1/5   4.90s  train loss 0.3246 acc 0.9113  |  test loss 0.1651 acc 0.9528
epoch   2/5   4.82s  train loss 0.1360 acc 0.9609  |  test loss 0.1136 acc 0.9666
epoch   3/5   4.91s  train loss 0.0935 acc 0.9728  |  test loss 0.0894 acc 0.9730
epoch   4/5   4.93s  train loss 0.0697 acc 0.9802  |  test loss 0.0824 acc 0.9760
epoch   5/5   5.08s  train loss 0.0538 acc 0.9844  |  test loss 0.0748 acc 0.9760
```

一轮约 5 秒，20 轮约 100 秒。5 轮的各类准确率：

```
0:0.985  1:0.990  2:0.972  3:0.980  4:0.984
5:0.981  6:0.963  7:0.969  8:0.967  9:0.967
最差的是 6（0.963）；最常见混淆：7→2(13)  6→5(10)  9→4(10)
```

**这就是标准结果。** 如果你的实现跑出这个量级（5 轮 ≥ 96%，20 轮 97.5~98.3%），说明反向传播是对的。如果你跑出 90% 以下或者卡在 10%/50%，看 §6。

注意 **train acc 已经高于 test acc**（第 5 轮 98.4% vs 97.6%）—— 过拟合的苗头，只是 MNIST 上这个规模还压得住。这就是 §9 里加正则化的动机。

---

## 5. 梯度检查：三级验收

手搓反向传播**必须**做数值梯度检查，而且分三级做。级别越高，越接近真实训练：

| 级别 | 检查对象 | 抓得到什么 |
|---|---|---|
| 1 | 单层（`Linear` / `ReLU` / 整个 `Sequential`） | 该层的 `dW`/`db`/`dx` 公式写错 |
| 2 | `softmax_cross_entropy` 本身 | **起点梯度错一个倍数** —— 最难发现的一类，loss 照样下降只是慢 |
| 3 | `model + loss` 端到端 | 链式连接顺序错、某层忘了返回 `dx` |

方法：中心差分 `(L(θ+ε) - L(θ-ε)) / (2ε)` 当标准答案，和解析梯度比。

要点：
- **用 float64**。float32 下 `ε` 一抖误差就有 1e-3，你会把浮点噪声当成写错了。骨架里 `nn.DTYPE` 就是为这个准备的，`check_grad.py` 会临时改成 float64。
- `ε = 1e-5`（float64 中心差分的最优量级约 `eps_machine^(1/3) ≈ 6e-6`）。
- 相对误差应 < 1e-6，**正常实现能到 1e-10 ~ 1e-12**。看到 1e-3 就是真错了。
- **相对误差大但绝对误差小的情况要单独看**：如果某项的解析梯度本身≈0（数值上就是噪声），相对误差会被放大。骨架里判定规则是"相对 **或** 绝对"，`_report` 会标成"梯度≈0，按绝对误差通过"。
- 检查时 `zero_grad()` 一定要调（`backward` 是累加的），解析梯度拿到后**立刻 copy**（`grads()` 返回的是会被下一轮覆盖的引用）。

本骨架的 `python check_grad.py` 一共 15 项，全部应在 1e-10 量级通过。

---

## 6. 训练不好的诊断表

| 症状 | 最可能的原因 | 处方 |
|---|---|---|
| 卡在 **~10%**（等于瞎猜） | 参数根本没更新；或 `logits` 与标签错位 | 检查 `Adam.step` 是否写成 `p = p - ...`（必须 `p -= ...`）；检查 `argmax(axis=1)`；打印一步前后 `W1[0,0]` 看有没有变 |
| 卡在 **~50%** | 起点梯度错一个倍数（漏 `/N`）；或某一层没返回 `dx`（梯度到不了前面几层） | 跑 `check_grad.py` 的第 2、3 级 |
| train/test 都 < 90% | 欠拟合：lr 太小 / 轮数不够 / 隐藏层太窄 / 反向传播错 | 先跑梯度检查；再把 lr 乘 3、隐藏层加到 512 |
| **loss 不降但 acc 在动** | 学习率过大导致震荡；或 `logits` 与 `y` 的 batch 没对齐 | lr 减半；检查 `x[idx], y[idx]` 用的是同一个 `idx` |
| **loss = NaN / inf** | softmax 没减 `max`（`exp` 溢出）；或 lr 太大 | §2.1 的稳定写法；lr 除以 10 |
| loss 降得飞快然后卡住 | 忘了 `zero_grad()`，梯度一直在累加 | 把 `zero_grad()` 放回 `backward()` 之前 |
| **train 98% 但 test 92%**（gap > 2%） | 过拟合 | 早停 / 加 L2 / 加 Dropout / 减小隐藏层 / 加数据增强 |
| 训练极慢 | 在 Python 里逐样本训练；或每步都重新分配大数组 | 必须分批（batch 128）；用 float32 让 BLAS 走 SGEMM |
| 第一次跑就很差、第二次突然很好 | 初始化太差（std 写成固定值 0.01 之类） | 按 §4.1 用 He / Xavier |

---

## 7. 评估：准确率之外的三个指标

只用准确率会漏掉很多信息 —— 98% 的准确率既可能是"所有类都很准"，也可能是"9 个类几乎全对、1 个类全错"。

1. **混淆矩阵**（`confusion.png`）：`conf[i,j]` = 真实 i 被预测成 j 的次数。一眼看出系统性的混淆。本机 5 轮的结果：`7→2` 13 次、`6→5` 10 次、`9→4` 10 次 —— 都是经典混淆（手写 7 的一横不明显就像 2；9 的上圈不闭合就像 4）。
2. **每类准确率**：`conf.diagonal() / conf.sum(axis=1)`。最差的那个类才反映模型的短板。
3. **错例图**（`misclassified.png`）：把预测错的样本连"真值→预测值"一起画出来。很多错误看一眼就知道是标注本身有歧义 —— 这类错误不是模型的问题，继续调参没有意义。

**一个方法论提醒**：不要反复用测试集来选超参。选多了，测试集就被你偷偷过拟合了，报出来的准确率会虚高。严格做法是从训练集里切出验证集（比如 50000/10000）用于选超参，测试集只在最后看一次。

---

## 8. 常见坑清单

1. **`softmax_cross_entropy` 的 `/N`** —— 漏了就是"loss 炸/不降"，写成 `/C` 就是"能学但慢 12.8 倍"。
2. **`zero_grad()` 的位置**在 `backward()` **之前**，不是之后。
3. **参数梯度用 `+=`**，别用 `=` 覆盖（虽然这版单分支网络覆盖也不会错，但养成累加的习惯，加多分支/多任务时不会踩）。
4. **`Adam.step` 里必须原地更新** `p -= ...`。
5. **`ReLU` 要缓存 mask**，`Linear` 要缓存输入。
6. **`params()` 返回同一批数组对象**，不要 copy、不要重新分配。
7. **输入归一化到 `[0,1]`**，用 0..255 会直接不收敛。
8. **`argmax(axis=1)`**，不是 `axis=0`。
9. **不要对 logits 再套 sigmoid/tanh**，也不要手动 one-hot 再算 MSE —— 分类就用交叉熵。
10. **`Layer.forward` 里不要顺手 `zero_grad()`**，也不要为了省事把 `x` 存成全局变量。缓存放 `self`，这样多个模型实例互不干扰。

---

## 9. 下一步：给这个网络加什么

按"性价比 × 难度"排序：

| 加什么 | 预期收益 | 难度 | 要点 |
|---|---|---|---|
| **权重衰减 L2** | 抑制过拟合，+0.1~0.3% | ★ | 在 Adam 里 `g += wd * p`，或者直接在 loss 里加 `0.5*wd*Σw²` |
| **Dropout(0.2)** | 抑制过拟合，+0.1~0.3% | ★ | `mask = rng.random(shape) >= p`，训练时 `x*mask/(1-p)`（inverted），推理时**不做任何缩放** |
| **学习率衰减** | 后期更稳，+0.1% | ★ | 每 5 轮 lr×0.5，或余弦退火 |
| **早停** | 省时间，防过拟合 | ★ | 验证集 loss 连续 3 轮不降就停 |
| **第二个隐藏层 + BN** | +0.2% | ★★ | BN 的反向是三个求和的组合，公式最长、最容易漏项 |
| **数据增强** | **+0.5~1%** | ★★ | 随机平移 ±2 像素、旋转 ±10°。纯 NumPy 可以用 `scipy.ndimage` 或手写仿射 |
| **动量 SGD 代替 Adam** | 泛化常更好 | ★ | `v = μv + g; p -= lr*v`，μ=0.9，lr 要放大到 0.1 量级 |
| **卷积网络** | 99.3%+ | ★★★ | 反向传播要手写 `im2col` 或 `col2im`，难度陡增一个台阶 |
| **超参对比实验** | 理解最深 | ★ | 固定 seed，只改一个变量（初始化 / lr / 宽度），把曲线画在一张图上 |

**建议的下一个实验：初始化对比。** 把 `W1` 分别按 `0.01`、`Xavier`、`He`、`0.5` 初始化，各跑 3 轮，看 loss 曲线的差别。你会亲眼看到"太小 → 前面几层学不动"、"太大 → 一开始就饱和"，这是理解 §4.1 最快的方式，而且只需要改一个数字。

---

## 10. 参考资料

**原始论文**
- Rumelhart, Hinton & Williams, *Learning representations by back-propagating errors*, Nature 1986 —— 反向传播的源头
- LeCun et al., *Gradient-Based Learning Applied to Document Recognition*, 1998 —— LeNet 与 MNIST 数据集本身

**初始化的两篇**
- Glorot & Bengio, *Understanding the difficulty of training deep feedforward neural networks*, 2010 — [Xavier 初始化](https://proceedings.mlr.press/v9/glorot10a.html)
- He et al., *Delving Deep into Rectifiers*, 2015 — [He 初始化](https://arxiv.org/abs/1502.01852)

**优化与正则**
- Kingma & Ba, *Adam*, 2014 — [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
- Srivastava et al., *Dropout*, 2014 — [arXiv:1207.0580](https://arxiv.org/abs/1207.0580)
- Ioffe & Szegedy, *Batch Normalization*, 2015 — [arXiv:1502.03167](https://arxiv.org/abs/1502.03167)

**教程向**
- [CS231n: Backpropagation, Intuitions](https://cs231n.github.io/optimization-2/) —— 讲"梯度在计算图上怎么流"，和手搓最对味
- [CS231n: Neural Networks Part 2](https://cs231n.github.io/neural-networks-2/) —— 初始化、正则化、超参搜索
- [3Blue1Brown 神经网络系列](https://www.3blue1brown.com/topics/neural-networks) —— 反向传播的几何直觉，看一遍就知道自己在算什么
- 李沐《动手学深度学习》[多层感知机](https://zh.d2l.ai/chapter_multilayer-perceptrons/index.html) 与 [数值稳定性和模型初始化](https://zh.d2l.ai/chapter_multilayer-perceptrons/numerical-stability-and-init.html)
- Michael Nielsen, [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) —— 手写反向传播最好的入门书，配套代码也是纯 Python
