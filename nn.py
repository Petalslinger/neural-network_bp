from __future__ import annotations

from collections.abc import Iterator

import numpy as np

Array = np.ndarray

DTYPE = np.float32

def softmax(logits: Array) -> Array:

    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def softmax_cross_entropy(logits: Array, y: Array) -> tuple[float, Array]:
    N,C = logits.shape

    #loss
    z = logits - logits.max(axis=1, keepdims=True)
    logsumexp = np.log(np.exp(z).sum(axis=1))
    loss = float(np.mean(logsumexp - z[np.arange(N), y]))

    #解析梯度
    probs = np.exp(z)/np.exp(z).sum(axis=1, keepdims=True) # (N,C)

    dlogits = probs.copy()
    dlogits[np.arange(N), y] -= 1
    dlogits /= N

    return loss, dlogits


class Module:

    def forward(self, x: Array) -> Array:
        raise NotImplementedError

    def backward(self, dout: Array) -> Array:
        raise NotImplementedError

    def own_params(self) -> list[Array]:
        return []

    def own_grads(self) -> list[Array]:
        return []

    def _own_state(self) -> dict[str, Array]:
        return {}

    def children(self) -> list["Module"]:
        return []


    def modules(self) -> Iterator["Module"]:
        yield self
        for child in self.children():
            yield from child.modules()

    def params(self) -> list[Array]:
        out = list(self.own_params())
        for child in self.children():
            out.extend(child.params())
        return out

    def grads(self) -> list[Array]:
        out = list(self.own_grads())
        for child in self.children():
            out.extend(child.grads())
        return out

    def zero_grad(self) -> None:
        for g in self.grads():
            g[...] = 0.0

    def num_params(self) -> int:
        return int(sum(p.size for p in self.params()))

    def _walk_state(self, prefix: str = "") -> Iterator[tuple[str, Array]]:
        for key, arr in self._own_state().items():
            yield prefix + key, arr
        for i, child in enumerate(self.children()):
            yield from child._walk_state(prefix=f"{prefix}{i}.")

    def state(self) -> dict[str, Array]:
        return {k: v for k, v in self._walk_state()}

    def load_state(self, sd: dict[str, Array]) -> None:
        for key, arr in self._walk_state():
            arr[...] = sd[key]


class Linear(Module):

    def __init__(self, in_dim: int, out_dim: int, init: str | float = "he", rng=None):
        if rng is None:
            rng = np.random.default_rng()

        if isinstance(init, str):
            if init == "he":
                scale = np.sqrt(2.0 / in_dim)
            elif init == "xavier":
                scale = np.sqrt(1.0 / in_dim)
            else:
                raise ValueError(f"未知初始化方式：{init}")
        else:
            scale = float(init)

        # 权重，偏置
        self.w = (rng.standard_normal((in_dim, out_dim)) * scale).astype(DTYPE)
        self.b = np.zeros(out_dim,dtype=DTYPE)

        # 梯度buffer
        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)

        #前向缓存
        self.x = None

    def forward(self, x: Array) -> Array:
        #缓存 self.x = x，返回 x @ self.w + self.b，形状 (N,in) -> (N,out)
        self.x = x
        return x @ self.w + self.b

    def backward(self, dout: Array) -> Array:
        #参数梯度【累加】，返回输入梯度。

        self.dw +=self.x.T @ dout
        self.db += dout.sum(axis=0)
        return dout @ self.w.T

    def own_params(self) -> list[Array]:
        #返回 [self.w, self.b]
        return [self.w, self.b]

    def own_grads(self) -> list[Array]:
        return [self.dw, self.db]

    def _own_state(self) -> dict[str, Array]:
        #返回 {'w': self.w, 'b': self.b}
        return {'w': self.w, 'b': self.b}

class ReLU(Module):
    def __init__(self):
        #准备 mask 缓存位
        self.mask = None

    def forward(self, x: Array) -> Array:
        self.mask = x > 0
        return np.where(self.mask, x, 0)

    def backward(self, dout: Array) -> Array:
        return dout*self.mask

class Sequential(Module):

    def __init__(self, *layers: Module):
        self.layers = list(layers)

    def children(self) -> list[Module]:
        return self.layers

    def forward(self, x: Array) -> Array:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout: Array) -> Array:
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, i):
        return self.layers[i]


class Adam:

    def __init__(self, params: list[Array], lr: float = 1e-3,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):

        self.params = list(params)
        self.lr = lr
        self.b1 = beta1
        self.b2 = beta2
        self.eps = eps
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0

    def step(self, grads: list[Array]) -> None:
        self.t += 1

        b1,b2 = self.b1,self.b2
        lr,eps = self.lr,self.eps

        bias1 = 1.0 - b1**self.t
        bias2 = 1.0 - b2**self.t

        for p,g,m,v in zip(self.params,grads,self.m,self.v):
            m *=b1
            m += (1.0-b1)*g

            v *= b2
            v += (1.0-b2)*g*g

            p -= lr *(m/bias1)/(np.sqrt(v/bias2) + eps)
