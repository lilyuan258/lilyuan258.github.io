---
title: π0.5 模型架构与训练机制学习笔记
description: >-
  这份笔记总结了前面对 π0.5 paper
  中“模型架构、离散与连续表征结合、预训练、后训练与模型技术细节”的讲解。重点解释：高层子任务预测、交叉熵损失、符号 x {1:M} 的含义、离散动作
  token 与连续 action expert 的关系、flow matching 公式、timestep embedding 与 adaptiv
pubDate: '2026-06-17'
updatedDate: '2026-06-18'
tags:
  - robotics
  - VLA
  - flow-matching
  - pi0.5
  - Obsidian
draft: false
source: pi05_architecture_learning_notes_obsidian.md
wordCount: 5186
readingTime: 11
---
> 这份笔记总结了前面对 π0.5 paper 中“模型架构、离散与连续表征结合、预训练、后训练与模型技术细节”的讲解。重点解释：高层子任务预测、交叉熵损失、符号 $x_{1:M}$ 的含义、离散动作 token 与连续 action expert 的关系、flow matching 公式、timestep embedding 与 adaptive RMSNorm。

---
<img src="../../notes-assets/pi0-5-1.jpg" alt="pi0.5_1.jpg" loading="lazy" />
<img src="../../notes-assets/pi0-5-2.jpg" alt="pi0.5_2.jpg" loading="lazy" />
## 1. π0.5 的核心思想

π0.5 可以理解为在 π0 基础上的一次扩展：

- π0 的核心是 **VLM + continuous action expert + flow matching**。
- π0.5 进一步加入了 **大规模离散 token 预训练** 和 **高层子任务预测**。
- 训练阶段同时利用：
  - 离散 token 预测的可扩展性；
  - 连续 flow matching 的实时控制能力。

一句话概括：

> π0.5 用离散 token 表征做大规模预训练，用连续 action expert 做实时控制，中间通过高层子任务 $\hat{\ell}$ 把长程语义任务和低层动作执行连接起来。

---

## 2. 层级策略分解：先预测子任务，再生成连续动作

π0.5 将策略写成：

$$
\pi_\theta(\mathbf a_{t:t+H}, \hat{\ell} \mid \mathbf o_t, \ell)
=
\pi_\theta(\mathbf a_{t:t+H} \mid \mathbf o_t, \hat{\ell})
\pi_\theta(\hat{\ell} \mid \mathbf o_t, \ell)
$$

其中：

- $\mathbf o_t$：当前观测，包括多路图像和机器人状态。
- $\ell$：高层任务指令，例如 `clean the bedroom`。
- $\hat{\ell}$：模型生成的当前子任务，例如 `pick up the pillow`。
- $\mathbf a_{t:t+H}$：未来一段连续动作 chunk。

当前观测可以写成：

$$
\mathbf o_t = [\mathbf I_t^1, \ldots, \mathbf I_t^n, \mathbf q_t]
$$

其中：

- $\mathbf I_t^1, \ldots, \mathbf I_t^n$ 是多路相机图像；
- $\mathbf q_t$ 是 proprioceptive state，例如关节角、夹爪状态、底盘速度等。

这个分解的含义是：

1. 高层模块根据当前图像、状态和任务指令预测当前应该做什么：

$$
\hat{\ell} \sim \pi_\theta(\hat{\ell} \mid \mathbf o_t, \ell)
$$

2. 低层 action expert 根据当前观测和子任务生成连续动作：

$$
\mathbf a_{t:t+H} \sim \pi_\theta(\mathbf a_{t:t+H} \mid \mathbf o_t, \hat{\ell})
$$

重要点：低层动作分布主要依赖 $\hat{\ell}$，而不是直接依赖原始长程任务 $\ell$。也就是说，模型先把长程任务压缩成一个短期可执行的语义目标，然后再生成动作。

---

## 3. Transformer 视角：所有输入都被组织成 token 序列

论文把模型抽象成一个 Transformer：

$$
\mathbf y_{1:N} = f(\mathbf x_{1:N}, A(\mathbf x_{1:N}), \rho(\mathbf x_{1:N}))
$$

其中：

- $\mathbf x_{1:N}$：输入 token 序列；
- $\mathbf y_{1:N}$：输出 token 序列；
- $A(\mathbf x_{1:N})$：attention mask，决定哪些 token 能 attend 到哪些 token；
- $\rho(\mathbf x_{1:N})$：token type，表示每个 token 属于文本、图像、动作等哪种类型。

这里的 token 概念比较宽泛，不只包括文本 token，也包括图像 patch 和连续 action token。

常见 token 类型包括：

### 3.1 文本 token

$$
x_i^w \in \mathbb N
$$

例如：

- prompt token；
- 子任务描述 token；
- caption token；
- bounding box 离散坐标 token；
- FAST action token。

### 3.2 图像 patch token

$$
x_i^I \in \mathbb R^{p \times p \times 3}
$$

每个图像 patch 是一个连续 RGB patch。

### 3.3 连续动作 token

$$
x_i^a \in \mathbb R^d
$$

这是 flow matching 中的 noisy action token，$d$ 是动作维度。

---

## 4. 模型输出：文本 logits 与连续 action expert 输出

模型输出可以拆成两类：

$$
(y_{1:M}^{\ell}, y_{1:H}^{a})
$$

其中：

- $y_{1:M}^{\ell}$：文本/token head 的输出 logits，用于离散 token 预测；
- $y_{1:H}^{a}$：action expert 的输出，用于连续动作 flow matching。

注意：

- 文本/token head 负责生成语言、子任务、bbox token、FAST action token 等离散符号；
- action expert 负责预测连续动作生成过程中的 vector field。

---

## 5. 为什么 π0.5 同时使用离散动作和连续动作？

π0.5 的一个关键设计是同时使用两种动作表征：

1. **离散动作 token**，用于预训练；
2. **连续 action expert**，用于实时控制。

### 5.1 离散动作 token 的优点

通过 FAST action tokenizer，连续动作 chunk 可以被编码成离散 token。这样动作数据就可以像文本一样训练：

$$
\text{image/state/prompt} \rightarrow \text{FAST action tokens}
$$

优点是：

- 可以使用标准 autoregressive next-token prediction；
- 可以直接用交叉熵损失；
- 容易混合 web 数据、机器人数据、多 embodiment 数据；
- 适合大规模 VLM/VLA 预训练。

### 5.2 离散动作 token 的问题

离散动作 token 在推理时通常需要 autoregressive decoding，也就是一个 token 一个 token 地生成。这对实时机器人控制不够理想。

例如：

$$
\hat{x}_1 \rightarrow \hat{x}_2 \rightarrow \cdots \rightarrow \hat{x}_M
$$

每一步都依赖前一步，延迟较高。

### 5.3 连续 action expert 的优点

连续 action expert 通过 flow matching 并行生成整个动作 chunk：

$$
\mathbf a_{t:t+H}
$$

它不需要逐 token 解码动作，因此更适合高频控制。

### 5.4 π0.5 的折中

π0.5 的策略是：

- **预训练阶段**：主要使用离散 token，包括 FAST action token；
- **后训练阶段**：加入连续 action expert，并用 flow matching 训练；
- **推理阶段**：用高层 token head 生成子任务，用 action expert 生成连续动作。

---

## 6. 交叉熵损失：到底是谁和谁之间的差异？

论文中的离散 token 预测损失写成：

$$
H(x_{1:M}, f_\theta^{\ell}(\mathbf o_t, \ell))
$$

这项是交叉熵损失。

它比较的不是两个普通向量之间的欧氏距离，而是：

> 真实 token 的 one-hot 分布 与 模型预测的 token 概率分布 之间的差异。

---

## 7. $x_{1:M}$ 的真正含义

### 7.1 $x_{1:M}$ 不是“数据集中的 token 分布”

更准确地说：

$$
x_{1:M}
$$

表示某个训练样本中的 **ground-truth token 序列**。

它不是概率分布本身。真正的概率分布是模型输出的：

$$
p_\theta(\cdot \mid \text{context})
$$

### 7.2 交叉熵展开式

对于一个目标 token 序列 $x_{1:M}$，交叉熵可以写成：

$$
H(x_{1:M}, f_\theta^{\ell}(\mathbf o_t, \ell))
=
-\sum_{i=1}^{M}
\log p_\theta(x_i \mid \mathbf o_t, \ell, x_{<i})
$$

其中：

- $x_i$：第 $i$ 个真实 token；
- $x_{<i}$：真实 token 前缀；
- $p_\theta(x_i \mid \mathbf o_t, \ell, x_{<i})$：模型在当前位置给真实 token $x_i$ 分配的概率。

如果模型给真实 token 的概率高，loss 小：

$$
-\log 0.95 \quad \text{小}
$$

如果模型给真实 token 的概率低，loss 大：

$$
-\log 0.01 \quad \text{大}
$$

### 7.3 one-hot 视角

假设词表是：

$$
[\text{pick}, \text{place}, \text{open}, \text{close}, \text{move}]
$$

真实 token 是 `pick`，则真实分布是：

$$
q = [1,0,0,0,0]
$$

模型预测分布可能是：

$$
p = [0.7,0.1,0.05,0.05,0.1]
$$

交叉熵为：

$$
H(q,p) = -\sum_j q_j \log p_j
$$

由于 $q$ 是 one-hot，只有真实 token 那一项保留：

$$
H(q,p) = -\log p(\text{pick})
$$

---

## 8. 为什么论文中 $x$ 既像输入，又像标签？

这是因为 autoregressive language modeling 的训练方式会使用 **teacher forcing**。

同一条 token 序列既提供输入前缀，也提供下一个 token 的监督标签。

例如目标子任务是：

```text
pick up the pillow
```

tokenized 后抽象为：

$$
x_{1:M}=[\text{pick},\text{up},\text{the},\text{pillow}]
$$

训练时不是让模型直接整句输出，而是逐 token 预测：

$$
p_\theta(\text{pick} \mid \mathbf o_t, \ell)
$$

$$
p_\theta(\text{up} \mid \mathbf o_t, \ell, \text{pick})
$$

$$
p_\theta(\text{the} \mid \mathbf o_t, \ell, \text{pick up})
$$

$$
p_\theta(\text{pillow} \mid \mathbf o_t, \ell, \text{pick up the})
$$

因此：

- $x_{<i}$ 是输入上下文的一部分；
- $x_i$ 是当前位置的监督标签。

更严谨的写法应该是：

$$
\mathcal L_{\text{CE}}
=
-\sum_{i=1}^{M}
\log p_\theta(x_i^{\text{target}} \mid \mathbf o_t, \ell, x_{<i}^{\text{target}})
$$

但论文为了简洁，写成了：

$$
H(x_{1:M}, f_\theta^{\ell}(\mathbf o_t,\ell))
$$

所以这里的 $x$ 存在符号重载。

---

## 9. 高层子任务预测任务中，$x_{1:M}$ 具体是什么？

在高层子任务预测任务中，训练样本可以写成：

$$
(\mathbf o_t, \ell, \hat{\ell}_{\text{gt}})
$$

其中：

- $\mathbf o_t$：当前图像和机器人状态；
- $\ell$：高层任务指令；
- $\hat{\ell}_{\text{gt}}$：人类或作者标注的当前子任务描述。

例如：

```text
High-level instruction: clean the bedroom
Ground-truth subtask: pick up the pillow
```

那么：

$$
\hat{\ell}_{\text{gt}} = \text{``pick up the pillow''}
$$

经过 tokenizer 后：

$$
x_{1:M}=\operatorname{Tokenize}(\hat{\ell}_{\text{gt}})
$$

抽象理解为：

$$
x_{1:M}=[\text{pick},\text{up},\text{the},\text{pillow}]
$$

训练目标是最大化：

$$
p_\theta(x_{1:M} \mid \mathbf o_t,\ell)
$$

也就是最大化：

$$
p_\theta(\text{``pick up the pillow''} \mid \mathbf o_t, \text{``clean the bedroom''})
$$

展开成 autoregressive 形式：

$$
p_\theta(x_{1:M} \mid \mathbf o_t, \ell)
=
\prod_{i=1}^{M}
p_\theta(x_i \mid \mathbf o_t, \ell, x_{<i})
$$

对应损失是：

$$
\mathcal L_{\text{HL}}
=
-\sum_{i=1}^{M}
\log p_\theta(x_i \mid \mathbf o_t, \ell, x_{<i})
$$

所以，在高层子任务预测任务里：

> $x_{1:M}$ 就是训练数据中已有的、根据当前观测 $\mathbf o_t$ 和高层指令 $\ell$ 标注出来的子任务描述 token 序列。

---

## 10. 为什么有时 $x_{1:M}$ 里还包括 bounding box token？

有些高层任务数据不仅训练模型输出子任务文本，还训练模型先定位相关物体。

例如：

```text
Instruction: clean the bedroom
Relevant object: pillow
Bounding box: [125, 348, 260, 470]
Subtask: pick up the pillow
```

如果 bounding box 被离散化成 token，目标输出可能是：

$$
x_{1:M}
=
[
\text{<loc0125>},
\text{<loc0348>},
\text{<loc0260>},
\text{<loc0470>},
\text{pick},
\text{up},
\text{the},
\text{pillow}
]
$$

也就是说：

$$
x_{1:M}
=
[
\text{bbox coordinate tokens},
\text{subtask text tokens}
]
$$

这仍然是一个普通的离散 token 序列，只是 token 的语义不同。

### 10.1 为什么要先预测 bbox？

因为这可以增强视觉 grounding。

只输出：

```text
pick up the pillow
```

模型可能只学到语义层面的下一步任务。

但如果还要求它输出 pillow 的位置：

```text
<loc0125> <loc0348> <loc0260> <loc0470> pick up the pillow
```

模型就被迫学习：

$$
\text{当前图像中哪个物体与当前子任务有关}
$$

这对机器人执行很重要，因为低层控制不只需要知道“做什么”，还需要知道“对哪个物体做”。

---

## 11. Flow matching：连续 action expert 如何训练？

给定真实动作 chunk：

$$
\mathbf a_{t:t+H}
$$

采样高斯噪声：

$$
\omega \sim \mathcal N(0,I)
$$

再采样 timestep：

$$
\tau \in [0,1]
$$

构造 noisy action：

$$
\mathbf a_{t:t+H}^{\tau,\omega}
=
\tau \mathbf a_{t:t+H} + (1-\tau)\omega
$$

当 $\tau=0$ 时：

$$
\mathbf a_{t:t+H}^{0,\omega}=\omega
$$

即纯噪声。

当 $\tau=1$ 时：

$$
\mathbf a_{t:t+H}^{1,\omega}=\mathbf a_{t:t+H}
$$

即真实动作。

所以 $\mathbf a_{t:t+H}^{\tau,\omega}$ 是从噪声到真实动作之间的一条线性插值路径。

---

## 12. Flow matching 损失

action expert 学习预测一个 vector field。论文中的损失项可以写成：

$$
\left\|
\omega - \mathbf a_{t:t+H}
-
f_\theta^a(\mathbf a_{t:t+H}^{\tau,\omega}, \mathbf o_t, \ell)
\right\|^2
$$

其中：

- $f_\theta^a$：action expert；
- 输入是 noisy action、当前观测和语言条件；
- 输出是 vector field 的估计。

需要注意符号方向。

由插值路径：

$$
\mathbf a^{\tau,\omega}
=
\tau \mathbf a + (1-\tau)\omega
$$

对 $\tau$ 求导：

$$
\frac{d}{d\tau}\mathbf a^{\tau,\omega}
=
\mathbf a - \omega
$$

而论文的 target 写成：

$$
\omega - \mathbf a
$$

这是相反方向。这个正负号取决于论文定义的 denoising / integration convention。关键是训练阶段和推理阶段的积分方向要一致。

---

## 13. 总损失：交叉熵 + flow matching

π0.5 的联合目标可以写成：

$$
\mathbb E_{\mathcal D,\tau,\omega}
\left[
H(x_{1:M}, f_\theta^{\ell}(\mathbf o_t,\ell))
+
\alpha
\left\|
\omega-\mathbf a_{t:t+H}
-
f_\theta^a(\mathbf a_{t:t+H}^{\tau,\omega},\mathbf o_t,\ell)
\right\|^2
\right]
$$

两项分别是：

### 13.1 离散 token 预测损失

$$
H(x_{1:M}, f_\theta^{\ell}(\mathbf o_t,\ell))
$$

用于训练：

- caption；
- VQA answer；
- 子任务描述；
- bounding box token；
- FAST action token。

### 13.2 连续动作 flow matching 损失

$$
\alpha
\left\|
\omega-\mathbf a_{t:t+H}
-
f_\theta^a(\mathbf a_{t:t+H}^{\tau,\omega},\mathbf o_t,\ell)
\right\|^2
$$

用于训练 continuous action expert。

### 13.3 $\alpha$ 的作用

$\alpha$ 控制 flow matching 损失的权重。

- 预训练阶段：

$$
\alpha=0
$$

此时只训练离散 token prediction。

- 后训练阶段：

$$
\alpha=10.0
$$

此时同时训练 token prediction 和 continuous action expert。

---

## 14. timestep $\tau$ 为什么重要？

在 flow matching 里，模型看到的是 noisy action：

$$
\mathbf a^{\tau,\omega}
=
\tau \mathbf a + (1-\tau)\omega
$$

这个动作处在从噪声到真实动作之间的某个阶段。模型必须知道当前是哪个阶段。

如果：

$$
\tau=0.1
$$

则：

$$
\mathbf a^{0.1,\omega}=0.1\mathbf a+0.9\omega
$$

动作非常 noisy，模型应该做较大修正。

如果：

$$
\tau=0.9
$$

则：

$$
\mathbf a^{0.9,\omega}=0.9\mathbf a+0.1\omega
$$

动作已经接近真实动作，模型只需要细微修正。

所以 action expert 需要同时知道：

$$
\mathbf a^{\tau,\omega}
$$

和：

$$
\tau
$$

---

## 15. timestep embedding：从标量 $\tau$ 到高维向量

$\tau$ 是一个标量，例如 $0.37$。Transformer 不太适合直接利用裸标量，因此论文先把它编码成高维向量。

论文使用 sinusoidal positional encoding：

$$
\phi: \mathbb R \rightarrow \mathbb R^w
$$

即：

$$
\phi(\tau) \in \mathbb R^w
$$

可以抽象理解为：

$$
\phi(\tau)
=
[
\sin(c_1\tau),
\cos(c_1\tau),
\sin(c_2\tau),
\cos(c_2\tau),
\ldots
]
$$

然后经过一个两层 MLP：

$$
e_\tau
=
\operatorname{swish}
\left(
W_2 \cdot
\operatorname{swish}(W_1 \cdot \phi(\tau))
\right)
$$

其中：

- $e_\tau$：timestep embedding；
- $W_1,W_2 \in \mathbb R^{w\times w}$：可学习权重；
- $\operatorname{swish}$：激活函数。

swish 定义为：

$$
\operatorname{swish}(x)=x\cdot\sigma(x)
$$

其中：

$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$

这整个公式的本质是：

$$
\tau \rightarrow \phi(\tau) \rightarrow e_\tau
$$

即把 denoising timestep 编码成 action expert 可以使用的高维时间条件。

---

## 16. RMSNorm 与 adaptive RMSNorm

### 16.1 RMSNorm

给定 hidden state：

$$
h \in \mathbb R^d
$$

普通 RMSNorm 可以写成：

$$
\operatorname{RMSNorm}(h)
=
\frac{h}{\sqrt{\frac{1}{d}\sum_{j=1}^{d}h_j^2+\epsilon}}
\odot g
$$

其中：

- $g$ 是可学习缩放参数；
- $\odot$ 表示逐元素乘法；
- $\epsilon$ 是数值稳定项。

RMSNorm 的作用是稳定训练，使 hidden state 的尺度不会剧烈漂移。

### 16.2 adaptive RMSNorm

普通 RMSNorm 的缩放参数基本固定，与 $\tau$ 无关。

adaptive RMSNorm 的核心是：

> 归一化后的 hidden state 会被 timestep embedding $e_\tau$ 调制。

常见形式可以抽象写成：

$$
\operatorname{AdaRMSNorm}(h,e_\tau)
=
\gamma(e_\tau)\odot \operatorname{RMSNorm}(h)
$$

或者更一般地写成：

$$
\operatorname{AdaRMSNorm}(h,e_\tau)
=
\gamma(e_\tau)\odot \operatorname{RMSNorm}(h)+\beta(e_\tau)
$$

其中：

- $\gamma(e_\tau)$：由 timestep embedding 生成的缩放参数；
- $\beta(e_\tau)$：由 timestep embedding 生成的偏移参数。

因此：

$$
\tau
\rightarrow
e_\tau
\rightarrow
\gamma(e_\tau),\beta(e_\tau)
\rightarrow
\text{调制 Transformer 每一层}
$$

---

## 17. π0 和 π0.5 在 timestep 注入方式上的区别

可以粗略理解为：

### π0

π0 更像是在输入端把 timestep 和 noisy action 融合：

$$
[\mathbf a^{\tau,\omega},\tau]
\rightarrow
\text{embedding}
\rightarrow
\text{Transformer}
$$

### π0.5

π0.5 先将 $\tau$ 编码成 timestep embedding：

$$
\tau \rightarrow \phi(\tau) \rightarrow e_\tau
$$

再通过 adaptive RMSNorm 注入 action expert 的每一层：

$$
h^{(l+1)}
=
\operatorname{TransformerBlock}^{(l)}(h^{(l)};e_\tau)
$$

也就是说，π0.5 不是只在输入层告诉模型一次 timestep，而是在每一层都让 action expert 知道当前处于 denoising 的哪个阶段。

---

## 18. attention mask：防止离散动作和连续动作互相泄漏

<img src="../../notes-assets/image-1781683421772.webp" alt="image-1781683421772.webp" loading="lazy" width="498" />
π0.5 同时训练：

- FAST discrete action tokens；
- continuous action expert embeddings。

如果这两种动作表示可以随意互相 attend，就可能发生信息泄漏。

例如 continuous action expert 在训练时直接看到了 FAST action tokens，而 FAST tokens 本身是由真实动作编码来的。这样模型就不是根据图像和指令生成动作，而是在偷看另一个动作标签。

因此 π0.5 使用特殊 attention mask：

### 18.1 prefix token

图像、prompt、机器人状态等 prefix token 可以互相 attend。

### 18.2 FAST action tokens

FAST action tokens：

- 可以 attend 到 prefix；
- 可以 causal attend 到之前的 FAST tokens；
- 不能看未来 FAST tokens。

这和标准语言模型一致。

### 18.3 continuous action expert embeddings

continuous action embeddings：

- 可以 attend 到 prefix；
- 可以彼此之间双向 attend；
- 不 attend FAST action tokens。

连续动作 chunk 是并行生成的，因此 continuous action tokens 之间不需要 causal mask。

### 18.4 VLM embedding 不 attend action expert

信息流主要是：

$$
\text{VLM / prefix} \rightarrow \text{action expert}
$$

而不是：

$$
\text{action expert} \rightarrow \text{VLM}
$$

这可以避免连续动作分支反向污染 VLM 的离散 token 预测。

---

## 19. 预训练阶段

预训练阶段的目标是让模型成为一个通用 VLA，能够处理图像、语言、机器人状态和离散动作 token。

预训练阶段通常设置：

$$
\alpha=0
$$

因此总损失退化为：

$$
\mathcal L_{\text{pretrain}}
=
H(x_{1:M}, f_\theta^{\ell}(\mathbf o_t,\ell))
$$

也就是说，只训练离散 token prediction。

预训练数据包括：

- mobile manipulator household data；
- non-mobile robot household data；
- cross-embodiment lab data；
- high-level subtask prediction data；
- multimodal web data。

动作数据会被 FAST tokenizer 离散化，纳入 autoregressive token prediction 框架。

---

## 20. 后训练阶段

后训练阶段加入 continuous action expert，并训练 flow matching。

后训练阶段通常设置：

$$
\alpha=10.0
$$

因此损失包含：

$$
\mathcal L_{\text{posttrain}}
=
\mathcal L_{\text{CE}}
+
10.0\cdot \mathcal L_{\text{FM}}
$$

其中：

$$
\mathcal L_{\text{CE}}
=
H(x_{1:M}, f_\theta^{\ell}(\mathbf o_t,\ell))
$$

$$
\mathcal L_{\text{FM}}
=
\left\|
\omega-\mathbf a_{t:t+H}
-
f_\theta^a(\mathbf a_{t:t+H}^{\tau,\omega},\mathbf o_t,\ell)
\right\|^2
$$

后训练数据更聚焦目标部署场景，例如家庭移动操作数据，并保留部分 web data 和 high-level subtask data，以维持视觉语言能力和高层子任务预测能力。

---

## 21. 推理流程

推理时没有 ground-truth token，也没有真实动作标签。

流程是：

1. 输入当前观测和高层任务：

$$
(\mathbf o_t,\ell)
$$

2. VLM/token head 自回归生成子任务：

$$
\hat{\ell}\sim \pi_\theta(\hat{\ell}\mid \mathbf o_t,\ell)
$$

3. action expert 以 $\mathbf o_t$ 和 $\hat{\ell}$ 为条件，通过 flow matching 从噪声生成动作 chunk：

$$
\mathbf a_{t:t+H}\sim \pi_\theta(\mathbf a_{t:t+H}\mid \mathbf o_t,\hat{\ell})
$$

4. 机器人执行动作 chunk 的一部分或全部；

5. 重新观察环境并进入下一轮。

这类似 receding horizon control：不断观察、重新规划子任务、重新生成动作。

---

## 22. 一个完整例子

假设高层指令是：

```text
clean the bedroom
```

当前图像中有 pillow、blanket、shirt 等物体。

### 22.1 高层子任务预测

模型根据当前观测和高层指令生成：

```text
pick up the pillow
```

即：

$$
\hat{\ell}=\text{``pick up the pillow''}
$$

### 22.2 连续动作生成

action expert 采样噪声：

$$
\omega\sim \mathcal N(0,I)
$$

然后通过若干 denoising / integration steps 生成连续动作 chunk：

$$
\mathbf a_{t:t+H}
$$

这段动作可能包括：

- 移动底盘；
- 调整手臂；
- 张开夹爪；
- 接近 pillow；
- 闭合夹爪抓取。

执行一段动作后，系统再次观察环境，可能生成下一个子任务：

```text
place the pillow on the bed
```

---

## 23. 关键易混点总结

### 23.1 $x_{1:M}$ 到底是什么？

在离散 token 预测任务中：

$$
x_{1:M}=\text{ground-truth output token sequence}
$$

在高层子任务预测任务中：

$$
x_{1:M}=\operatorname{Tokenize}(\hat{\ell}_{\text{gt}})
$$

也就是人类标注的子任务描述 token 序列。

如果该样本还包含 object localization，则可能是：

$$
x_{1:M}
=
[
\text{bbox tokens},
\text{subtask tokens}
]
$$

### 23.2 为什么 $x$ 既是输入又是标签？

因为 teacher forcing：

$$
x_{<i}\text{ 是输入前缀},\qquad x_i\text{ 是预测目标}
$$

因此同一条训练序列在不同位置上既提供上下文，又提供监督。

### 23.3 交叉熵比较什么？

它比较：

$$
\text{真实 token 的 one-hot 标签}
$$

和：

$$
\text{模型预测的词表概率分布}
$$

单个 token 的 loss 是：

$$
-\log p_\theta(x_i\mid \mathbf o_t,\ell,x_{<i})
$$

### 23.4 flow matching 比较什么？

flow matching 的 MSE 比较：

$$
\text{真实 vector field target}
$$

和：

$$
\text{action expert 预测的 vector field}
$$

即：

$$
\omega-\mathbf a_{t:t+H}
$$

和：

$$
f_\theta^a(\mathbf a_{t:t+H}^{\tau,\omega},\mathbf o_t,\ell)
$$

之间的差异。

### 23.5 timestep embedding 的本质是什么？

它把标量 $\tau$ 转成高维向量：

$$
\tau\rightarrow e_\tau
$$

再通过 adaptive RMSNorm 调制 action expert 每一层，使模型知道当前 noisy action 处在 denoising 过程的哪个阶段。

---

## 24. 最短版本记忆

π0.5 的训练目标可以压缩成：

$$
\mathcal L
=
\underbrace{H(x_{1:M}, f_\theta^{\ell}(\mathbf o_t,\ell))}_{\text{离散 token 预测}}
+
\underbrace{\alpha
\left\|
\omega-\mathbf a_{t:t+H}
-
f_\theta^a(\mathbf a_{t:t+H}^{\tau,\omega},\mathbf o_t,\ell)
\right\|^2}_{\text{连续动作 flow matching}}
$$

其中：

- 第一项训练 VLM/token head 输出语言、bbox、子任务、FAST action token；
- 第二项训练 action expert 输出连续动作 vector field；
- 预训练时 $\alpha=0$，只做离散 token 学习；
- 后训练时 $\alpha>0$，加入连续动作流匹配；
- 推理时先生成子任务 $\hat{\ell}$，再生成连续动作 $\mathbf a_{t:t+H}$。

---

## 25. 与 π0 的关系

如果已经理解 π0，可以这样理解 π0.5：

- π0：重点是 VLM + action expert + flow matching。
- π0.5：在这个基础上加入：
  - 更强的大规模 VLM/VLA 预训练；
  - 离散 FAST action token；
  - 高层子任务预测；
  - 更明确的层级策略结构；
  - timestep 通过 adaptive RMSNorm 注入 action expert 每一层。

因此 π0.5 不是放弃 π0 的 continuous flow matching，而是把它和大规模离散 token 预训练结合起来。
