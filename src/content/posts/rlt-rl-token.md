---
title: RL-Token / RLT 学习笔记：基于 RL Token 的 VLA 在线强化学习
description: RL Token / RLT 学习笔记：基于 RL Token 的 VLA 在线强化学习
pubDate: '2026-07-01'
updatedDate: '2026-07-01'
tags: []
draft: false
source: RLT_RL-Token_学习笔记_Obsidian.md
wordCount: 6220
readingTime: 13
---
> **Summary**
> 核心一句话
> RLT 的核心思想是：冻结大规模 VLA，把它作为“视觉语言状态表征器 + 参考动作生成器”，再通过一个压缩的 RL token 暴露给轻量 actor-critic，使在线 RL 不再从零探索，而是在 VLA 生成的 action chunk 附近做局部策略改进。

---

<img src="../../notes-assets/2026-07-01-145348.png" alt="屏幕截图 2026-07-01 145348.png" loading="lazy" width="536" />


## 1. 这篇工作想解决什么问题

已有 VLA，也就是 Vision-Language-Action 模型，已经通过大规模数据学到了较强的视觉理解、语言理解和动作先验。但如果直接对整个 VLA 做在线强化学习，会遇到几个问题。

第一，VLA 参数量大。截图里的 VLA backbone 包含 SigLIP、Gemma、action expert 等模块，整体是十亿参数量级。在线 RL 需要频繁采样和更新，直接更新整个 VLA 成本很高。

第二，机器人在线数据昂贵。真实机器人交互速度慢，失败可能损坏物体或机器人，所以不能像仿真里那样大规模随机探索。

第三，VLA 内部 token 表示很大。VLA 输出的是一组高维 token embeddings，而不是一个小状态向量。直接把所有 token 输入 actor-critic，会导致状态维度过大、critic 难训练、样本效率低。

所以 RLT 的设计思路不是“重新训练 VLA”，而是：

> 利用 VLA 的已有能力，让 RL 在 VLA 已经比较接近成功的行为附近做局部 refinement。

---

## 2. 方法总览

RLT 分成两个阶段。

第一阶段：调整 VLA，使其暴露 RL 接口。具体做法是训练一个小的 encoder-decoder transformer，从 VLA 内部 token 表示中提取一个压缩的 RL token，记作 $z_{\mathrm{rl}}$。

第二阶段：冻结 VLA 和 RL token 模块，只在线训练轻量 actor $\pi_\theta$ 和 critic $Q_\psi$。actor 的输入包括当前 RL state 和 VLA 生成的 reference action chunk，输出最终执行的 action chunk。

整体映射关系可以写成：

$$
s, \ell \xrightarrow{\mathrm{VLA}} z_{1:M}, \tilde{a}_{1:C}
$$

$$
z_{1:M} \xrightarrow{\mathrm{RL\ token\ module}} z_{\mathrm{rl}}
$$

$$
(z_{\mathrm{rl}}, s^p, \tilde{a}_{1:C}) \xrightarrow{\mathrm{actor}} a_{1:C}
$$

其中：

$s$ 表示当前机器人观测状态，例如图像、传感器信息等。

$\ell$ 表示语言指令。

$z_{1:M}$ 表示 VLA 内部的 $M$ 个 token embeddings。

$\tilde{a}_{1:C}$ 表示 VLA 生成的参考动作块。

$z_{\mathrm{rl}}$ 表示从 VLA token 中提取出的 RL token。

$s^p$ 表示机器人 proprioceptive state，也就是机器人本体状态，例如关节角、末端执行器位姿、夹爪状态等。

$a_{1:C}$ 表示 actor 最终输出并执行的动作块。

---

## 3. RL token：为什么需要它

VLA 的内部表示通常是：

$$
z_{1:M} = \{z_1, z_2, \ldots, z_M\}
$$

其中每个 $z_i$ 都是高维 embedding。截图里 hidden dimension 是 $2048$，所以每个 token 是 $2048$ 维。

如果直接把所有 token 输入 RL actor 和 critic，会导致输入维度很大，也会让 Q-learning 变得不稳定。

RLT 因此引入一个压缩 token：

$$
z_{\mathrm{rl}} \in \mathbb{R}^{2048}
$$

它把原来的 $M \times 2048$ 表示压缩成 $1 \times 2048$ 表示。这个 token 的作用类似一个面向 RL 的状态摘要。

可以把它理解为：

> RL token 是 VLA 内部表示和轻量 RL 网络之间的接口。

---

## 4. 公式 1：RL token 的提取

### 4.1 符号定义

$s$：当前观测状态。

$\ell$：语言指令。

$\theta_{\mathrm{vla}}$：VLA 参数。

$f(s,\ell;\theta_{\mathrm{vla}})$：VLA 前向传播后输出的 final-layer token embeddings。

$z_{1:M}$：VLA 输出的 $M$ 个 token embeddings。

$z_i$：第 $i$ 个 VLA token embedding。

$e_{\mathrm{rl}}$：新增的可学习特殊 token embedding，也就是 RL token 的初始 embedding。

$g_\phi$：轻量 encoder transformer，参数为 $\phi$。

$[z_{1:M}, e_{\mathrm{rl}}]$：把 VLA 的 token 序列和 RL token 拼接成长度为 $M+1$ 的序列。

$g_\phi([z_{1:M}, e_{\mathrm{rl}}])_{M+1}$：encoder 输出序列中第 $M+1$ 个位置的 embedding，也就是 RL token 位置的输出。

### 4.2 公式

$$
z_{\mathrm{rl}}
=
g_\phi([z_{1:M}, e_{\mathrm{rl}}])_{M+1}
$$

意思是：

先把 VLA 的 $M$ 个 token embeddings 和一个可学习的 RL token 拼接起来，然后送入 encoder transformer。encoder 输出后，取最后一个位置的 embedding，作为压缩后的 RL token。

这个机制类似 BERT 或 ViT 里的 `[CLS]` token，但目的不同。这里的 $z_{\mathrm{rl}}$ 不是用于分类，而是用于后续 actor-critic 的状态表示。

---

## 5. 为什么要用重建目标训练 RL token

仅仅加入一个可学习 token，并不能保证它学到有用信息。它可能什么都不编码，或者只编码局部信息。

所以 RLT 用一个 decoder transformer 训练它做重建任务：让 decoder 根据 $z_{\mathrm{rl}}$ 重建原始 VLA token embeddings。

这形成一个 autoencoding bottleneck。

直观上：

> 如果 $z_{\mathrm{rl}}$ 不能保留足够多的 VLA 表示信息，decoder 就无法重建 $z_{1:M}$。因此，为了降低重建损失，encoder 必须把重要信息压缩进 $z_{\mathrm{rl}}$。

---

## 6. 公式 2：自回归重建损失

### 6.1 符号定义

$\mathcal{D}$：少量 task-specific demonstration dataset。

$z_i$：VLA 输出的第 $i$ 个 token embedding。

$\bar{z}_i = \mathrm{sg}(z_i)$：对 $z_i$ 做 stop-gradient 后得到的目标 embedding。

$\mathrm{sg}(\cdot)$：stop-gradient 操作。它表示该变量在反向传播中被视为常量，不会向 VLA 反传梯度。

$z_{\mathrm{rl}}$：encoder 提取出的 RL token。

$d_\phi$：decoder transformer，参数为 $\phi$。

$h_\phi$：decoder 输出后的线性 projection head，用于把 decoder hidden state 投影到和 $z_i$ 相同的 embedding 空间。

$[z_{\mathrm{rl}}, \bar{z}_{1:i-1}]$：decoder 的输入前缀，包含 RL token 和前 $i-1$ 个 ground-truth VLA token embeddings。

$d_\phi([z_{\mathrm{rl}}, \bar{z}_{1:i-1}])_i$：decoder 在第 $i$ 个自回归预测位置上的 hidden state。

$h_\phi(d_\phi([z_{\mathrm{rl}}, \bar{z}_{1:i-1}])_i)$：对第 $i$ 个 VLA token embedding 的预测值。

### 6.2 公式

$$
\mathcal{L}_{\mathrm{ro}}
=
\mathbb{E}_{\mathcal{D}}
\left[
\sum_{i=1}^{M}
\left\|
h_\phi
\left(
d_\phi([z_{\mathrm{rl}}, \bar{z}_{1:i-1}])_i
\right)
-
\bar{z}_i
\right\|^2
\right]
$$

这就是 RL token 的 reconstruction objective。

它表示：在 demonstration dataset $\mathcal{D}$ 上，对每个 VLA token embedding 都进行重建，并最小化预测 embedding 和原始 embedding 之间的 L2 距离。

---

## 7. 关于公式 2 的一个精确理解

需要注意：

$$
d_\phi([z_{\mathrm{rl}}, \bar{z}_{1:i-1}])_i
$$

本身不是最终预测，而只是 decoder 在第 $i$ 个位置的 hidden state。

真正对 $\bar{z}_i$ 的预测是：

$$
\hat{z}_i
=
h_\phi
\left(
d_\phi([z_{\mathrm{rl}}, \bar{z}_{1:i-1}])_i
\right)
$$

因此重建损失也可以写成：

$$
\mathcal{L}_{\mathrm{ro}}
=
\mathbb{E}_{\mathcal{D}}
\left[
\sum_{i=1}^{M}
\left\|
\hat{z}_i - \bar{z}_i
\right\|^2
\right]
$$

其中：

$$
\hat{z}_i
=
h_\phi
\left(
d_\phi([z_{\mathrm{rl}}, \bar{z}_{1:i-1}])_i
\right)
$$

自回归过程可以展开理解为：

当 $i=1$ 时：

$$
\hat{z}_1
=
h_\phi(d_\phi([z_{\mathrm{rl}}])_1)
$$

即只用 RL token 预测第一个 VLA token embedding。

当 $i=2$ 时：

$$
\hat{z}_2
=
h_\phi(d_\phi([z_{\mathrm{rl}}, \bar{z}_1])_2)
$$

即用 RL token 和第一个真实 VLA embedding 预测第二个 embedding。

当 $i=3$ 时：

$$
\hat{z}_3
=
h_\phi(d_\phi([z_{\mathrm{rl}}, \bar{z}_1,\bar{z}_2])_3)
$$

即用 RL token 和前两个真实 VLA embeddings 预测第三个 embedding。

---

## 8. 可选的 VLA supervised fine-tuning

RLT 中，RL token reconstruction loss 对 VLA embedding 使用 stop-gradient，所以它本身不会更新 VLA。

但作者也允许可选地对 VLA 做 supervised fine-tuning，使 VLA 在当前任务上的初始动作更好。

### 8.1 符号定义

$\phi$：RL token encoder-decoder 模块参数。

$\theta_{\mathrm{vla}}$：VLA 参数。

$\mathcal{L}_{\mathrm{ro}}(\phi)$：RL token reconstruction loss。

$\mathcal{L}_{\mathrm{vla}}(\theta_{\mathrm{vla}})$：VLA 的监督动作预测损失。

$\alpha$：VLA supervised fine-tuning loss 的权重。

### 8.2 目标

$$
\phi, \theta_{\mathrm{vla}}
=
\arg\min_{\phi,\theta_{\mathrm{vla}}}
\mathcal{L}_{\mathrm{ro}}(\phi)
+
\alpha
\mathcal{L}_{\mathrm{vla}}(\theta_{\mathrm{vla}})
$$

如果 $\alpha = 0$，则不微调 VLA，只训练 RL token 模块。

如果 $\alpha > 0$，则在示范数据上同时对 VLA 做监督微调。

关键点是：在线 RL 阶段，VLA 仍然是冻结的。RL 只训练轻量 actor 和 critic。

---

## 9. action chunk 是什么

RLT 中，策略不是每一步输出一个动作，而是一次输出一段动作序列，称为 action chunk。

设单次 chunk 长度为 $C$，则动作块写作：

$$
a_{1:C} = (a_1, a_2, \ldots, a_C)
$$

其中 $a_t$ 表示 chunk 内第 $t$ 个动作。

VLA 输出的参考动作块记作：

$$
\tilde{a}_{1:C}
$$

actor 最终输出并执行的动作块记作：

$$
a_{1:C}
$$

action chunking 的好处是：减少高频重新规划压力，让动作更连续，也更适合 VLA 这类生成式动作模型。

---

## 10. RL state 的定义

在线 RL 阶段，RLT 的状态不是原始图像，而是：

$$
x = (z_{\mathrm{rl}}, s^p)
$$

其中：

$z_{\mathrm{rl}}$ 是从 VLA 内部 token 表示中提取出的 RL token。

$s^p$ 是机器人 proprioceptive state。

这样做的理由是：

$z_{\mathrm{rl}}$ 提供视觉语言语义、任务上下文、物体关系等高层信息。

$s^p$ 提供机器人低层控制所需的本体信息，例如关节状态、夹爪状态、末端位姿等。

---

## 11. critic：评价一个 action chunk 的价值

critic 写作：

$$
Q_\psi(x, a_{1:C})
$$

其中：

$Q_\psi$ 是 critic 网络，参数为 $\psi$。

$x$ 是当前 RL state。

$a_{1:C}$ 是当前状态下执行的动作块。

$Q_\psi(x, a_{1:C})$ 输出一个标量，表示在状态 $x$ 下执行动作块 $a_{1:C}$ 后，预期能够获得的折扣回报。

注意，这里 critic 评价的是整个 action chunk，而不是单步动作。

---

## 12. 公式 3：chunk-level TD target

### 12.1 符号定义

$\mathcal{B}$：replay buffer。

$(x, a_{1:C}, x') \sim \mathcal{B}$：从 replay buffer 中采样一个 chunk-level transition。

$x$：当前 RL state。

$a_{1:C}$：当前执行的动作块。

$x'$：执行完当前 action chunk 后的下一个 RL state。

$r_{t'}$：chunk 内第 $t'$ 个时间步获得的 reward。

$C$：action chunk 长度。

$\gamma$：discount factor，折扣因子。

$Q_\psi$：当前 critic。

$Q_{\psi'}$：target critic，参数为 $\psi'$。

$\pi_\theta$：当前 actor policy。

$a'$：在下一个状态 $x'$ 下，由 actor 采样出的下一个 action chunk。

### 12.2 TD target

$$
\hat{Q}
=
\sum_{t'=1}^{C}
\gamma^{t'-1} r_{t'}
+
\gamma^C
\mathbb{E}_{a' \sim \pi_\theta}
\left[
Q_{\psi'}(x', a')
\right]
$$

critic loss 是：

$$
\mathcal{L}_Q
=
\mathbb{E}_{(x,a_{1:C},x') \sim \mathcal{B}}
\left[
(\hat{Q} - Q_\psi(x, a_{1:C}))^2
\right]
$$

### 12.3 逐项解释

当前 chunk 内真实得到的折扣奖励是：

$$
\sum_{t'=1}^{C}
\gamma^{t'-1} r_{t'}
$$

也就是：

$$
r_1 + \gamma r_2 + \gamma^2 r_3 + \cdots + \gamma^{C-1}r_C
$$

chunk 结束后的未来价值估计是：

$$
\gamma^C
\mathbb{E}_{a' \sim \pi_\theta}
\left[
Q_{\psi'}(x', a')
\right]
$$

因为已经执行了 $C$ 个时间步，所以后续价值要乘 $\gamma^C$。

---

## 13. bootstrap value 是什么

在公式 3 中：

$$
\gamma^C
\mathbb{E}_{a' \sim \pi_\theta}
\left[
Q_{\psi'}(x', a')
\right]
$$

这一项就是折扣后的 bootstrap value。

bootstrap value 的意思是：

> 当前 action chunk 执行完之后，从下一个状态 $x'$ 开始，未来还能获得多少回报；但这个未来回报不是完整 rollout 出来的，而是由 target critic 估计出来的。

经典单步 TD target 是：

$$
\hat{Q}
=
r + \gamma Q_{\psi'}(s', a')
$$

其中：

$$
Q_{\psi'}(s', a')
$$

就是 bootstrap value。

RLT 中因为一次执行 $C$ 步，所以变成：

$$
\hat{Q}
=
r_1 + \gamma r_2 + \cdots + \gamma^{C-1}r_C
+
\gamma^C Q_{\psi'}(x', a')
$$

bootstrap 的好处是：不需要等整条 episode 结束，就可以用局部 transition 更新 critic。

代价是：它依赖 critic 自己的估计。如果 critic 估错，target 也会有偏差。因此通常使用 target network $Q_{\psi'}$ 来稳定训练。

---

## 14. actor：基于 VLA reference action 的动作分布

RLT 的 actor 不是从零生成动作，而是条件化在两个输入上：

$$
x = (z_{\mathrm{rl}}, s^p)
$$

以及：

$$
\tilde{a}_{1:C}
$$

其中 $\tilde{a}_{1:C}$ 是 VLA 给出的 reference action chunk。

---

## 15. 公式 4：actor 的高斯动作分布

### 15.1 符号定义

$\pi_\theta$：actor policy，参数为 $\theta$。

$a_{1:C}$：actor 最终采样出的 action chunk。

$x$：当前 RL state。

$\tilde{a}_{1:C}$：VLA 生成的 reference action chunk。

$\mu_\theta(x,\tilde{a}_{1:C})$：actor 网络输出的动作均值。

$\sigma$：动作噪声标准差。

$\sigma^2$：动作噪声方差。

$I$：单位矩阵。

$\mathcal{N}(\mu,\Sigma)$：均值为 $\mu$、协方差为 $\Sigma$ 的高斯分布。

### 15.2 公式

$$
\pi_\theta(a_{1:C}\mid x,\tilde a_{1:C})
=
\mathcal{N}
\left(
\mu_\theta(x,\tilde a_{1:C}),
\sigma^2 I
\right)
$$

这表示：

> 在给定当前状态 $x$ 和 VLA 参考动作 $\tilde{a}_{1:C}$ 的条件下，actor 对最终动作块 $a_{1:C}$ 定义了一个高斯分布。动作大概率落在 $\mu_\theta(x,\tilde a_{1:C})$ 附近。

---

## 16. 公式 4 右边是什么意思

右边：

$$
\mathcal{N}
\left(
\mu_\theta(x,\tilde a_{1:C}),
\sigma^2 I
\right)
$$

是一个多维高斯分布。

如果单步动作维度是 $d_a$，chunk 长度是 $C$，那么整个 action chunk 可以展平成：

$$
a_{1:C} \in \mathbb{R}^{C \cdot d_a}
$$

actor 输出的均值也是同样维度：

$$
\mu_\theta(x,\tilde a_{1:C}) \in \mathbb{R}^{C \cdot d_a}
$$

协方差矩阵是：

$$
\sigma^2 I
$$

它表示每个动作维度都有相同方差 $\sigma^2$，不同动作维度之间相互独立。

这个高斯采样过程也可以写成：

$$
a_{1:C}
=
\mu_\theta(x,\tilde a_{1:C})
+
\sigma \epsilon
$$

其中：

$$
\epsilon \sim \mathcal{N}(0,I)
$$

直观含义是：

先让 actor 网络输出一个动作块中心值，然后在这个中心值附近加入高斯噪声，用于探索。

---

## 17. 为什么 actor 要看 VLA reference action

如果 actor 只看 $x$，它需要从零学习控制策略。

如果 actor 只看 $\tilde{a}_{1:C}$，它可能只能复制或微调参考动作，缺少状态判断。

RLT 同时输入 $x$ 和 $\tilde{a}_{1:C}$，让 actor 学会：

> 在当前状态和机器人本体信息条件下，判断 VLA 的参考动作哪里需要修正。

这降低了探索难度。actor 不需要在整个高维动作空间里随机搜索，而是在 VLA 给出的合理动作附近做局部优化。

---

## 18. 公式 5：actor 的优化目标

### 18.1 符号定义

$\theta$：actor 参数。

$\pi_\theta$：actor policy。

$Q_\psi$：critic。

$x$：当前 RL state。

$a_{1:C}$：actor 采样出的 action chunk。

$\tilde{a}_{1:C}$：VLA reference action chunk。

$\beta$：policy constraint coefficient，控制 actor 靠近 reference action 的强度。

$\|a_{1:C} - \tilde{a}_{1:C}\|^2$：actor 动作和 VLA 参考动作之间的平方距离。

### 18.2 公式

$$
\mathcal{L}_\pi(\theta)
=
\mathbb{E}_{s \sim \mathcal{B},\ a_{1:C} \sim \pi_\theta}
\left[
-
Q_\psi(x, a_{1:C})
+
\beta
\|a_{1:C} - \tilde{a}_{1:C}\|^2
\right],
\quad
\tilde{a}_{1:C} \sim \pi_{\mathrm{vla}}(\cdot \mid s,\ell)
$$

第一项：

$$
-
Q_\psi(x, a_{1:C})
$$

表示希望最大化 critic 预测的价值。因为训练时是最小化 loss，所以最大化 $Q$ 等价于最小化 $-Q$。

第二项：

$$
\beta
\|a_{1:C} - \tilde{a}_{1:C}\|^2
$$

表示希望 actor 的动作不要离 VLA reference 太远。

如果 $\beta$ 大，actor 会更保守，更接近 VLA。

如果 $\beta$ 小，actor 更自由，但也更可能探索到不稳定动作。

---

## 19. reference action dropout

reference action conditioning 有一个风险：actor 可能直接复制 VLA 的 reference action。

因为 actor 的输入里有 $\tilde{a}_{1:C}$，loss 里又有：

$$
\beta
\|a_{1:C} - \tilde{a}_{1:C}\|^2
$$

所以 actor 可能学成一个 copy machine。

为缓解这个问题，RLT 使用 reference action dropout。

做法是：训练时随机选择一部分 batch，把输入给 actor 的 reference action 替换成零向量：

$$
\tilde{a}_{1:C} \leftarrow 0
$$

这样 actor 被迫依赖状态 $x = (z_{\mathrm{rl}}, s^p)$ 生成动作，而不能总是复制 VLA。

其作用是平衡两件事：

有 reference 时，actor 学会在 VLA 动作附近做局部修正。

没有 reference 时，actor 仍然保留独立动作生成能力。

---

## 20. 完整系统流程

RLT 的完整系统包括以下组件。

### 20.1 warmup

在线 RL 开始前，先用 VLA policy 执行 $N_{\mathrm{warm}}$ 个环境步，把数据存入 replay buffer $\mathcal{B}$。

这避免了 RL 从随机动作开始探索。

warmup 的作用是：

- 让 critic 一开始就能看到相对合理的数据。
- 让 actor 的改进从 VLA 的动作分布附近开始。
- 提升真实机器人在线训练的安全性和样本效率。

---

### 20.2 chunk-level rollout

在每个 action chunk 边界，系统执行：

首先，VLA 根据当前状态和语言指令采样 reference action chunk：

$$
\tilde{a}_{t:t+C-1}
\sim
\pi_{\mathrm{vla}}(\cdot \mid s_t,\ell)
$$

然后，从 VLA 表示中提取 RL token：

$$
z_{\mathrm{rl}}(s_t)
$$

构造 RL state：

$$
x_t = (z_{\mathrm{rl}}(s_t), s_t^p)
$$

actor 输出最终动作块：

$$
a_{t:t+C-1}
\sim
\pi_\theta(\cdot \mid x_t, \tilde{a}_{t:t+C-1})
$$

执行动作块后，观察 reward 和下一状态，并存入 replay buffer。

---

### 20.3 human intervention

如果人类在执行中干预，则执行人类动作：

$$
a^{\mathrm{human}}_{1:C}
$$

而不是 actor 或 VLA 的动作。

更重要的是，如果发生 human intervention，RLT 会把 replay buffer 中的 reference action 替换为人类动作：

$$
\tilde{a}_{1:C}
\leftarrow
a^{\mathrm{human}}_{1:C}
$$

这样 actor 的 regularization target 就变成人类修正动作，而不是原本可能错误的 VLA reference。

这对于 contact-rich 或 safety-critical 阶段很重要。

---

### 20.4 replay buffer 中存什么

每个 chunk-level transition 可以写成：

$$
\langle
x_t,
a_{t:t+C-1},
\tilde{a}_{t:t+C-1},
r_{t:t+C-1},
x_{t+C}
\rangle
$$

其中：

$x_t$ 是当前 RL state。

$a_{t:t+C-1}$ 是实际执行的 action chunk。

$\tilde{a}_{t:t+C-1}$ 是 reference action chunk，可能来自 VLA，也可能在人类干预时被替换为 human action。

$r_{t:t+C-1}$ 是 chunk 内的 reward 序列。

$x_{t+C}$ 是执行完 chunk 后的下一个 RL state。

---

### 20.5 subsampling action chunks

虽然策略每次执行长度为 $C$ 的 chunk，但执行过程中每个中间时间步也有 observation。

因此可以从一条轨迹中用 stride 采样多个重叠 action chunks，例如 stride 为 2：

$$
\langle x_0, a_{0:C} \rangle,
\langle x_2, a_{2:C+2} \rangle,
\langle x_4, a_{4:C+4} \rangle,
\ldots
$$

这样可以增加 replay buffer 中的训练样本数量，提高数据利用率。

---

### 20.6 异步 rollout 与 learning

RLT 可以让数据采集和模型更新异步进行。

机器人继续 rollout，学习线程同时从 replay buffer 中采样 batch 更新 critic 和 actor。

这对真实机器人很重要，因为真实交互慢，每份数据都需要高效复用。

---

### 20.7 critical phase targeting

RLT 不一定要接管整个长程任务，而是可以只优化最困难的关键阶段。

例如一个任务可以拆成：

1. 接近物体。
2. 抓取物体。
3. 移动到目标附近。
4. 执行插入、对齐、放置等精密接触动作。

VLA 可能已经能做好前面几步，但在最后接触阶段失败。RLT 可以只在这个关键阶段接管并训练 RL policy。

这样做的好处是：

- 降低长程 credit assignment 难度。
- 把在线数据集中到真正失败的阶段。
- 减少不必要探索。
- 保留 VLA 在长程任务上的通用能力。

---

## 21. Algorithm 1：RLT 伪代码笔记

```text
Require:
    Frozen VLA backbone f_{theta_vla}
    VLA action distribution pi_vla
    Demonstration data D
    Chunk length C
    Replay buffer B
    Warmup steps N_warm
    Update ratio G
    VLA fine-tuning weight alpha
    Policy constraint beta

Stage 1: Train RL token module

1. For samples (s, ell) from D:
       Compute VLA embeddings:
           z_i = f_i(s, ell; theta_vla)

       Extract RL token:
           z_rl = g_phi([z_{1:M}, e_rl])_{M+1}

       Train encoder-decoder by reconstruction:
           minimize L_ro(phi)

2. Optionally fine-tune VLA:
       minimize L_ro(phi) + alpha * L_vla(theta_vla)

Stage 2: Online RL

3. Initialize critic Q_psi and actor pi_theta

4. For environment steps t = 0, C, 2C, ...:
       Sample VLA reference action chunk:
           tilde_a_{t:t+C-1} ~ pi_vla(. | s_t, ell)

       Construct RL state:
           x_t = (z_rl(s_t), s_t^p)

       Choose executed action:
           if human intervention:
               a_{t:t+C-1} = a_human
           else if t < N_warm:
               a_{t:t+C-1} = tilde_a_{t:t+C-1}
           else:
               a_{t:t+C-1} ~ pi_theta(. | x_t, tilde_a_{t:t+C-1})

       Execute action chunk and observe rewards and next state

       If human intervention:
           tilde_a_{t:t+C-1} = a_human

       Store transition in replay buffer B

       For g = 1 to G:
           Sample batch b from B

           Compute TD target:
               Q_hat = sum_{t'=1}^{C} gamma^{t'-1} r_{t'}
                       + gamma^C E_{a' ~ pi_theta}[Q_{psi'}(x', a')]

           Update critic:
               minimize (Q_hat - Q_psi(x,a))^2

           Update actor:
               minimize -Q_psi(x,a) + beta ||a - tilde_a||^2
```

---

## 22. 什么是在线 RL

在线 RL 指的是：智能体一边和环境交互采集新数据，一边用这些新数据更新策略。

基本闭环是：

$$
s_t \rightarrow a_t \rightarrow r_t, s_{t+1} \rightarrow \text{update policy}
$$

其中：

$s_t$ 是当前状态。

$a_t$ 是执行动作。

$r_t$ 是 reward。

$s_{t+1}$ 是下一状态。

在机器人里，就是机器人真的执行动作、观察成败、把经验存入 replay buffer，然后更新 actor 和 critic。

---

## 23. 在线 RL、离线 RL、模仿学习的区别

### 23.1 在线 RL

在线 RL 的数据流是：

$$
\text{当前策略交互环境}
\rightarrow
\text{收集新数据}
\rightarrow
\text{更新策略}
\rightarrow
\text{继续交互}
$$

它可以根据当前任务失败模式主动收集新数据。

---

### 23.2 离线 RL

离线 RL 的数据流是：

$$
\text{固定数据集}
\rightarrow
\text{训练策略}
$$

训练期间不再和环境交互。策略只能从已有数据中学习，不能主动试新动作。

---

### 23.3 模仿学习

模仿学习通常学习：

$$
s \mapsto a^{\mathrm{demo}}
$$

目标是让模型动作接近专家示范：

$$
\|a - a^{\mathrm{demo}}\|^2
$$

它关心的是“动作是否像专家”，不直接最大化任务 reward。

在线 RL 关心的是“动作是否真的带来更高回报”。

所以 RL 可以学到一些不完全像示范、但成功率更高的动作。

---

## 24. RLT 中的在线 RL 具体指什么

在 RLT 中，在线 RL 不是更新整个 VLA，而是：

- 冻结 VLA。
- 冻结 RL token 模块。
- 让机器人真实执行任务。
- 用真实交互数据训练轻量 actor 和 critic。
- 让 actor 在 VLA reference action 附近做改进。

流程可以写成：

$$
\tilde{a}_{1:C}
\sim
\pi_{\mathrm{vla}}(\cdot \mid s,\ell)
$$

$$
x = (z_{\mathrm{rl}}, s^p)
$$

$$
a_{1:C}
\sim
\pi_\theta(\cdot \mid x,\tilde{a}_{1:C})
$$

然后执行 $a_{1:C}$，得到 reward 和下一状态，并更新：

$$
Q_\psi(x,a_{1:C})
$$

以及：

$$
\pi_\theta
$$

---

## 25. RLT 和 residual RL 的关系

传统 residual RL 常写成：

$$
a = a_{\mathrm{base}} + \Delta a_{\mathrm{rl}}
$$

其中：

$a_{\mathrm{base}}$ 是基础策略动作。

$\Delta a_{\mathrm{rl}}$ 是 RL 学到的残差动作。

RLT 更一般，它不是显式输出残差，而是让 actor 条件化在 VLA reference action 上：

$$
a \sim \pi_\theta(\cdot \mid x,\tilde{a})
$$

同时通过正则项限制它不要离 reference 太远：

$$
\beta \|a-\tilde{a}\|^2
$$

所以 RLT 可以理解为一种 implicit residual policy 或 reference-conditioned policy improvement。

---

## 26. RLT 的稳定性来源

RLT 之所以适合真实机器人在线 RL，是因为它叠加了多个稳定化设计。

- VLA 冻结，避免在线 RL 破坏大模型预训练能力。
- RL token 冻结，保证 critic 输入表示稳定。
- warmup 用 VLA 数据填充 replay buffer，避免随机探索。
- actor 条件化在 reference action 上，降低动作搜索空间。
- L2 constraint 限制 actor 不要偏离 VLA 太远。
- reference action dropout 防止 actor 只复制 VLA。
- human intervention 在危险或困难阶段提供人工修正。
- critical phase targeting 只优化关键失败阶段，降低长程任务难度。

---

## 27. 可能的局限

RLT 假设 VLA 已经能给出接近合理的动作。如果 VLA reference 离成功行为太远，局部 refinement 可能不够。

RL token 的质量依赖 reconstruction objective。如果“重建 VLA embeddings”不等价于“保留控制相关信息”，那么 $z_{\mathrm{rl}}$ 可能不是最优 RL state。

critic 仍然可能受到 sparse reward、分布偏移和 bootstrapping bias 的影响。

$\beta$ 需要调节。$\beta$ 太大，actor 退化为复制 VLA；$\beta$ 太小，actor 可能偏离先验导致探索不稳定。

reference action dropout 的比例也需要平衡。dropout 太少，actor 依赖 reference；dropout 太多，会削弱 VLA prior 的作用。

critical phase handover 训练阶段可能需要人类参与。

---

## 28. 常见易混点

### 28.1 $d_\phi(\cdot)_i$ 是不是预测值？

不是。

$$
d_\phi([z_{\mathrm{rl}}, \bar{z}_{1:i-1}])_i
$$

是 decoder hidden state。

真正预测第 $i$ 个 VLA token embedding 的是：

$$
h_\phi
\left(
d_\phi([z_{\mathrm{rl}}, \bar{z}_{1:i-1}])_i
\right)
$$

---

### 28.2 bootstrap value 是真实未来奖励吗？

不是。

bootstrap value 是 critic 对未来回报的估计：

$$
Q_{\psi'}(x',a')
$$

它不是完整 rollout 到 episode 结束后得到的真实总回报，而是用当前学到的 target critic 估出来的未来价值。

---

### 28.3 公式 4 的等号右边是不是动作？

不是。

$$
\mathcal{N}
\left(
\mu_\theta(x,\tilde a_{1:C}),
\sigma^2 I
\right)
$$

是动作分布，不是动作本身。

动作 $a_{1:C}$ 是从这个分布中采样出来的：

$$
a_{1:C}
=
\mu_\theta(x,\tilde a_{1:C})
+
\sigma \epsilon,
\quad
\epsilon \sim \mathcal{N}(0,I)
$$

---

### 28.4 在线 RL 是不是直接微调 VLA？

在 RLT 中，不是。

在线 RL 阶段冻结 VLA 和 RL token 模块，只训练 actor 和 critic。

VLA 只负责提供：

$$
z_{1:M}
$$

和：

$$
\tilde{a}_{1:C}
$$

actor-critic 负责在线学习如何改进动作块。

---

## 29. 最终理解

RLT 可以概括为：

$$
\text{Frozen VLA}
+
\text{RL Token Interface}
+
\text{Reference-conditioned Actor-Critic}
$$

VLA 提供强大的视觉语言动作先验。

RL token 把 VLA 的大规模 token 表示压缩成适合 RL 使用的状态。

actor 在 VLA reference action 附近做局部策略改进。

critic 用 chunk-level TD learning 评价动作块价值。

因此，RLT 的目标不是从零学机器人策略，而是：

> 在 VLA 已经有一定能力的基础上，通过真实机器人在线交互，对困难任务阶段进行高样本效率的局部强化学习优化。

## 30. 纸质笔记

<img src="../../notes-assets/rlt1.jpg" alt="RLT1.jpg" loading="lazy" />


<img src="../../notes-assets/rlt2.jpg" alt="RLT2.jpg" loading="lazy" />


<img src="../../notes-assets/rlt3.jpg" alt="RLT3.jpg" loading="lazy" />


<img src="../../notes-assets/rlt4.jpg" alt="RLT4.jpg" loading="lazy" />


<img src="../../notes-assets/rlt5.jpg" alt="RLT5.jpg" loading="lazy" />


<img src="../../notes-assets/rlt6.jpg" alt="RLT6.jpg" loading="lazy" />


<img src="../../notes-assets/rlt7.jpg" alt="RLT7.jpg" loading="lazy" />


<img src="../../notes-assets/rlt8.jpg" alt="RLT8.jpg" loading="lazy" />

