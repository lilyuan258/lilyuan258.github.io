---
title: π*₀.₆ / Pistar06 RECAP 学习笔记
description: >-
  RECAP 的核心思想 ： 不直接用传统 policy gradient 去更新大型 VLA，而是训练一个 价值函数
  评估机器人当前状态的好坏；再用价值函数给数据里的动作计算 advantage（优势） ；然后把 advantage 二值化成 positive /
  negative 条件 token，喂给 VLA 继续做监督学习 / fl
pubDate: '2026-06-18'
updatedDate: '2026-06-23'
tags:
  - robotics
  - VLA
  - RL
  - RECAP
  - pi0
  - pi05
  - pi06
draft: false
source: pistar06_recap_obsidian_notes.md
wordCount: 8182
readingTime: 17
---
> **Summary**
> **RECAP 的核心思想**：  
> 不直接用传统 policy gradient 去更新大型 VLA，而是训练一个**价值函数**评估机器人当前状态的好坏；再用价值函数给数据里的动作计算 **advantage（优势）**；然后把 advantage 二值化成 `positive / negative` 条件 token，喂给 VLA 继续做监督学习 / flow matching。  
> 推理时，直接让模型在 `Advantage: positive` 条件下生成动作，相当于从全部数据中提取更优策略。

---

<img src="../../notes-assets/pi0-6-1.jpg" alt="pi0.6_1.jpg" loading="lazy" />


<img src="../../notes-assets/pi0-6-2.jpg" alt="pi0.6_2.jpg" loading="lazy" />


<img src="../../notes-assets/pi0-6-3.jpg" alt="pi0.6_3.jpg" loading="lazy" width="598" height="797" />

## 0. 一句话总览

π₀ / π₀.₅ 已经解决了：

- 用 VLA 处理图像、语言、机器人状态；
- 用 flow matching 生成连续动作；
- 用离散 token 做高层语义子任务预测；
- 用 action expert 隔离连续动作生成；
- 用 Knowledge Insulation 避免动作专家破坏主干 VLM。

π*₀.₆ 在此基础上引入 RECAP：

$$
\text{Value Function}
\rightarrow
\text{Advantage}
\rightarrow
\text{Advantage Token}
\rightarrow
\text{Advantage-conditioned VLA}
$$

也就是：

1. 先让机器人执行任务，收集 autonomous rollouts；
2. 用成功/失败/时间代价训练价值函数；
3. 用价值函数计算每个动作的 advantage；
4. 把 advantage 变成 `positive / negative` 标签；
5. 训练 VLA 在给定 advantage 标签时生成对应动作；
6. 推理时指定 `Advantage: positive`，让模型更偏向好动作。

---

## 1. RECAP 到底解决什么问题？

传统机器人 RL 很难直接用于 π₀.₆ 这种大型 VLA，主要原因是：

1. **VLA 太大**，直接 policy gradient 很不稳定。
2. **动作由 flow matching / diffusion 类模型生成**，不容易得到传统 RL 所需的精确 log-likelihood。
3. **真实机器人数据昂贵**，不能像仿真里一样随便采样。
4. **已有数据很杂**：有人类 demonstration、旧策略 rollout、失败数据、专家 intervention。
5. **坏数据也有价值**，不能简单丢掉。失败轨迹能告诉模型哪些动作是 negative advantage。

RECAP 的选择是：

> 把 RL policy improvement 问题改写成条件生成问题。

也就是不直接说：

$$
\text{用 RL 梯度把策略往更优方向推}
$$

而是说：

$$
\text{训练模型学会：在 positive advantage 条件下，动作分布是什么}
$$

---

## 2. 基础符号：trajectory、policy、return

## 2.1 轨迹 trajectory

一条机器人执行轨迹写成：

$$
\tau = (o_0, a_0, o_1, a_1, \dots, o_T)
$$

符号解释：

- $\tau$：trajectory，一整段机器人执行过程，也就是一个 episode。
- $o_t$：第 $t$ 个时间步的 observation，包含图像、机器人关节状态、夹爪状态等。
- $a_t$：第 $t$ 个时间步执行的 action。对 π₀.₆ 来说，经常是 action chunk 的一部分。
- $T$：episode 的最后一个时间步。
- $t$：当前时间步索引。

---

## 2.2 策略 policy

策略写作：

$$
\pi(a_t \mid o_t)
$$

含义：在观测 $o_t$ 下，策略 $\pi$ 选择动作 $a_t$ 的概率分布。

符号解释：

- $\pi$：policy，策略模型。
- $a_t$：第 $t$ 步动作。
- $o_t$：第 $t$ 步观测。
- $\mid$：条件概率里的“给定”。

如果加上语言任务，则写成：

$$
\pi(a_t \mid o_t, \ell)
$$

其中：

- $\ell$：language command / task instruction，例如 `"make me an espresso"`。

---

## 2.3 轨迹分布

一条轨迹在策略 $\pi$ 下出现的概率可以写成：

$$
\rho_\pi(\tau)
=
p(o_0)
\prod_{t=0}^{T-1}
\pi(a_t \mid o_t)
p(o_{t+1} \mid o_t, a_t)
$$

符号解释：

- $\rho_\pi(\tau)$：策略 $\pi$ 产生轨迹 $\tau$ 的概率分布。
- $p(o_0)$：初始观测 / 初始状态的概率。
- $\prod_{t=0}^{T-1}$：从 $t=0$ 到 $T-1$ 对每一步概率连乘。
- $\pi(a_t \mid o_t)$：策略在 $o_t$ 下选中 $a_t$ 的概率。
- $p(o_{t+1} \mid o_t, a_t)$：环境动力学。执行动作 $a_t$ 后，从 $o_t$ 转移到 $o_{t+1}$ 的概率。
- $o_{t+1}$：下一时间步观测。

直觉：

$$
\text{轨迹概率}
=
\text{初始状态概率}
\times
\text{每一步选动作概率}
\times
\text{每一步环境转移概率}
$$

---

## 3. Reward、Return、Value、Advantage

## 3.1 reward

每一步 reward 记作：

$$
r_t
$$

也可以写作：

$$
r(o_t,a_t)
$$

符号解释：

- $r_t$：第 $t$ 步的奖励。
- $r(o_t,a_t)$：奖励是观测 $o_t$ 和动作 $a_t$ 的函数。
- $o_t$：当前观测。
- $a_t$：当前动作。

在 π*₀.₆ / RECAP 的实现里，reward 设计大致是：

$$
r_t =
\begin{cases}
0, & \text{if } t=T \text{ and success} \\
-C_{\mathrm{fail}}, & \text{if } t=T \text{ and failure} \\
-1, & \text{otherwise}
\end{cases}
$$

符号解释：

- $r_t$：第 $t$ 步 reward。
- $t$：当前时间步。
- $T$：episode 终止时间步。
- $\text{success}$：任务成功。
- $\text{failure}$：任务失败。
- $C_{\mathrm{fail}}$：失败惩罚系数，是一个正数；失败时 reward 为 $-C_{\mathrm{fail}}$。
- $\text{otherwise}$：既没有成功也没有失败，还在任务过程中。

直觉：

- 成功终点 reward 为 $0$；
- 失败终点 reward 是一个很大的负数；
- 每多走一步扣 $1$ 分。

所以 value 近似表达：

$$
V(o_t) \approx -(\text{距离成功还需要的步数})
$$

如果失败风险很高，value 会更负。

---

## 3.2 return

一条轨迹的累计回报：

$$
R(\tau)
=
\sum_{t=0}^{T} r_t
$$

符号解释：

- $R(\tau)$：轨迹 $\tau$ 的总回报。
- $\sum_{t=0}^{T}$：从 $t=0$ 加到 $t=T$。
- $r_t$：第 $t$ 步 reward。
- $T$：episode 结束时间。

从某个时间 $t$ 开始直到结束的 return：

$$
R_t(\tau)
=
\sum_{t'=t}^{T} r_{t'}
$$

符号解释：

- $R_t(\tau)$：轨迹 $\tau$ 中，从时间 $t$ 开始到结束的累计 reward。
- $t'$：求和用的时间索引，避免和当前时间 $t$ 混淆。
- $r_{t'}$：第 $t'$ 步 reward。

---

## 3.3 RL 目标

RL 的目标是最大化策略的期望 return：

$$
J(\pi)
=
\mathbb{E}_{\tau \sim \rho_\pi}
\left[
R(\tau)
\right]
=
\mathbb{E}_{\tau \sim \rho_\pi}
\left[
\sum_{t=0}^{T} r_t
\right]
$$

符号解释：

- $J(\pi)$：策略 $\pi$ 的性能目标。
- $\mathbb{E}$：期望，也就是对很多可能轨迹取平均。
- $\tau \sim \rho_\pi$：轨迹 $\tau$ 来自策略 $\pi$ 诱导出的轨迹分布。
- $R(\tau)$：轨迹总回报。
- $r_t$：第 $t$ 步 reward。

---

## 3.4 value function

价值函数定义为：

$$
V^\pi(o_t)
=
\mathbb{E}
\left[
\sum_{t'=t}^{T} r_{t'}
\right]
$$

如果加语言条件：

$$
V^\pi(o_t,\ell)
=
\mathbb{E}
\left[
\sum_{t'=t}^{T} r_{t'}
\mid o_t,\ell
\right]
$$

符号解释：

- $V^\pi$：策略 $\pi$ 对应的 value function。
- $V^\pi(o_t)$：在观测 $o_t$ 下，按照策略 $\pi$ 继续执行的预期未来回报。
- $V^\pi(o_t,\ell)$：在观测 $o_t$ 和语言任务 $\ell$ 下的预期未来回报。
- $\mathbb{E}$：期望。
- $\sum_{t'=t}^{T} r_{t'}$：从当前时间 $t$ 到结束的累计 reward。
- $\ell$：语言任务。

直觉：

$$
\text{value}
=
\text{当前状态离成功还有多好 / 多远}
$$

在本文 reward 设计下：

- 越接近 $0$ 越好；
- 越负越差；
- 很负通常表示失败风险高或离成功很远。

---

## 4. 图 3 中的 advantage 公式

图 3 里的核心公式是 n-step advantage：

$$
A^\pi(o_t,a_t)
=
\sum_{t'=t}^{t+N-1} r_{t'}
+
V^\pi(o_{t+N})
-
V^\pi(o_t)
$$

加上语言条件可以写成：

$$
A^\pi(o_t,a_t,\ell)
=
\sum_{t'=t}^{t+N-1} r_{t'}
+
V^\pi(o_{t+N},\ell)
-
V^\pi(o_t,\ell)
$$

符号解释：

- $A^\pi(o_t,a_t)$：advantage，表示在 $o_t$ 下采取动作 $a_t$ 比策略 $\pi$ 的平均预期好多少。
- $A^\pi(o_t,a_t,\ell)$：带语言任务 $\ell$ 的 advantage。
- $\pi$：用于定义 value 和 advantage 的参考策略。
- $o_t$：当前观测。
- $a_t$：当前动作。
- $\ell$：语言任务。
- $N$：向前看多少步。论文 post-training 实现中使用 $N=50$ 的 lookahead。
- $t'$：求和索引。
- $r_{t'}$：第 $t'$ 步 reward。
- $\sum_{t'=t}^{t+N-1} r_{t'}$：从当前时间 $t$ 到 $t+N-1$ 的实际累计 reward。
- $o_{t+N}$：$N$ 步之后的观测。
- $V^\pi(o_{t+N},\ell)$：$N$ 步之后状态的价值。
- $V^\pi(o_t,\ell)$：当前状态的价值基线。

直觉公式：

$$
\text{advantage}
=
\text{实际走了 N 步拿到的 reward}
+
\text{N 步后的剩余价值}
-
\text{当前本来预期的价值}
$$

如果：

$$
A^\pi(o_t,a_t,\ell) > 0
$$

说明这段动作让局面变得比预期更好。

如果：

$$
A^\pi(o_t,a_t,\ell) < 0
$$

说明这段动作让局面变得比预期更差。

---

## 4.1 数字例子

假设当前价值函数估计：

$$
V(o_t)=-30
$$

意思是：大概还需要 30 步才能成功。

执行 10 步后，实际每步扣 $-1$，所以：

$$
\sum r = -10
$$

10 步后价值变成：

$$
V(o_{t+10})=-15
$$

则：

$$
A=(-10)+(-15)-(-30)=+5
$$

advantage 为正，说明这 10 步让任务进展比原本预期更快。

如果 10 步后价值是：

$$
V(o_{t+10})=-25
$$

则：

$$
A=(-10)+(-25)-(-30)=-5
$$

advantage 为负，说明虽然走了 10 步，但实际进展不够，甚至比预期差。

---

## 5. 价值函数：distributional value function

RECAP 不直接训练一个标量 value，而是训练一个分布式价值函数：

$$
p_\phi(V \mid o_t,\ell) \in \Delta_B
$$

符号解释：

- $p_\phi$：带参数 $\phi$ 的概率模型。
- $\phi$：价值函数网络参数。
- $V$：价值这个随机变量。
- $p_\phi(V \mid o_t,\ell)$：给定观测 $o_t$ 和语言任务 $\ell$ 后，价值 $V$ 的概率分布。
- $o_t$：当前观测。
- $\ell$：语言任务。
- $\Delta_B$：$B$ 维概率 simplex，即 $B$ 个非负概率且和为 $1$。
- $B$：value bin 数量。论文实现中使用 $B=201$。

换句话说，模型不是输出：

$$
V=-23.7
$$

而是输出：

$$
\text{value 落在第 1 个 bin 的概率、第 2 个 bin 的概率、...、第 B 个 bin 的概率}
$$

---

## 5.1 为什么用 value distribution？

原因：

1. 真实机器人任务 reward sparse，标量回归不稳定。
2. 不同任务长度差异大，value scale 不统一。
3. 成功/失败有明显分布结构，用分类形式更稳。
4. value distribution 可以表达不确定性。

最终需要连续 value 时，再把分布转成期望。

---

## 5.2 value 训练标签

从时间 $t$ 到结束的真实 return：

$$
R_t(\tau)
=
\sum_{t'=t}^{T} r_{t'}
$$

然后把它离散化到 $B$ 个 bin：

$$
R_t^B(\tau)
$$

符号解释：

- $R_t(\tau)$：从时间 $t$ 到 episode 结束的真实累计 reward。
- $R_t^B(\tau)$：把 $R_t(\tau)$ 离散化以后得到的 bin 标签。
- $B$：bin 的数量。
- $\tau$：一条轨迹。

---

## 5.3 价值函数训练目标：Equation 1

$$
\min_\phi
\mathbb{E}_{\tau \in \mathcal{D}}
\left[
\sum_{o_t \in \tau}
H
\left(
R_t^B(\tau),
p_\phi(V \mid o_t,\ell)
\right)
\right]
$$

符号解释：

- $\min_\phi$：优化参数 $\phi$，让损失最小。
- $\phi$：value function 参数。
- $\mathbb{E}_{\tau \in \mathcal{D}}$：对数据集 $\mathcal{D}$ 里的轨迹取平均。
- $\mathcal{D}$：当前训练数据集。
- $\tau$：一条轨迹。
- $\sum_{o_t \in \tau}$：对轨迹中每个观测 $o_t$ 求和。
- $o_t$：第 $t$ 步观测。
- $H(\cdot,\cdot)$：cross-entropy loss，交叉熵损失。
- $R_t^B(\tau)$：真实 return 离散化后的目标 bin。
- $p_\phi(V \mid o_t,\ell)$：价值函数预测的 value-bin 分布。
- $\ell$：语言任务。

这本质是一个 Monte Carlo value estimator：

$$
\text{用真实 rollout 后面发生了什么，监督当前状态的 value}
$$

它不是 TD 学习，也不是显式 Q-learning。

---

## 5.4 从 value distribution 变回连续 value

训练完后，连续 value 通过期望得到：

$$
V^{\pi_{\mathrm{ref}}}(o_t,\ell)
=
\sum_{b \in [0,B]}
p_\phi(V=b \mid o_t,\ell)v(b)
$$

符号解释：

- $V^{\pi_{\mathrm{ref}}}(o_t,\ell)$：参考策略 $\pi_{\mathrm{ref}}$ 下，观测 $o_t$ 和任务 $\ell$ 的连续 value。
- $\pi_{\mathrm{ref}}$：reference policy，参考策略。实际是当前数据集所代表的行为策略混合体。
- $b$：value bin 的索引。
- $[0,B]$：所有 value bin 的索引集合。
- $p_\phi(V=b \mid o_t,\ell)$：value 落在第 $b$ 个 bin 的概率。
- $v(b)$：第 $b$ 个 bin 对应的实际 value 数值。
- $\sum$：对所有 bin 求和。

直觉：

$$
\text{连续 value}
=
\sum
\text{每个 bin 的概率}
\times
\text{该 bin 对应数值}
$$

---

## 6. reference policy 是什么？

RECAP 中常用：

$$
\pi_{\mathrm{ref}}
$$

它表示 reference policy。

但这里的 reference policy 不是一个单独模型，而是数据集代表的行为分布。

数据集可能包含：

- 人类 demonstration；
- 旧策略 rollout；
- 当前策略 autonomous rollout；
- 专家 intervention；
- 成功数据；
- 失败数据。

所以可以理解为：

$$
\pi_{\mathrm{ref}}
=
\text{当前训练数据集中所有行为来源的混合策略}
$$

RECAP 训练的是：

$$
V^{\pi_{\mathrm{ref}}}
$$

也就是“在当前数据行为分布下的 value”。

---

## 7. advantage 二值化：positive / negative

计算出 advantage 后，不直接把连续 advantage 喂给模型，而是二值化：

$$
I_t
=
\mathbf{1}
\left(
A^{\pi_{\mathrm{ref}}}(o_t,a_t,\ell) > \epsilon_\ell
\right)
$$

符号解释：

- $I_t$：第 $t$ 步的 improvement indicator。
- $\mathbf{1}(\cdot)$：indicator function。条件为真则为 $1$，否则为 $0$。
- $A^{\pi_{\mathrm{ref}}}(o_t,a_t,\ell)$：相对于参考策略 $\pi_{\mathrm{ref}}$ 的 advantage。
- $o_t$：当前观测。
- $a_t$：当前动作。
- $\ell$：语言任务。
- $\epsilon_\ell$：任务 $\ell$ 对应的 advantage 阈值。
- $\pi_{\mathrm{ref}}$：参考策略。

如果：

$$
A^{\pi_{\mathrm{ref}}}(o_t,a_t,\ell) > \epsilon_\ell
$$

则：

$$
I_t=1
$$

对应：

```text
Advantage: positive
```

如果：

$$
A^{\pi_{\mathrm{ref}}}(o_t,a_t,\ell) \leq \epsilon_\ell
$$

则：

$$
I_t=0
$$

对应：

```text
Advantage: negative
```

---

## 7.1 为什么要阈值 $\epsilon_\ell$？

不是所有 $A>0$ 都一定要标 positive，因为不同任务：

- 时间长度不同；
- 难度不同；
- reward scale 不同；
- demonstration 质量不同；
- autonomous rollout 质量不同。

所以需要任务相关阈值：

$$
\epsilon_\ell
$$

它控制：

$$
\text{多好的动作才算 positive advantage}
$$

直觉：

- $\epsilon_\ell$ 低：positive 样本多，模型更保守。
- $\epsilon_\ell$ 高：positive 样本少，模型更挑剔，但可能不稳定。
- 合适的 $\epsilon_\ell$：既能筛出好动作，又不至于正样本太少。

---

## 8. policy extraction：从 value 到更优 policy

## 8.1 为什么叫 policy extraction？

已经有一个数据集：

$$
\mathcal{D}_{\pi_{\mathrm{ref}}}
$$

里面既有好动作，也有坏动作。

传统方法可能会：

- 只保留高 reward 数据；
- 给高 advantage 数据更大权重；
- 对 policy 做梯度更新。

RECAP 的方法是：

> 让模型学习一个条件动作分布：给定 `I=True` 时动作应该长什么样。

因此推理时输入：

```text
Advantage: positive
```

模型就会采样更接近 positive-advantage 数据的动作。

这叫 policy extraction，因为它不是直接创造一个全新策略，而是从参考数据分布中提取更优的条件分布。

---

## 8.2 regularized policy improvement 形式

一个常见的正则化 RL 改进形式是：

$$
\hat{\pi}(a \mid o)
\propto
\pi_{\mathrm{ref}}(a \mid o)
\exp
\left(
\frac{
A^{\pi_{\mathrm{ref}}}(o,a)
}{
\beta
}
\right)
$$

符号解释：

- $\hat{\pi}$：改进后的目标策略。
- $a$：动作。
- $o$：观测。
- $\pi_{\mathrm{ref}}(a \mid o)$：参考策略在 $o$ 下选择 $a$ 的概率。
- $A^{\pi_{\mathrm{ref}}}(o,a)$：动作 $a$ 相对于参考策略的 advantage。
- $\beta$：温度 / 正则强度参数。
- $\exp(\cdot)$：指数函数。
- $\propto$：正比于。右边还需要归一化成合法概率分布。

直觉：

- advantage 越大，动作概率被放大越多；
- advantage 越小，动作概率被压低；
- $\beta$ 控制放大程度。

---

## 9. Equation 2：用 Bayes rule 得到 advantage-conditioned policy

RECAP 使用一种与 CFGRL / classifier-free guidance 思路相近的写法。

首先定义 improvement event：

$$
I
$$

其中：

- $I$：动作是 improvement action 的事件。
- 可以粗略理解为：$I=True$ 表示 positive advantage。

假设：

$$
p(I \mid A^{\pi_{\mathrm{ref}}}(o,a))
$$

表示动作 $a$ 在状态 $o$ 下成为 improvement action 的概率。

可以写出：

$$
\hat{\pi}(a \mid o)
\propto
\pi_{\mathrm{ref}}(a \mid o)
p(I \mid A^{\pi_{\mathrm{ref}}}(o,a))^\beta
$$

再用 Bayes rule，可得近似关系：

$$
p(I \mid A^{\pi_{\mathrm{ref}}}(o,a))
\propto
\frac{
\pi_{\mathrm{ref}}(a \mid I,o)
}{
\pi_{\mathrm{ref}}(a \mid o)
}
$$

注意这里写的是 $\propto$，因为严格来说会差一个只依赖 $o$ 的归一化常数；在策略公式中这个常数会被整体归一化吸收。

加入语言条件后，得到论文中的 Equation 2：

$$
\hat{\pi}(a \mid o,\ell)
\propto
\pi_{\mathrm{ref}}(a \mid o,\ell)
\left(
\frac{
\pi_{\mathrm{ref}}(a \mid I,o,\ell)
}{
\pi_{\mathrm{ref}}(a \mid o,\ell)
}
\right)^\beta
$$

符号解释：

- $\hat{\pi}(a \mid o,\ell)$：改进后的策略，在观测 $o$ 和语言任务 $\ell$ 下选择动作 $a$ 的分布。
- $\pi_{\mathrm{ref}}(a \mid o,\ell)$：参考策略中，不带 improvement 条件的动作分布。
- $\pi_{\mathrm{ref}}(a \mid I,o,\ell)$：参考策略中，带 improvement 条件 $I$ 的动作分布。
- $I$：improvement indicator，表示动作来自 positive-advantage 条件。
- $o$：观测。
- $a$：动作。
- $\ell$：语言任务。
- $\beta$：guidance / sharpening 强度。
- $\left(\frac{\pi_{\mathrm{ref}}(a \mid I,o,\ell)}{\pi_{\mathrm{ref}}(a \mid o,\ell)}\right)$：动作 $a$ 在 positive 数据中相对普通数据有多突出。
- $\propto$：正比于，最后仍需归一化。

---

## 9.1 特殊情况：$\beta=1$

当：

$$
\beta=1
$$

Equation 2 变成：

$$
\hat{\pi}(a \mid o,\ell)
\propto
\pi_{\mathrm{ref}}(a \mid o,\ell)
\frac{
\pi_{\mathrm{ref}}(a \mid I,o,\ell)
}{
\pi_{\mathrm{ref}}(a \mid o,\ell)
}
$$

约掉 $\pi_{\mathrm{ref}}(a \mid o,\ell)$：

$$
\hat{\pi}(a \mid o,\ell)
=
\pi_{\mathrm{ref}}(a \mid I,o,\ell)
$$

关键结论：

> 如果能训练一个模型表示 $\pi(a \mid I,o,\ell)$，那么推理时给 $I=True$，就等价于采样改进策略。

这就是 advantage conditioning 的数学理由。

---

## 10. Equation 3：advantage-conditioned 策略训练目标

论文策略训练目标：

$$
\min_\theta
\mathbb{E}_{\mathcal{D}_{\pi_{\mathrm{ref}}}}
\left[
-
\log \pi_\theta(a_t \mid o_t,\ell)
-
\alpha
\log \pi_\theta(a_t \mid I_t,o_t,\ell)
\right]
$$

符号解释：

- $\min_\theta$：优化策略参数 $\theta$。
- $\theta$：VLA policy 参数。
- $\mathbb{E}_{\mathcal{D}_{\pi_{\mathrm{ref}}}}$：对数据集 $\mathcal{D}_{\pi_{\mathrm{ref}}}$ 中样本取平均。
- $\mathcal{D}_{\pi_{\mathrm{ref}}}$：由参考策略产生的数据集，包括 demonstrations、rollouts、interventions 等。
- $a_t$：第 $t$ 步动作。
- $o_t$：第 $t$ 步观测。
- $\ell$：语言任务。
- $I_t$：第 $t$ 步 advantage indicator。
- $\pi_\theta(a_t \mid o_t,\ell)$：无 advantage 条件的普通策略分布。
- $\pi_\theta(a_t \mid I_t,o_t,\ell)$：带 advantage 条件的策略分布。
- $-\log \pi_\theta(\cdot)$：负对数似然，越小表示模型越愿意生成数据中的动作。
- $\alpha$：trade-off hyperparameter，控制 advantage-conditioned 项的重要性。

---

## 10.1 第一项：普通模仿

第一项是：

$$
-\log \pi_\theta(a_t \mid o_t,\ell)
$$

作用：

$$
\text{让模型学会普通数据分布}
$$

它使用所有数据，不管 positive 还是 negative。

---

## 10.2 第二项：带 advantage 条件的模仿

第二项是：

$$
-\alpha
\log \pi_\theta(a_t \mid I_t,o_t,\ell)
$$

作用：

$$
\text{让模型学会在不同 } I_t \text{ 条件下生成不同动作分布}
$$

如果 $I_t=1$，模型学：

```text
Advantage: positive 下，这个状态应该如何动作
```

如果 $I_t=0$，模型学：

```text
Advantage: negative 下，这个状态会出现什么动作
```

所以 RECAP 并不丢弃失败数据。失败数据也能帮助模型理解 negative-conditioned 分布。

---

## 11. 人类 intervention 怎么处理？

专家 intervention 是人类在机器人执行过程中提供的纠正动作。

论文采用一个实用假设：

$$
\text{human correction} \Rightarrow I_t=True
$$

也就是把专家纠正动作强制标成 positive advantage。

原因：

- 专家通常在机器人将要失败或已经偏离时介入；
- intervention 动作是为了纠正错误；
- 相比让价值函数判断，直接设为 positive 更稳。

---

## 12. π₀.₆ 模型架构

π₀.₆ 可以写成：

$$
\pi_\theta(a_{t:t+H}, \hat{\ell} \mid o_t,\ell)
$$

符号解释：

- $\pi_\theta$：参数为 $\theta$ 的 VLA policy。
- $a_{t:t+H}$：从时间 $t$ 到 $t+H$ 的连续 action chunk。
- $H$：action chunk horizon。
- $\hat{\ell}$：模型预测的高层子任务文本，例如 `"pick up the coffee cup"`。
- $o_t$：当前观测。
- $\ell$：输入语言任务。

---

## 12.1 观测输入

论文写作：

$$
o_t =
[X_t^1,\dots,X_t^n,q_t]
$$

符号解释：

- $o_t$：当前 observation。
- $X_t^1,\dots,X_t^n$：第 $t$ 步来自 $n$ 个摄像头的图像。
- $n$：摄像头数量。
- $q_t$：机器人配置，例如关节角、末端执行器状态、夹爪状态等。
- $[\cdot]$：拼接 / 组合成整体观测。

---

## 12.2 语言输入

语言输入写作：

$$
\ell = \ell_t + s
$$

符号解释：

- $\ell$：最终输入模型的语言条件。
- $\ell_t$：主任务 prompt，例如 `"make me an espresso"`。
- $s$：额外 metadata，例如任务风格、环境信息、advantage token 等。
- $+$：这里不是数学加法，而是语言输入拼接。

---

## 12.3 π₀.₆ 输出

π₀.₆ 输出两类东西。

第一类：高层离散 token：

$$
\hat{\ell}
$$

含义：预测下一步高层子任务，例如：

```text
pick up the coffee cup
```

第二类：连续动作 chunk：

$$
a_{t:t+H}
$$

含义：一段未来连续控制动作，包括关节角目标、夹爪命令等。

因为动作生成在 $\hat{\ell}$ 之后，所以：

$$
a_{t:t+H}
\text{ 被 }
\hat{\ell}
\text{ 条件化}
$$

即：

$$
\text{先想高层子任务，再生成低层动作}
$$

---

## 12.4 action expert

π₀.₆ 使用一个 action expert 来生成连续动作。

特点：

- action expert 是专门负责动作生成的一组参数；
- 论文中 action expert 规模为 860M；
- 连续动作通过 flow matching 训练；
- action expert 可以 attend 到 VLM 主干激活；
- 但通过 stop gradient 避免动作损失破坏主干 VLM 表示。

直觉：

$$
\text{VLM 主干负责理解世界}
$$

$$
\text{action expert 负责把理解转成连续动作}
$$

---

## 13. π₀.₆ 的概率分解

π₀.₆ 同时预测：

- 高层子任务 $\hat{\ell}$；
- 离散动作 token $a^\ell_{t:t+H}$；
- 连续动作 chunk $a_{t:t+H}$。

总 log-likelihood 可分解为：

$$
\begin{aligned}
&
\log \pi_\theta
(a_{t:t+H},a^\ell_{t:t+H},\hat{\ell}
\mid
o_t,\ell)
\\
&=
\log \pi_\theta(\hat{\ell}\mid o_t,\ell)
\\
&\quad+
\log \pi_\theta(a^\ell_{t:t+H}\mid o_t,\ell,\hat{\ell})
\\
&\quad+
\log \pi_\theta(a_{t:t+H}\mid o_t,\ell,\hat{\ell})
\end{aligned}
$$

符号解释：

- $\log$：对数。
- $\pi_\theta$：参数为 $\theta$ 的策略模型。
- $a_{t:t+H}$：连续动作 chunk。
- $a^\ell_{t:t+H}$：离散化后的动作 token。注意这里的上标 $\ell$ 是论文记号，不等于语言任务本身。
- $\hat{\ell}$：预测的高层子任务文本。
- $o_t$：当前观测。
- $\ell$：语言任务。
- $\mid$：条件。
- 第一项 $\log \pi_\theta(\hat{\ell}\mid o_t,\ell)$：预测高层子任务的 log-likelihood。
- 第二项 $\log \pi_\theta(a^\ell_{t:t+H}\mid o_t,\ell,\hat{\ell})$：预测离散动作 token 的 log-likelihood。
- 第三项 $\log \pi_\theta(a_{t:t+H}\mid o_t,\ell,\hat{\ell})$：预测连续动作 chunk 的 log-likelihood。

---

## 14. π*₀.₆：advantage token 插在哪里？

π*₀.₆ 在 π₀.₆ 基础上增加：

$$
I_t
$$

也就是 advantage indicator。

输入形式大概是：

```text
Task: make me an espresso
Subtask: pick up the coffee cup
Advantage: positive
```

或者：

```text
Task: make me an espresso
Subtask: pick up the coffee cup
Advantage: negative
```

关键细节：

> advantage token 放在高层子任务 $\hat{\ell}$ 之后、动作生成之前。

因此：

- 它主要影响低层动作；
- 不主要影响高层子任务预测；
- 模型先决定“要做什么”，再根据 advantage 条件决定“怎么做更好”。

形式上可以写成：

$$
\pi_\theta(a_{t:t+H} \mid I_t,o_t,\ell,\hat{\ell})
$$

符号解释：

- $a_{t:t+H}$：连续动作 chunk。
- $I_t$：positive / negative advantage indicator。
- $o_t$：当前观测。
- $\ell$：原始语言任务。
- $\hat{\ell}$：高层子任务。

---

## 15. flow matching 与 advantage conditioning

连续动作无法像普通分类 token 一样直接用 softmax likelihood 训练，所以用 flow matching。

论文的训练形式可以写成：

$$
\begin{aligned}
&
\log \pi_\theta(a_{t:t+H},a^\ell_{t:t+H}
\mid
I_t,o_t,\ell,\hat{\ell})
\\
&\ge
\mathbb{E}_{\eta,\omega}
\left[
\log p_\theta(a^\ell_{t:t+H}
\mid
I_t,o_t,\ell,\hat{\ell})
-
\alpha_\eta
\left\|
\omega-a_{t:t+H}
-
f_\theta(a^{\eta,\omega}_{t:t+H},I_t,o_t,\ell,\hat{\ell})
\right\|^2
\right]
\end{aligned}
$$

其中加噪动作：

$$
a^{\eta,\omega}_{t:t+H}
=
\eta a_{t:t+H}
+
(1-\eta)\omega
$$

符号解释：

- $\log \pi_\theta(\cdot)$：连续动作和离散动作的联合 log-likelihood。
- $\ge$：右边是训练 surrogate / 下界，不是精确 likelihood。
- $\mathbb{E}_{\eta,\omega}$：对 flow 时间 $\eta$ 和噪声 $\omega$ 取期望。
- $\eta$：flow matching 时间，通常在 $[0,1]$。
- $\omega$：噪声动作，通常来自高斯分布。
- $p_\theta(a^\ell_{t:t+H}\mid I_t,o_t,\ell,\hat{\ell})$：离散动作 token 的概率模型。
- $\alpha_\eta$：flow matching loss 权重，可能随 $\eta$ 变化。
- $\|\cdot\|^2$：平方 L2 范数。
- $f_\theta$：flow matching action expert 预测的向量场 / denoising 目标。
- $a^{\eta,\omega}_{t:t+H}$：真实动作和噪声动作的插值。
- $a_{t:t+H}$：真实连续动作 chunk。
- $I_t$：advantage indicator。
- $o_t$：观测。
- $\ell$：语言任务。
- $\hat{\ell}$：高层子任务。

关键点：

$$
I_t
\text{ 也作为条件输入给 flow matching 动作专家}
$$

所以 continuous action generation 本身会受到：

```text
Advantage: positive / negative
```

的影响。

---

## 16. classifier-free guidance 视角

因为模型同时学：

$$
\pi_\theta(a \mid o,\ell)
$$

和：

$$
\pi_\theta(a \mid I,o,\ell)
$$

所以可以做类似 classifier-free guidance 的强化采样：

$$
\hat{\pi}(a \mid o,\ell)
\propto
\pi_{\mathrm{ref}}(a \mid o,\ell)
\left(
\frac{
\pi_{\mathrm{ref}}(a \mid I,o,\ell)
}{
\pi_{\mathrm{ref}}(a \mid o,\ell)
}
\right)^\beta
$$

对应 score / gradient 形式：

$$
\nabla_a \log \pi_\theta(a \mid o,\ell)
+
\beta
\left(
\nabla_a \log \pi_\theta(a \mid I,o,\ell)
-
\nabla_a \log \pi_\theta(a \mid o,\ell)
\right)
$$

符号解释：

- $\nabla_a$：对动作 $a$ 求梯度。
- $\log \pi_\theta(a \mid o,\ell)$：无 advantage 条件的 log probability。
- $\log \pi_\theta(a \mid I,o,\ell)$：带 advantage 条件的 log probability。
- $\beta$：guidance strength。
- 括号里的差值：positive 条件相对于普通分布，想把动作往哪个方向推。
- $\hat{\pi}$：guided / improved policy。

直觉：

- $\beta=1$：基本使用 positive-conditioned policy。
- $\beta>1$：更强地偏向 positive 分布。
- $\beta$ 太大：动作可能过于 aggressive，甚至离开安全数据分布。

---

## 17. Algorithm 1：RECAP 完整流程

<img src="../../notes-assets/2026-06-23-174337.png" alt="屏幕截图 2026-06-23 174337.png" loading="lazy" />

算法输入：

$$
\mathcal{D}_{\mathrm{demo}}
$$

符号解释：

- $\mathcal{D}_{\mathrm{demo}}$：multi-task demonstration dataset，多任务人类示范数据集。

---

## 17.1 预训练阶段

第一步：

$$
\text{Train } V_{\mathrm{pre}} \text{ on } \mathcal{D}_{\mathrm{demo}} \text{ using Eq. 1}
$$

含义：

用所有 demonstration 数据训练一个预训练价值函数。

符号：

- $V_{\mathrm{pre}}$：pre-trained value function。
- $\mathcal{D}_{\mathrm{demo}}$：示范数据。
- Eq. 1：distributional value function training objective。

第二步：

$$
\text{Train } \pi_{\mathrm{pre}} \text{ on } \mathcal{D}_{\mathrm{demo}} \text{ using Eq. 3 and } V_{\mathrm{pre}}
$$

含义：

用 $V_{\mathrm{pre}}$ 给 demonstration 数据算 advantage，再训练 advantage-conditioned policy。

符号：

- $\pi_{\mathrm{pre}}$：pre-trained advantage-conditioned VLA policy。
- $V_{\mathrm{pre}}$：用于计算 advantage 的价值函数。

---

## 17.2 针对具体技能初始化

对每个技能 / 任务：

$$
\ell
$$

初始化任务数据集：

$$
\mathcal{D}_\ell
$$

符号：

- $\ell$：某个具体语言任务 / 技能。
- $\mathcal{D}_\ell$：该任务的数据集。

训练初始任务专用 value：

$$
V_\ell^0
$$

训练初始任务专用 policy：

$$
\pi_\ell^0
$$

符号解释：

- $V_\ell^0$：任务 $\ell$ 的第 $0$ 轮价值函数。
- $\pi_\ell^0$：任务 $\ell$ 的第 $0$ 轮策略。

---

## 17.3 在线迭代阶段

对于：

$$
k=1,\dots,K
$$

循环：

### Step 1：收集数据

$$
\text{Collect data with } \pi_\ell^{k-1}, \text{ add it to } \mathcal{D}_\ell
$$

符号解释：

- $k$：当前迭代轮次。
- $K$：总迭代轮数。
- $\pi_\ell^{k-1}$：上一轮任务策略。
- $\mathcal{D}_\ell$：当前累计任务数据集。

含义：

让上一轮策略去真实机器人上执行任务，收集新的 rollout，并加入数据集。

---

### Step 2：更新 value function

$$
\text{Train } V_\ell^k \text{ from } V_{\mathrm{pre}} \text{ on } \mathcal{D}_\ell \text{ using Eq. 1}
$$

符号解释：

- $V_\ell^k$：任务 $\ell$ 第 $k$ 轮价值函数。
- $V_{\mathrm{pre}}$：预训练价值函数初始化。
- $\mathcal{D}_\ell$：累计任务数据集。

---

### Step 3：更新 policy

$$
\text{Train } \pi_\ell^k \text{ from } \pi_{\mathrm{pre}} \text{ on } \mathcal{D}_\ell \text{ using Eq. 3 and } V_\ell^k
$$

符号解释：

- $\pi_\ell^k$：任务 $\ell$ 第 $k$ 轮策略。
- $\pi_{\mathrm{pre}}$：预训练 advantage-conditioned policy 初始化。
- $V_\ell^k$：当前轮 value function，用来重新计算 advantage。
- Eq. 3：advantage-conditioned 策略训练目标。

---

## 17.4 算法本质

RECAP 可以压缩成三件事反复做：

$$
\text{Collect Data}
\rightarrow
\text{Train Value Function}
\rightarrow
\text{Train Advantage-conditioned Policy}
$$

每一轮的关键区别只是：

$$
\mathcal{D}_\ell
\text{ 变大了}
$$

数据从一开始的人类 demonstration，逐渐加入：

- autonomous rollout；
- 成功尝试；
- 失败尝试；
- human correction；
- old policy data；
- new policy data。

---

## 18. 图 3 的整体数据流

图 3 可以理解为上下两部分。

## 18.1 上半部分：π*₀.₆ VLA

输入：

$$
(o_t,\ell,\hat{\ell},I_t)
$$

输出：

$$
a_{t:t+H}
$$

含义：

- 看图像、机器人状态；
- 看语言任务；
- 预测或使用高层子任务；
- 加上 advantage 条件；
- 由 action expert 生成连续动作。

---

## 18.2 下半部分：value function

输入：

$$
(o_t,\ell)
$$

输出：

$$
p_\phi(V \mid o_t,\ell)
$$

再得到：

$$
V^{\pi_{\mathrm{ref}}}(o_t,\ell)
$$

然后结合 rollout 中的未来状态和 reward，计算：

$$
A^{\pi_{\mathrm{ref}}}(o_t,a_t,\ell)
=
\sum_{t'=t}^{t+N-1} r_{t'}
+
V^{\pi_{\mathrm{ref}}}(o_{t+N},\ell)
-
V^{\pi_{\mathrm{ref}}}(o_t,\ell)
$$

再二值化：

$$
I_t
=
\mathbf{1}
\left(
A^{\pi_{\mathrm{ref}}}(o_t,a_t,\ell) > \epsilon_\ell
\right)
$$

最后把 $I_t$ 作为 token 喂回 VLA 训练。

---

## 19. 和 π₀ / π₀.₅ 的关系

## 19.1 π₀

π₀ 的核心：

$$
\text{VLA} + \text{flow matching continuous action generation}
$$

它主要解决：

- 图像语言输入；
- 连续动作输出；
- 机器人动作生成。

---

## 19.2 π₀.₅

π₀.₅ 在 π₀ 上增强：

- 高层语义子任务预测；
- 离散 token 预训练；
- continuous action expert；
- FAST action token；
- Knowledge Insulation；
- web data + robot data co-training。

可以理解为：

$$
\text{pi0.5}
=
\text{pi0}
+
\text{更强的高层语义和动作专家机制}
$$

---

## 19.3 π₀.₆

π₀.₆ 是 π₀.₅ 的演进：

- 更多多机器人数据；
- 更强的 VLM backbone；
- 更大的 action expert；
- 继续使用 flow matching；
- 继续输出高层子任务；
- 支持更复杂的 metadata。

---

## 19.4 π*₀.₆

π*₀.₆ 是：

$$
\text{pi0.6}
+
\text{RECAP advantage conditioning}
$$

也就是：

$$
\pi_\theta(a_{t:t+H} \mid I_t,o_t,\ell,\hat{\ell})
$$

相比 π₀.₆，多了：

$$
I_t
$$

这个 positive / negative advantage 条件。

---

## 20. 容易混淆的点

## 20.1 value function 不是 reward model

reward 是规则给的：

$$
r_t =
\begin{cases}
0, & \text{success at terminal} \\
-C_{\mathrm{fail}}, & \text{failure at terminal} \\
-1, & \text{otherwise}
\end{cases}
$$

value function 是模型学出来的：

$$
V(o_t,\ell)
\approx
\text{未来累计 reward}
$$

---

## 20.2 value function 不是 Q-function

Q-function 通常是：

$$
Q(o,a)
$$

它直接输入动作。

RECAP 的 value function 是：

$$
V(o,\ell)
$$

它不直接输入候选动作。

advantage 是通过实际轨迹估计：

$$
A(o_t,a_t,\ell)
=
\sum r
+
V(o_{t+N},\ell)
-
V(o_t,\ell)
$$

不是通过：

$$
Q(o,a)-V(o)
$$

显式计算。

---

## 20.3 positive advantage 不是人工标注的“好动作”

大多数情况下：

$$
I_t
=
\mathbf{1}
(A>\epsilon_\ell)
$$

是价值函数计算出来的。

只有 human correction 会被强制设为：

$$
I_t=True
$$

---

## 20.4 RECAP 不丢弃失败数据

失败数据仍然有用，因为它们训练：

$$
\pi_\theta(a_t \mid I_t=0,o_t,\ell)
$$

也就是 negative-conditioned 行为分布。

推理时选择：

$$
I_t=1
$$

即可偏向好动作。

---

## 20.5 $\epsilon_\ell$ 和 $\beta$ 是两个不同旋钮

$\epsilon_\ell$ 控制训练标签：

$$
I_t
=
\mathbf{1}(A>\epsilon_\ell)
$$

它决定哪些数据算 positive。

$\beta$ 控制推理 guidance 强度：

$$
\hat{\pi}(a \mid o,\ell)
\propto
\pi_{\mathrm{ref}}(a \mid o,\ell)
\left(
\frac{
\pi_{\mathrm{ref}}(a \mid I,o,\ell)
}{
\pi_{\mathrm{ref}}(a \mid o,\ell)
}
\right)^\beta
$$

它决定推理时多强地偏向 positive 分布。

---

## 21. 最小心智模型

可以把 RECAP 想成一个三模块系统：

## 21.1 价值函数

回答问题：

```text
当前画面离任务成功还有多远？
```

数学上：

$$
V^{\pi_{\mathrm{ref}}}(o_t,\ell)
$$

---

## 21.2 advantage 计算器

回答问题：

```text
这段动作让任务进展比预期更好吗？
```

数学上：

$$
A^\pi(o_t,a_t,\ell)
=
\sum_{t'=t}^{t+N-1}r_{t'}
+
V^\pi(o_{t+N},\ell)
-
V^\pi(o_t,\ell)
$$

---

## 21.3 advantage-conditioned VLA

回答问题：

```text
如果我要求 positive advantage，你应该怎么动作？
```

数学上：

$$
\pi_\theta(a_t \mid I_t=1,o_t,\ell)
$$

---

## 22. 一句话最终总结

π*₀.₆ 的 RECAP 把机器人 RL 的策略改进问题转化成了条件生成建模问题：

$$
\boxed{
\text{用价值函数评价数据}
\rightarrow
\text{用 advantage 标记好坏}
\rightarrow
\text{训练 VLA 理解 positive / negative 条件}
\rightarrow
\text{推理时指定 positive 条件生成更优动作}
}
$$

它避免了直接对大型 flow-matching VLA 做困难的 policy gradient，同时能利用真实机器人 autonomous experience、人类 corrections、成功数据和失败数据。

---

## 23. 符号速查表

| 符号 | 含义 |
|---|---|
| $\tau$ | 一条 trajectory / episode |
| $o_t$ | 第 $t$ 步 observation |
| $a_t$ | 第 $t$ 步 action |
| $a_{t:t+H}$ | 从 $t$ 到 $t+H$ 的 action chunk |
| $H$ | action chunk horizon |
| $T$ | episode 终止时间 |
| $\ell$ | 语言任务 / task command |
| $\ell_t$ | 主任务 prompt |
| $s$ | metadata / 附加语言输入 |
| $\hat{\ell}$ | 模型预测的高层子任务文本 |
| $X_t^i$ | 第 $i$ 个摄像头在时间 $t$ 的图像 |
| $q_t$ | 机器人 proprioception / configuration |
| $\pi$ | policy |
| $\pi_\theta$ | 参数为 $\theta$ 的策略 |
| $\pi_{\mathrm{ref}}$ | reference policy / 数据行为分布 |
| $\hat{\pi}$ | 改进后的策略 |
| $r_t$ | 第 $t$ 步 reward |
| $R(\tau)$ | 整条轨迹累计 return |
| $R_t(\tau)$ | 从时间 $t$ 到结束的 return |
| $R_t^B(\tau)$ | 离散化后的 return bin 标签 |
| $V^\pi(o_t,\ell)$ | 策略 $\pi$ 下的 value function |
| $p_\phi(V\mid o_t,\ell)$ | value-bin 分布预测 |
| $\phi$ | value function 参数 |
| $B$ | value bin 数量 |
| $\Delta_B$ | $B$ 维概率 simplex |
| $v(b)$ | 第 $b$ 个 value bin 对应的数值 |
| $A^\pi(o_t,a_t,\ell)$ | advantage |
| $I_t$ | improvement / advantage indicator |
| $\epsilon_\ell$ | 任务相关 advantage 阈值 |
| $\alpha$ | Eq. 3 中条件项权重 |
| $\beta$ | guidance / sharpening 强度 |
| $\eta$ | flow matching 时间 |
| $\omega$ | 高斯噪声动作 |
| $f_\theta$ | flow matching action expert 输出 |
| $\mathcal{D}$ | 数据集 |
| $\mathcal{D}_{\mathrm{demo}}$ | 多任务 demonstration 数据集 |
| $\mathcal{D}_\ell$ | 任务 $\ell$ 的累计数据集 |
| $V_{\mathrm{pre}}$ | 预训练 value function |
| $\pi_{\mathrm{pre}}$ | 预训练 advantage-conditioned policy |
| $V_\ell^k$ | 任务 $\ell$ 第 $k$ 轮 value function |
| $\pi_\ell^k$ | 任务 $\ell$ 第 $k$ 轮 policy |
| $K$ | 总迭代轮数 |
| $k$ | 当前迭代轮次 |

---

## 24. Obsidian 渲染提示

本笔记使用 Obsidian / MathJax 兼容写法：

- 行内公式使用 `$...$`；
- 块级公式使用 `$$...$$`；
- 多行公式使用 `\begin{aligned}...\end{aligned}`；
- 分段函数使用 `\begin{cases}...\end{cases}`；
- 下标使用 `_`，例如 `$o_t$`；
- 上标使用 `^`，例如 `$V^\pi$`；
- 花体集合使用 `\mathcal{D}`；
- 罗马体文字使用 `\mathrm{ref}`、`\mathrm{demo}`；
- 不建议把复杂公式放进 Markdown 表格里，Obsidian 有时会渲染不稳定。
