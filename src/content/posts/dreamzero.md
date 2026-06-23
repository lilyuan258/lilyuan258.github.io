---
title: DreamZero 精读笔记
description: >-
  DreamZero 的核心思想是：以预训练视频生成模型作为世界动态骨干，在同一个因果 DiT 中联合生成未来视频 latent
  与未来动作；执行动作后，不继续相信模型预测的视频，而是将真实观测重新编码并写回 KV Cache，从而形成闭环、带视觉记忆的机器人策略。
pubDate: '2026-06-23'
updatedDate: '2026-06-23'
tags:
  - embodied-ai
  - robot-learning
  - world-action-model
  - video-diffusion
  - flow-matching
  - diffusion-transformer
  - closed-loop-control
  - paper-reading
draft: false
source: DreamZero_精读学习笔记_Obsidian.md
wordCount: 8335
readingTime: 17
---
> **abstract**
> DreamZero 的核心思想是：以预训练视频生成模型作为世界动态骨干，在同一个因果 DiT 中联合生成未来视频 latent 与未来动作；执行动作后，不继续相信模型预测的视频，而是将真实观测重新编码并写回 KV Cache，从而形成闭环、带视觉记忆的机器人策略。

---


<img src="../../notes-assets/2026-06-23-174110.png" alt="屏幕截图 2026-06-23 174110.png" loading="lazy" />


<img src="../../notes-assets/20260623174717-517-1.jpg" alt="微信图片_20260623174717_517_1.jpg" loading="lazy" width="508" />


<img src="../../notes-assets/20260623174720-518-1.jpg" alt="微信图片_20260623174720_518_1.jpg" loading="lazy" width="509" />

## 1. 一句话理解 DreamZero

DreamZero 可以概括为：

$$
\text{真实视觉历史}+\text{语言}+\text{本体状态}
\longrightarrow
\text{未来视频 latent}+\text{未来动作}
$$

但它与传统“先生成视频，再运行独立逆动力学模型”的串联系统不同。DreamZero 使用的是一个统一的 Joint Video-Action DiT，在同一个模型、同一个 Flow Matching 目标下联合去噪视频和动作。

训练时，未来视频是动作学习的结构化监督；推理时，未来视频是动作生成的 latent scaffold；执行后，预测视频会被真实观测替换。

---

## 2. DreamZero 不是哪类方法

### 2.1 不是两个独立模型串联

DreamZero 的概率分解可以写成“视频预测 × 逆动力学”，但工程实现不是：

1. 先运行视频模型；
2. 再运行另一个 IDM；
3. 最后输出动作。

实际是一个统一的 Joint Video-Action DiT，共享表示、上下文和优化目标。

### 2.2 不是传统显式规划器

它没有：

- 候选轨迹搜索；
- 奖励模型打分；
- MPC；
- 树搜索；
- 多条未来轨迹比较。

它更接近一种 amortized policy：

$$
\text{一次联合采样}
\longrightarrow
\text{动作 chunk}
$$

未来视频主要作为动作生成的结构化中间变量，而不是显式规划树中的 rollout。

### 2.3 不是长期依赖预测视频的 open-loop 世界模型

普通自回归视频生成容易形成：

$$
\hat{o}_1\rightarrow \hat{o}_2\rightarrow \hat{o}_3
$$

DreamZero 每轮执行动作后，将真实观测写回缓存：

$$
\hat{o}_1,\hat{a}_1
\rightarrow
\text{执行}
\rightarrow
o_1^{\text{real}}
\rightarrow
\hat{o}_2,\hat{a}_2
$$

因此预测视频不会成为长期历史事实。

---

## 3. DreamZero 试图解决的三个问题

## 3.1 视频—动作对齐

视频和动作必须描述同一个物理过程。例如：

- 视频里机械臂向左，动作不能向右；
- 夹爪闭合时刻要和视觉接触时刻一致；
- 语言描述的任务阶段要和当前视频阶段一致；
- 视频采样率与动作控制频率必须稳定对齐。

若视频和动作由两个相互独立的头预测，容易出现模态错位。

DreamZero 的做法是：让视频 token 与动作 token 在同一个因果 DiT 中共同去噪，并共享联合训练目标。

## 3.2 双向架构还是自回归架构

固定长度的双向视频扩散模型适合生成 clip，但闭环控制要求：

- 能从任意任务中间状态继续；
- 能处理可变长度历史；
- 不破坏原始视频 FPS；
- 能持续吸收真实新观测；
- 推理时使用 KV Cache。

因此 DreamZero 选择因果、自回归视频建模。

## 3.3 实时推理

朴素 DreamZero 存在三个主要瓶颈：

1. 16 个 flow/diffusion solver steps；
2. 约 14B 参数的 DiT；
3. 串行“推理完再执行”的控制流程。

基线约需：

$$
5.7\ \text{s/action chunk}
$$

经过系统级、实现级和模型级优化后，GB200 上累计加速约：

$$
38\times
$$

对应延迟：

$$
\frac{5.7}{38}\approx 0.15\ \text{s}
$$

即约：

$$
\frac{1}{0.15}\approx 6.67\ \text{Hz}
$$

论文近似称为 7 Hz policy inference。

> **Note**
> 7 Hz 指高层策略重新生成动作 chunk 的频率，不等于底层动作控制频率。底层控制仍可运行在 15 Hz 或 30 Hz。

---

## 4. 问题定义与概率分解

## 4.1 符号表

| 符号 | 含义 |
|---|---|
| $o_t$ | 时刻 $t$ 的视觉观测 |
| $o_{0:l}$ | 从轨迹起点到当前索引 $l$ 的视觉历史 |
| $o_{l:l+H}$ | 未来视觉序列 |
| $a_t$ | 时刻 $t$ 的机器人动作 |
| $a_{l:l+H}$ | 未来长度为 $H$ 的动作序列 |
| $q_l$ | 当前 proprioceptive state |
| $c$ | 语言指令 |
| $H$ | 预测或动作 horizon |
| $l$ | 从轨迹中随机采样的当前索引 |
| $\pi_0$ | DreamZero 所表示的联合条件分布/策略 |

DreamZero 建模：

$$
\pi_0
\left(
o_{l:l+H},a_{l:l+H}
\mid
o_{0:l},c,q_l
\right)
$$

并将其概率语义分解为：

$$
\pi_0
\left(
o_{l:l+H},a_{l:l+H}
\mid
o_{0:l},c,q_l
\right)
=
\pi_0
\left(
o_{l:l+H}
\mid
o_{0:l},c,q_l
\right)
\pi_0
\left(
a_{l:l+H}
\mid
o_{0:l+H},q_l
\right)
$$

第一项是视频预测：

$$
\pi_0
\left(
o_{l:l+H}
\mid
o_{0:l},c,q_l
\right)
$$

第二项具有逆动力学模型的语义：

$$
\pi_0
\left(
a_{l:l+H}
\mid
o_{0:l+H},q_l
\right)
$$

其归纳偏置可理解为：

$$
\text{语言与当前状态}
\rightarrow
\text{未来视觉动态}
\rightarrow
\text{动作}
$$

但实现上不是两个网络，而是同一个 Joint Video-Action DiT。

> **Important**
> 动作因子中没有显式写 $c$，不表示动作分支完全看不到语言。联合模型整体仍接收语言条件；公式主要强调未来视觉对动作的中介作用。

---

## 5. 模型架构

## 5.1 三类输入条件

### 5.1.1 视觉上下文

真实图像通过 VAE Encoder 压缩为视频 latent：

$$
o \xrightarrow{\text{VAE Encoder}} z
$$

对第 $k$ 个 chunk，记干净视频 latent 为：

$$
z_1^k
$$

其中下标 $1$ 表示 Flow Matching 路径的干净数据端。

### 5.1.2 语言条件

语言指令 $c$ 通过文本编码器形成条件 token，用于约束未来视频和动作。

### 5.1.3 Proprioception

机器人本体状态记作：

$$
q_k
$$

可能包含：

- 关节位置；
- 关节速度；
- 夹爪状态；
- 末端执行器状态；
- 其他机器人内部状态。

纯视觉无法唯一恢复机器人内部构型，因此 proprioception 对动作预测是必要的。

---

## 5.2 视频路径

训练时：

$$
\text{RGB 视频}
\rightarrow
\text{VAE Encoder}
\rightarrow
\text{视频 latent}
\rightarrow
\text{加噪}
\rightarrow
\text{Joint DiT}
$$

推理时，理论上可通过 VAE Decoder 将预测视频 latent 解码为未来帧，但实时控制伪代码中没有要求真正解码未来 RGB。

> **Important**
> 未来视频主要在 latent 空间中起作用，不必每轮都生成完整 RGB 视频。

---

## 5.3 动作路径

连续动作先被归一化，再通过 Action Encoder 映射到适合 DiT 的 token/latent 空间。

若动作维度为 $d_a$，action horizon 为 $H$，则一个动作 chunk 可抽象为：

$$
a^k\in\mathbb{R}^{H\times d_a}
$$

经过 Joint DiT 去噪后，再通过 Action Decoder 恢复到机器人动作空间。

---

## 5.4 Joint Video-Action DiT

核心模型是因果 DiT，同时接收：

- noisy video latent；
- noisy action latent；
- 历史视觉 KV Cache；
- 文本条件；
- proprioception；
- flow timestep。

输出联合 velocity：

$$
u_\theta(\cdot)
=
\left[
u_\theta^{\text{video}},
u_\theta^{\text{action}}
\right]
$$

其中：

- $\theta$：模型参数；
- $u_\theta^{\text{video}}$：视频 latent 的速度场预测；
- $u_\theta^{\text{action}}$：动作 latent 的速度场预测。

“联合”意味着视频 token 与动作 token 在允许的注意力区域内共同交互，而不仅是简单相加两个独立 loss。

---

## 5.5 为什么尽量少改视频基础模型

DreamZero 只增加少量机器人专用模块：

- state encoder；
- action encoder；
- action decoder；
- 必要的模态投影层。

多相机视角不是用专门 3D 架构融合，而是拼接成一张大图。

优点：

- 保留预训练视频模型结构；
- 可直接继承视频模型权重；
- 工程改动较小。

代价：

- 没有显式建模相机几何；
- 多视角拼接可能牺牲每个视角分辨率；
- 模型需自行学习拼接边界与相机身份。

---

## 6. 为什么使用自回归视频建模

## 6.1 双向模型的语言—视频错配

假设完整任务跨度为：

$$
T=0\rightarrow T=30
$$

当前闭环采样点位于：

$$
T=20
$$

语言描述的是完整任务，但固定长度双向视频模型只生成局部 clip。

若不做重采样，则语言可能描述整个任务，而视频只覆盖任务尾部。

若做时间重采样，则原始 FPS 被改变，视频与高频动作难以稳定对齐。

---

## 6.2 自回归架构的优势

自回归模型以历史为条件，只预测下一段：

$$
o_{\text{future}}
\sim
p
\left(
o_{\text{future}}
\mid
o_{\text{history}},c,q
\right)
$$

因此可以：

1. 保留原生 FPS；
2. 从任务中间任意位置继续；
3. 用 KV Cache 累积视觉历史；
4. 不必每轮重新处理全部历史帧；
5. 将真实观测持续写回缓存。

---

## 6.3 “只对视频做自回归”的含义

核心闭环为：

$$
\text{真实视觉历史}
\rightarrow
\text{当前未来视频/动作 chunk}
\rightarrow
\text{执行}
\rightarrow
\text{新真实视觉历史}
$$

而不是：

$$
\hat{a}_1\rightarrow\hat{a}_2\rightarrow\hat{a}_3
$$

过去的预测动作不会作为长期自回归事实持续回灌。

> **Warning**
> 论文正文中 $C_k$ 有时写成历史视频—动作对，但推理伪代码更新缓存时传入的是 $[z_{\text{real}},\varnothing]$。从运行逻辑看，长期缓存的核心是视觉历史，而不是预测动作。

---

## 7. Flow Matching 训练

## 7.1 符号定义

对第 $k$ 个 chunk：

| 符号 | 含义 |
|---|---|
| $z_1^k$ | 干净视频 latent |
| $a_1^k$ | 干净、归一化动作 |
| $z_0^k\sim\mathcal{N}(0,I)$ | 视频高斯噪声 |
| $a_0^k\sim\mathcal{N}(0,I)$ | 动作高斯噪声 |
| $t_k\in[0,1]$ | Flow Matching timestep |
| $z_{t_k}^k$ | timestep 为 $t_k$ 时的带噪视频 latent |
| $a_{t_k}^k$ | timestep 为 $t_k$ 时的带噪动作 |
| $I$ | 单位协方差矩阵 |

采用线性插值路径：

$$
z_{t_k}^k
=
t_k z_1^k
+
(1-t_k)z_0^k
$$

$$
a_{t_k}^k
=
t_k a_1^k
+
(1-t_k)a_0^k
$$

时间方向为：

$$
t=0
\Rightarrow
\text{纯噪声}
$$

$$
t=1
\Rightarrow
\text{干净数据}
$$

---

## 7.2 联合变量形式

定义：

$$
x_1^k=[z_1^k,a_1^k]
$$

$$
x_0^k=[z_0^k,a_0^k]
$$

则：

$$
x_t^k
=
t x_1^k+(1-t)x_0^k
$$

对 $t$ 求导：

$$
\frac{d x_t^k}{dt}
=
x_1^k-x_0^k
$$

因此目标速度为：

$$
v^k
=
[z_1^k,a_1^k]
-
[z_0^k,a_0^k]
$$

模型学习：

$$
u_\theta
\left(
x_t;\text{conditions},t
\right)
\approx
v
$$

推理时从：

$$
x_0\sim\mathcal{N}(0,I)
$$

出发，求解 ODE：

$$
\frac{dx_t}{dt}
=
u_\theta
\left(
x_t;\text{conditions},t
\right)
$$

积分到 $t=1$，得到干净的视频与动作。

---

## 8. 联合训练损失

## 8.1 统一记号

为避免论文记号复用，采用：

| 符号 | 含义 |
|---|---|
| $M$ | 轨迹 chunk 数 |
| $K_v$ | 每个 chunk 中的视频 latent 帧数 |
| $k$ | chunk 索引 |
| $C_k$ | 第 $k$ 个 chunk 之前的干净上下文 |
| $c$ | 语言条件 |
| $q_k$ | 第 $k$ 个 chunk 的 proprioception |
| $t_k$ | 当前 timestep |
| $w(t_k)>0$ | timestep 权重 |
| $u_\theta$ | Joint Video-Action DiT |
| $v^k$ | 真实联合 velocity |

更清晰的损失写法为：

$$
\mathcal{L}(\theta)
=
\mathbb{E}
\left[
\frac{1}{M}
\sum_{k=1}^{M}
w(t_k)
\left\|
u_\theta
\left(
[z_{t_k}^k,a_{t_k}^k];
C_k,c,q_k,t_k
\right)
-
v^k
\right\|_2^2
\right]
$$

其中：

$$
v^k
=
[z_1^k,a_1^k]
-
[z_0^k,a_0^k]
$$

---

## 8.2 各部分含义

当前 noisy chunk：

$$
[z_{t_k}^k,a_{t_k}^k]
$$

条件：

$$
C_k,\ c,\ q_k,\ t_k
$$

模型输出：

$$
u_\theta
\left(
[z_{t_k}^k,a_{t_k}^k];
C_k,c,q_k,t_k
\right)
$$

监督目标：

$$
v^k
$$

权重：

$$
w(t_k)>0
$$

用于调节不同噪声阶段的 loss 权重。

---

## 8.3 标准 DreamZero 的共享 timestep

标准模式中：

$$
t_k^{\text{video}}
=
t_k^{\text{action}}
=
t_k
$$

即视频和动作具有相同的去噪进度。

优点：有助于训练初期快速建立视频—动作对应关系。

缺点：动作收敛速度可能被视频去噪速度绑定。

---

## 9. Teacher Forcing 与 chunk-wise 训练

对第 $k$ 个 chunk，使用前面干净 chunk 作为上下文：

$$
C_k
=
\left\{
(z_1^j,a_1^j)
\right\}_{j=1}^{k-1}
$$

当前去噪对象是 noisy chunk：

$$
(z_{t_k}^k,a_{t_k}^k)
$$

但历史条件使用 ground truth，而不是前一段模型预测。

这样做：

1. 稳定训练；
2. 避免早期预测错误污染后续 chunk；
3. 与推理时“执行后用真实观测替换预测视频”的闭环机制一致。

---

## 10. Attention Mask

<img src="../../notes-assets/dreamzero-mask.png" alt="dreamzero mask.png" loading="lazy" />


## 10.1 图中符号

| 符号 | 含义 |
|---|---|
| $C_0,C_1,C_2$ | clean conditioning frame/chunk |
| $Z_1,Z_2,Z_3$ | 待预测视频 latent |
| $Y_1,Y_2,Y_3$ | 待预测动作 latent |
| 行 | Query |
| 列 | Key/Value |
| 蓝色 | 当前 forward 中 self-attention |
| 橙色 | 历史 KV Cache |

---

## 10.2 为什么训练和推理时的 attention mask 不一样

表面上 mask 不同，是因为计算组织方式不同；它们想实现的因果依赖基本一致：

$$
\text{当前视频/动作 chunk}
\text{可看过去真实视觉历史，但不可看未来 chunk}
$$

---

## 10.3 训练时的 mask

训练时，一次 forward 通常并行处理多个 chunk：

$$
C_0,\ Z_1,\ Y_1,\ C_1,\ Z_2,\ Y_2,\ C_2,\ Z_3,\ Y_3
$$

需要 block-causal mask 防止未来信息泄漏。

第一个 chunk：

$$
(Z_1,Y_1)
\leftarrow
(C_0,Z_1,Y_1)
$$

第二个 chunk：

$$
(Z_2,Y_2)
\leftarrow
(C_0,C_1,Z_2,Y_2)
$$

第三个 chunk：

$$
(Z_3,Y_3)
\leftarrow
(C_0,C_1,C_2,Z_3,Y_3)
$$

一般形式：

$$
(Z_k,Y_k)
\leftarrow
(C_0,\ldots,C_{k-1},Z_k,Y_k)
$$

不可访问未来：

$$
C_{k+1},Z_{k+1},Y_{k+1},\ldots
$$

训练 mask 是一个较大的块状因果矩阵，用于在一次并行 forward 中模拟多段自回归生成。

---

## 10.4 推理时的 mask

推理时，历史已经存在 KV Cache：

$$
KV_{\text{history}}
=
\left\{
K(C_0),V(C_0),\ldots,K(C_{k-1}),V(C_{k-1})
\right\}
$$

当前只需处理：

$$
Z_k,\ Y_k
$$

注意力为：

$$
\operatorname{Attention}
\left(
Q_{Z_k,Y_k},
[K_{\text{cache}},K_{Z_k,Y_k}],
[V_{\text{cache}},V_{Z_k,Y_k}]
\right)
$$

历史 token 不再作为 Query 重新计算，只作为 Key/Value 被当前 Query 查询。

因此：

- 训练：显式大 mask；
- 推理：历史 KV + 当前 self-attention。

> **Summary**
> 训练 mask 是 KV Cache 推理结构的并行化模拟。

---

## 10.5 是否存在 train-test mismatch

训练时第 $k$ 个 chunk 可见：

$$
\text{过去干净视觉上下文}
+
\text{当前 noisy video/action}
$$

推理时可见：

$$
\text{过去真实视觉的 KV Cache}
+
\text{当前 noisy video/action}
$$

核心条件分布一致：

$$
p
\left(
Z_k,Y_k
\mid
C_{<k},c,q_k
\right)
$$

区别主要在计算路径，而不是可见信息的因果语义。

---

## 11. 训练算法伪代码拆解

<img src="../../notes-assets/dreamzero-alg1.png" alt="dreamzero alg1.png" loading="lazy" />


## 11.1 输入

- 数据集 $\mathcal{D}$；
- 文本条件 $c$；
- chunk 数 $M$；
- Joint Video-Action DiT $u_\theta$。

## 11.2 采样轨迹

$$
\tau\sim\mathcal{D}
$$

一条轨迹包含视频、动作、proprioception 与语言标注。

## 11.3 编码与归一化

视频编码为：

$$
z_1^{1:M}
$$

动作归一化为：

$$
a_1^{1:M}
$$

## 11.4 分 chunk

轨迹被拆成 $M$ 个 chunk，每个 chunk 包含：

- 固定数量的视频 latent 帧；
- 对应时间范围内的动作；
- 对应 proprioception。

## 11.5 定义历史上下文

对第 $k$ 个 chunk：

$$
C_k
=
\left\{
(z_1^j,a_1^j)
\right\}_{j=1}^{k-1}
$$

## 11.6 采样 timestep

标准模式：

$$
t_k\sim\mathcal{U}(0,1)
$$

## 11.7 DreamZero-Flash 可选分支

Flash 模式：

$$
t_{\text{vid}}\sim 1-\operatorname{Beta}(7,1)
$$

$$
t_{\text{act}}\sim\mathcal{U}(0,1)
$$

标准模式：

$$
t_{\text{vid}}=t_{\text{act}}=t_k
$$

## 11.8 采样噪声

$$
z_0^k,a_0^k\sim\mathcal{N}(0,I)
$$

## 11.9 构造带噪变量

$$
z_t^k
=
t_{\text{vid}}z_1^k
+
(1-t_{\text{vid}})z_0^k
$$

$$
a_t^k
=
t_{\text{act}}a_1^k
+
(1-t_{\text{act}})a_0^k
$$

## 11.10 预测速度

$$
v_{\text{pred}}
=
u_\theta
\left(
[z_t^k,a_t^k];
C_k,c,q_k,t_k
\right)
$$

## 11.11 目标速度

$$
v^k
=
[z_1^k,a_1^k]
-
[z_0^k,a_0^k]
$$

## 11.12 损失与更新

$$
\mathcal{L}
=
\left\|
v_{\text{pred}}-v^k
\right\|_2^2
$$

$$
\theta
\leftarrow
\theta-\eta\nabla_\theta\mathcal{L}
$$

这里的 $\eta$ 是学习率，与 Flash 中 Beta 分布随机变量同名，属于符号复用。

---

## 12. 闭环推理算法拆解

<img src="../../notes-assets/dreamzero-alg2.png" alt="dreamzero alg2.png" loading="lazy" />

## 12.1 输入

- 语言指令 $c$；
- 初始图像 $o_{\text{init}}$；
- 初始 proprioception $q_{\text{init}}$。

## 12.2 超参数

- $N$：solver steps；
- $\tau$：DiT Cache 的余弦相似度阈值。

## 12.3 初始化

$$
KV\leftarrow\varnothing
$$

$$
v_{\text{prev}}\leftarrow\varnothing
$$

$$
q_{\text{curr}}\leftarrow q_{\text{init}}
$$

---

## 12.4 阶段一：Prefill Cache

编码初始图像：

$$
z_{\text{init}}
=
\operatorname{VAE}(o_{\text{init}})
$$

通过模型的 cache-update 模式，将真实视觉写入 KV Cache。

> **Warning**
> 推理伪代码中 cache update 使用 $t=0$，但 Flow Matching 公式中 $t=0$ 是噪声端。更合理的理解是：这里的 $t=0$ 是 context-phase 的特殊标记，输入 latent 本身仍是干净真实 latent，而不是重新加噪后的变量。

---

## 12.5 阶段二：初始化联合噪声

每轮生成：

$$
x_0=[z_0,a_0]\sim\mathcal{N}(0,I)
$$

其中：

- $z_0$：未来视频 latent 初始噪声；
- $a_0$：未来动作初始噪声。

---

## 12.6 阶段三：从 $t=0$ 积分到 $t=1$

对：

$$
i=0,\ldots,N-1
$$

scheduler 给出：

$$
t_i,t_{i+1}
$$

模型预测速度：

$$
v_i
=
u_\theta
\left(
x_{t_i};
KV,c,q_{\text{curr}},t_i
\right)
$$

随后数值求解器更新：

$$
x_{t_{i+1}}
=
x_{t_i}
+
\operatorname{Step}
\left(
v_i,t_i,t_{i+1}
\right)
$$

---

## 12.7 DiT Caching

定义余弦相似度：

$$
\operatorname{CosSim}(v_i,v_{i-1})
=
\frac{
v_i^\top v_{i-1}
}{
\|v_i\|_2\|v_{i-1}\|_2
}
$$

若：

$$
\operatorname{CosSim}(v_i,v_{i-1})>\tau
$$

则复用缓存 velocity：

$$
v_i\leftarrow v_{\text{prev}}
$$

否则运行完整 DiT forward。

---

## 12.8 阶段四：提取动作并执行

最终：

$$
x_1=[z_1,a_1]
$$

取动作部分：

$$
\hat{a}
=
\operatorname{Filter}
\left(
x_1^{\text{act}}
\right)
$$

动作被异步发送给机器人执行。

---

## 12.9 阶段五：真实观测写回

执行过程中得到：

- 新图像 $o_{\text{real}}$；
- 新 proprioception $q_{\text{real}}$。

编码：

$$
z_{\text{real}}
=
\operatorname{VAE}(o_{\text{real}})
$$

并写入 KV Cache。

本轮预测视频 latent：

$$
x_1^{\text{vid}}
$$

被丢弃。

---

## 13. Scheduler 是什么

## 13.1 定义

在 DreamZero 中，scheduler 是 Flow Matching 推理阶段的数值积分控制模块，负责：

1. 生成或管理离散 timesteps；
2. 决定从 $t_i$ 到 $t_{i+1}$ 的更新方式；
3. 管理 solver 所需系数和历史 velocity。

它不是：

- Transformer；
- 训练优化器；
- 机器人动作调度器。

---

## 13.2 Flow Matching ODE

联合状态：

$$
x_t=[z_t,a_t]
$$

模型学习：

$$
\frac{dx_t}{dt}
=
u_\theta
\left(
x_t;\text{condition},t
\right)
$$

推理从：

$$
x_0\sim\mathcal{N}(0,I)
$$

积分到：

$$
x_1=[z_1,a_1]
$$

由于计算机不能连续积分，需要离散时间网格：

$$
0=t_0<t_1<\cdots<t_N=1
$$

scheduler 就是管理这条离散采样路径的组件。

---

## 13.3 Scheduler 与 solver 的区别

概念上：

### Scheduler

决定：

$$
t_0,t_1,\ldots,t_N
$$

以及步长、噪声/信号系数、数值更新参数。

### Solver

根据模型给出的速度，将状态从：

$$
x_{t_i}
$$

更新到：

$$
x_{t_{i+1}}
$$

很多扩散/Flow Matching 库会把 scheduler 与 solver 封装在同一个对象里。

---

## 13.4 最简单的例子：均匀时间步 + Euler

设：

$$
N=4
$$

则：

$$
t_0=0,\quad
t_1=0.25,\quad
t_2=0.5,\quad
t_3=0.75,\quad
t_4=1
$$

模型输出：

$$
v_i=u_\theta(x_{t_i},t_i)
$$

Euler 更新：

$$
x_{t_{i+1}}
=
x_{t_i}
+
(t_{i+1}-t_i)v_i
$$

---

## 13.5 Flow UniPC

DreamZero 使用 Flow UniPC scheduler。

UniPC 可理解为多步 predictor-corrector 求解器，会利用多个历史 velocity：

$$
v_i,v_{i-1},v_{i-2},\ldots
$$

抽象更新形式：

$$
x_{t_{i+1}}
=
x_{t_i}
+
\sum_{j=0}^{p-1}
\gamma_j v_{i-j}
$$

其中：

- $p$：求解器阶数或历史步数；
- $\gamma_j$：依赖时间间隔的系数。

相比 Euler，多步求解器通常能在较少模型评估下保持更高采样质量。

---

## 13.6 训练 timestep sampling 与推理 scheduler 的区别

### 训练时 timestep sampling

例如：

$$
t\sim\mathcal{U}(0,1)
$$

或 Flash 中：

$$
t^{\text{video}}\sim\operatorname{Beta}(1,7)
$$

目的是构造不同噪声程度的训练样本。

### 推理时 scheduler

使用预设或动态时间路径：

$$
0=t_0<t_1<\cdots<t_N=1
$$

目的是数值积分，将噪声逐步变成干净样本。

两者都涉及 $t$，但功能不同。

---

## 13.7 Scheduler 如何影响速度和质量

若每次 DiT forward 延迟为 $L$，总步数为 $N$，则近似：

$$
T_{\text{inference}}\approx NL
$$

减少 solver steps 或减少真实 DiT evaluations 可以显著加速，但步数过少会增加积分误差，表现为：

- 动作不平滑；
- 接触时刻错误；
- 视频质量下降；
- 轨迹偏移；
- 末端执行精度下降。

---

## 13.8 DiT Cache 与 scheduler 的关系

scheduler 可以保留 16 个数值时间点，但不一定每个点都运行一次完整 DiT。

当：

$$
\operatorname{CosSim}(v_i,v_{i-1})>\tau
$$

时复用旧 velocity。

因此：

- 数值时间网格可能仍有 16 步；
- 真实昂贵的 DiT forward 平均只有约 4 次；
- 其余步骤仍由 scheduler 执行状态更新。

“16 步降到 4 步”更准确地说，是有效 DiT 模型评估次数降到约 4 次。

---

## 13.9 为什么 scheduler 要迁移到 GPU

若 scheduler 部分运行在 CPU，则每步可能发生：

$$
\text{GPU DiT}
\rightarrow
\text{CPU scheduler}
\rightarrow
\text{GPU 下一步}
$$

频繁 CPU–GPU 同步会导致 GPU stall。

将 scheduler 放到 GPU 后：

$$
\text{GPU DiT}
\rightarrow
\text{GPU scheduler step}
\rightarrow
\text{GPU DiT}
$$

可减少同步、Python 调度和 kernel launch 开销，也更容易被 CUDA Graph 捕获。

---

## 14. 异步闭环执行

## 14.1 同步执行的问题

朴素流程：

1. 机器人等待；
2. 模型推理；
3. 得到动作；
4. 机器人执行；
5. 再次等待。

这种方式会产生长时间停顿或 open-loop。

## 14.2 异步结构

控制线程：

$$
\text{执行 action chunk}_n
$$

推理线程：

$$
\text{同时生成 action chunk}_{n+1}
$$

延迟约束从：

> 推理必须在机器人开始运动前完成

变成：

> 推理必须在当前可用动作耗尽前完成

---

## 14.3 Action horizon 与时间跨度

Agibot 设置：

$$
H=48
$$

动作频率：

$$
30\ \text{Hz}
$$

每个 chunk 覆盖：

$$
\frac{48}{30}=1.6\ \text{s}
$$

DROID 设置：

$$
H=24,\quad 15\ \text{Hz}
$$

同样得到：

$$
\frac{24}{15}=1.6\ \text{s}
$$

尽管 action chunk 有 1.6 秒，论文仍将推理目标设为约 200 ms 以下，以保证：

- 更快吸收新观测；
- 更强扰动响应；
- 降低基于旧观测执行的时间；
- 为编码、通信、调度留出余量。

---

## 15. 系统级优化

## 15.1 CFG Parallelism

Classifier-Free Guidance 通常需要条件与无条件两次前向：

$$
v_{\text{cfg}}
=
v_{\text{uncond}}
+
s
\left(
v_{\text{cond}}-v_{\text{uncond}}
\right)
$$

其中：

- $s$：guidance scale；
- $v_{\text{cond}}$：有条件 velocity；
- $v_{\text{uncond}}$：无条件 velocity；
- $v_{\text{cfg}}$：最终 velocity。

DreamZero 将两次前向分配到两块 GPU 并行执行，单步延迟降低约 47%。

代价是增加 GPU 资源占用。

---

## 15.2 DiT Caching

相邻 solver step 的 velocity 往往方向相近：

$$
v_{i+1}\approx v_i
$$

当余弦相似度超过阈值时，复用 velocity，跳过部分 DiT forward。

平均有效 DiT steps：

$$
16\rightarrow 4
$$

风险：

- 接触、抓取、释放等阶段可能变化剧烈；
- 阈值过低会过度缓存，损害动作质量；
- 阈值过高则加速有限。

---

## 15.3 异步执行

异步执行本身不减少模型 FLOPs，但改变系统 critical path，使机器人运动与模型推理并发。

---

## 16. 实现级优化

## 16.1 `torch.compile` 与 CUDA Graph

使用：

```python
torch.compile(mode="reduce-overhead")
```

并启用 CUDA Graph。

目标：

- 减少 Python 开销；
- 减少 CPU kernel launch；
- 融合算子；
- 降低显存带宽需求；
- 重复执行固定计算图。

编译组件包括：

- diffusion transformer；
- scheduler；
- text encoder；
- image encoder；
- VAE。

采用静态 shape：

```python
dynamic=False
```

由于 KV Cache shape 会演化，第一条轨迹可能发生多次 recompilation；之后可复用已捕获图。

模型被重构为函数式接口：

$$
(\text{output},KV_{\text{new}})
=
f(\text{input},KV_{\text{old}})
$$

而不是隐式修改全局缓存。

---

## 16.2 Post-Training Quantization

Blackwell SM100 上采用混合精度。

大多数权重和激活：

$$
\text{NVFP4 (E2M1)}
$$

敏感 QKV projection 与 Softmax：

$$
\text{FP8 (E4M3)}
$$

LayerNorm、RoPE 等非线性/累积：

$$
\text{FP16}
$$

这样在降低显存与带宽的同时保留注意力数值稳定性。

---

## 16.3 Kernel-Level Enhancements

使用 cuDNN backend 的 Scaled Dot-Product Attention：

- 更高效 QK 点积；
- 更高效 Softmax；
- 更高效 attention × V；
- 更好的内存布局与 kernel fusion。

---

## 16.4 Scheduler Optimization

将 Flow UniPC scheduler 中原本运行在 CPU 的操作迁移到 GPU，避免频繁 CPU–GPU 同步与 GPU stall。

---

## 17. DreamZero-Flash

## 17.1 标准耦合噪声日程

标准模式：

$$
t_k^{\text{video}}
=
t_k^{\text{action}}
=
t_k
$$

$$
t_k\sim\mathcal{U}(0,1)
$$

视频和动作总处在相同去噪阶段。

---

## 17.2 解耦噪声日程

Flash 模式：

$$
t_k^{\text{video}}
=
1-\eta
$$

$$
\eta\sim\operatorname{Beta}(\alpha,\beta)
$$

$$
t_k^{\text{action}}
\sim
\mathcal{U}(0,1)
$$

论文使用：

$$
\alpha=7,\quad\beta=1
$$

当：

$$
\alpha>\beta
$$

时，$\eta$ 集中在 1 附近，因此：

$$
t_k^{\text{video}}=1-\eta
$$

集中在 0 附近。

因为论文定义：

$$
t=0
\Rightarrow
\text{高噪声}
$$

所以 Flash 训练中视频常处于高噪声状态。

---

## 17.3 期望值

若：

$$
\eta\sim\operatorname{Beta}(7,1)
$$

则：

$$
\mathbb{E}[\eta]
=
\frac{7}{7+1}
=
0.875
$$

因此：

$$
\mathbb{E}
\left[
t^{\text{video}}
\right]
=
1-0.875
=
0.125
$$

标准均匀分布的期望为：

$$
\mathbb{E}[t]=0.5
$$

等价地：

$$
t^{\text{video}}
\sim
\operatorname{Beta}(1,7)
$$

其概率密度：

$$
p(t)=7(1-t)^6,\quad 0\le t\le 1
$$

---

## 17.4 解耦加噪

视频：

$$
z_{t_k}^{k,\text{video}}
=
t_k^{\text{video}}z_1^k
+
\left(
1-t_k^{\text{video}}
\right)
z_0^k
$$

动作：

$$
a_{t_k}^{k,\text{action}}
=
t_k^{\text{action}}a_1^k
+
\left(
1-t_k^{\text{action}}
\right)
a_0^k
$$

---

## 17.5 Flash 为什么能加速动作

标准耦合训练使模型习惯于：

> 动作较干净时，视频也应较干净。

因此动作质量依赖于视频被充分去噪。

Flash 刻意训练：

> 视频仍高度嘈杂时，动作也必须能够被正确恢复。

模型因此被迫更多利用：

- 真实历史 KV Cache；
- 语言；
- proprioception；
- noisy future video 中有限的结构；

直接预测高质量动作。

本质是：

$$
\text{解除动作收敛速度对视频收敛速度的依赖}
$$

---

## 18. 动作平滑

去噪生成的动作 chunk 可能含高频抖动。

处理流程：

## 18.1 立方插值上采样

原动作长度为 $H$，先上采样到约 $2H$。

## 18.2 Savitzky–Golay 滤波

参数：

- window size = 21；
- polynomial order = 3。

它通过局部三次多项式拟合抑制高频噪声，同时尽量保留轨迹形状。

## 18.3 下采样

滤波后恢复到原始动作频率。

代价：

- 可能削弱真正需要的快速接触动作；
- 可能掩盖模型本身的高频预测问题。

---

## 19. 累计加速结果

| 累计配置 | H100 | GB200 |
|---|---:|---:|
| Baseline | $1\times$ | $1.1\times$ |
| + CFG Parallelism | $1.9\times$ | $1.8\times$ |
| + DiT Caching | $5.5\times$ | $5.4\times$ |
| + Compile + CUDA Graphs | $8.9\times$ | $10.9\times$ |
| + Kernel & Scheduler | $9.6\times$ | $14.8\times$ |
| + NVFP4 Quantization | 不适用 | $16.6\times$ |
| + DreamZero-Flash | 不适用 | $38\times$ |

关键观察：

1. DiT Caching 是早期最大收益项之一；
2. GB200 更能从编译、内核与 scheduler 优化中获益；
3. NVFP4 边际收益约：

$$
\frac{16.6}{14.8}\approx1.12
$$

4. Flash 的边际收益约：

$$
\frac{38}{16.6}\approx2.29
$$

5. 38 倍是端到端累计实测，不是把每项理论倍数直接相乘。

---

## 20. 训练配置

## 20.1 每个 chunk 的视频 latent 帧数

$$
K_v=2
$$

预实验中 $K_v=2$ 优于 $K_v=1$。

## 20.2 Chunk 数

默认：

$$
M=4
$$

最大 latent context：

$$
4\times 2=8
$$

个 latent frames。

## 20.3 Agibot

- 视频：5 FPS；
- 动作：30 Hz；
- action horizon：

$$
H=48
$$

chunk 时间跨度：

$$
\frac{48}{30}=1.6\ \text{s}
$$

## 20.4 DROID

- 视频：5 FPS；
- 动作：15 Hz；
- action horizon：

$$
H=24
$$

chunk 时间跨度：

$$
\frac{24}{15}=1.6\ \text{s}
$$

## 20.5 最大视觉上下文

- 8 个 latent frames；
- 33 个 raw frames；
- 约 6.6 秒视觉历史。

$$
\frac{33}{5}=6.6\ \text{s}
$$

---

## 21. 与 action-only Flow Policy 的区别

Action-only policy：

$$
(\text{image},\text{text},q)
\rightarrow
\text{action flow}
$$

DreamZero：

$$
(\text{video history},\text{text},q)
\rightarrow
(\text{future video flow},\text{action flow})
$$

DreamZero 的潜在优势：

- 继承视频预训练中的物体运动与交互先验；
- 未来视频为动作提供结构约束；
- 视觉历史可作为显式状态记忆；
- 模态对齐更强。

代价：

- 14B 视频 DiT 成本很高；
- 联合去噪难以实时；
- 动作可能受视频生成质量牵制；
- 需要复杂系统优化。

---

## 22. 重要技术洞察

## 22.1 未来视频不必解码

预测视频可以停留在 latent 空间，控制只提取动作。

## 22.2 真实观测更新的是 KV Cache

新观测只在下一轮推理中生效，不会中途修正当前 ODE 求解。

## 22.3 一个 action chunk 内仍相对 open-loop

异步推理缩短 open-loop 时间，但不能完全消除。

## 22.4 KV Cache 不是无限免费

尽管避免重复计算，缓存仍消耗显存，注意力访问成本也随历史增长。实际部署中上下文被限制在约 6.6 秒。

## 22.5 Teacher forcing 不能完全消除分布偏移

训练历史来自干净示范；推理历史来自机器人真实执行后的状态。真实状态虽无视频预测伪影，但可能偏离示范分布。

## 22.6 动作平滑既是部署技巧，也是信号

如果需要较强 Savitzky–Golay 滤波，说明原始动作生成仍有高频噪声。

---

## 23. 论文记号与伪代码中的不完全一致

## 23.1 $K$ 的复用

正文用 $K$ 表示 chunk 数，附录又用 $K=2$ 表示每个 chunk 的 latent 帧数。

建议统一为：

- $M$：chunk 数；
- $K_v$：每 chunk 的视频 latent 帧数。

## 23.2 Cache update 的 $t=0$

Flow Matching 中 $t=0$ 是噪声端，但 cache update 使用 clean latent 与 $t=0$。更可能是 context-mode 特殊标记或伪代码简化。

## 23.3 历史上下文是否含动作

正文 $C_k$ 写成历史视频—动作对，但推理时写回缓存的是：

$$
[z_{\text{real}},\varnothing]
$$

运行逻辑更支持长期缓存真实视觉，而不是预测动作。

## 23.4 Flash 中 timestep 输入不完整

Flash 中：

$$
t_{\text{video}}\neq t_{\text{action}}
$$

但伪代码调用模型时仍只写一个 $t_k$。真实实现必须使用模态专用 timestep embedding 或其他方式区分两者。

---

## 24. 一次真实控制循环

给定语言：

> 把水果放进袋子。

## 24.1 写入真实状态

$$
o_t
\rightarrow
z_t^{\text{real}}
\rightarrow
KV\text{ Cache}
$$

同时读取：

$$
q_t
$$

## 24.2 初始化未来噪声

$$
z_0^{\text{future}}\sim\mathcal{N}(0,I)
$$

$$
a_0^{\text{future}}\sim\mathcal{N}(0,I)
$$

## 24.3 联合求解

模型利用：

- 历史视觉；
- 语言；
- proprioception；
- noisy future video；
- noisy action；

逐步预测联合 velocity。

## 24.4 提取动作

未来视频 latent 不必解码，动作经过平滑后发送给机器人。

## 24.5 异步执行

机器人执行当前 action chunk，模型并发生成下一 chunk。

## 24.6 真实世界纠错

新图像写回 KV Cache，上一轮预测视频不再保留。

---

## 25. 核心结论

DreamZero 的关键价值不是简单“用视频模型预测动作”，而是将以下目标统一起来：

1. 利用视频基础模型的时空先验；
2. 用联合 Flow Matching 对齐未来视觉与动作；
3. 用因果 KV Cache 保存视觉历史；
4. 用真实观测替换预测视频，抑制误差累积；
5. 用异步执行隐藏大模型延迟；
6. 用 DreamZero-Flash 解耦视频与动作去噪难度；
7. 用系统、编译、量化和内核优化达到实时部署。

可以将其设计逻辑压缩为：

> 训练时，未来视频是动作学习的结构化监督。  
> 推理时，未来视频是动作生成的 latent scaffold。  
> 执行后，未来视频预测立即让位于真实世界观测。

---

## 26. 快速复习问答

## Q1：为什么训练和推理的 attention mask 不一样？

因为训练一次并行处理多个 chunk，需要显式 block-causal mask 防止未来泄漏；推理时历史已经存入 KV Cache，只需计算当前 chunk 的 Query 和当前 Key/Value。

## Q2：两者的因果语义是否一致？

基本一致：

$$
\text{当前 chunk}
\leftarrow
\text{过去真实视觉历史}
+
\text{当前 noisy video/action}
$$

## Q3：scheduler 是什么？

scheduler 是 Flow Matching 推理阶段的数值积分控制器，负责：

- 管理 timesteps；
- 计算从 $t_i$ 到 $t_{i+1}$ 的更新；
- 配合 solver 将噪声逐步变成视频与动作。

## Q4：scheduler 和训练 timestep sampling 一样吗？

不一样。

训练 timestep sampling 用来制造训练噪声状态；推理 scheduler 用来实际积分 ODE。

## Q5：为什么 DreamZero-Flash 有效？

它让动作学会在未来视频仍高度嘈杂时快速恢复，从而解除动作去噪速度对视频去噪速度的依赖。

## Q6：为什么预测视频要丢弃？

因为执行后的真实观测比预测未来更可靠。长期使用真实观测可避免自回归视频误差累积。

---

## 27. 最终心智模型

```text
真实视觉历史 ───────┐
语言指令 ──────────┼─> Joint Video-Action DiT
本体状态 ──────────┤       │
未来视频噪声 ──────┤       ├─> 未来视频 latent
未来动作噪声 ──────┘       └─> 动作 chunk
                               │
                               v
                         异步真实执行
                               │
                               v
                         新真实观测
                               │
                               v
                        更新视觉 KV Cache
```

核心闭环：

$$
\text{真实历史}
\rightarrow
\text{联合想象与动作生成}
\rightarrow
\text{执行}
\rightarrow
\text{真实观测纠偏}
$$
