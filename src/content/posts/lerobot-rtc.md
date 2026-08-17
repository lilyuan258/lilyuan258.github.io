---
title: LeRobot RTC 推理机制与真机调参笔记
description: >-
  核心结论 LeRobot 的 RTC（Real Time Chunking）并不是“每执行固定 N 个 action 就重新预测”的机制。 它由两部分组成：
  1. 异步 action queue / inference engine ：机器人继续消费旧 chunk，同时后台预测新 chunk； 2. RTC
  guidance ：新 chu
pubDate: '2026-08-15'
updatedDate: '2026-08-17'
tags:
  - VLA
  - LeRobot
  - RTC
  - Robotics
  - Inference
  - Action-Chunking
  - Real-Time-Control
draft: false
source: LeRobot_RTC_推理机制与真机调参笔记.md
wordCount: 2569
readingTime: 6
---
> **abstract**
> 核心结论
> LeRobot 的 RTC（Real-Time Chunking）并不是“每执行固定 N 个 action 就重新预测”的机制。
> 它由两部分组成：
> 1. **异步 action queue / inference engine**：机器人继续消费旧 chunk，同时后台预测新 chunk；
> 2. **RTC guidance**：新 chunk 在前缀区域参考旧 chunk 的剩余轨迹，以降低 chunk 交替时的不连续。
>
> 真正影响“每个 chunk 实际执行多少步”的主要参数是 `queue_threshold`，而不是 `execution_horizon`。
> 在稳定状态、延迟估计较准确时：
>
> $$
> N_{\text{exec}} \approx C-Q
> $$
>
> 其中 $C=\text{chunk\_size}$，$Q=\text{queue\_threshold}$。

## 1. RTC 整体架构

LeRobot RTC 可以理解为两层机制。

### 1.1 异步执行层

主控制线程持续执行 action queue 中的动作：

```text
observation
    |
    v
RTCInferenceEngine
    |
    +--> ActionQueue.get() --> 每个 control tick 执行 1 个 action
    |
    +--> 后台 RTCInference 线程
             |
             +--> queue 剩余量 <= queue_threshold
             +--> 获取最新 observation
             +--> 获取 previous chunk leftover
             +--> 估计 inference_delay
             +--> predict_action_chunk(...)
             +--> 丢弃已经过时的前 delay 个 action
             +--> merge / replace action queue
```

核心目的：**VLA 推理时机器人不停下来，而是继续执行旧 chunk。**

### 1.2 RTC guidance 层

新 chunk 不是完全独立生成，而是在 flow-matching denoising 过程中对其前缀加入 previous chunk consistency guidance。

直观上：

```text
old chunk:  ------------------------>

new chunk without RTC:
                             /
                            /
                           /

new chunk with RTC:
                         -----
                              \
                               \
                                -------->
```

RTC guidance 解决的是“chunk 接管时是否平滑”，而 action queue 解决的是“推理期间是否会停”。

---

## 2. 五个最重要参数的职责

| 参数 | 主要语义 | 是否直接决定每个 chunk 执行步数 |
|---|---|---|
| `chunk_size = C` | 模型一次预测多少个 action | 是，作为上限和计算基准 |
| `queue_threshold = Q` | queue 剩多少 action 时启动下一次推理 | **主要影响** |
| `inference_delay = d` | 一次推理耗时折算成多少 control step | 影响时序对齐和安全余量 |
| `execution_horizon = H` | 新 chunk 前缀参考旧 chunk 的最大范围 | **不直接决定** |
| `max_guidance_weight` | RTC guidance 的强度 | 不决定执行步数 |

最简记忆：

```text
queue_threshold
    = 多久换一次 chunk

execution_horizon
    = 换 chunk 时过渡多久

max_guidance_weight
    = 过渡时拉旧轨迹有多强

inference_delay
    = 推理期间物理时间前进了多少

chunk_size
    = 模型一次给出的总预测长度
```

---

## 3. 每个 chunk 实际执行多少步：核心推导

设：


$$
C=\text{chunk\_size}
$$


$$
Q=\text{queue\_threshold}
$$


$$
d=\text{inference\_delay}
$$

### 3.1 什么时候启动下一次 inference？

当前 chunk 的 action queue 从 $C$ 开始逐步被消费。

当：


$$
\text{queue size} \le Q
$$

后台线程启动下一次预测。

因此，在启动新 inference 之前，当前 chunk 已执行：


$$
C-Q
$$

步。

### 3.2 inference 期间机器人还会继续执行

假设控制频率为 $f$，一次推理延迟为 $T$，则：


$$
d=\lceil T f \rceil
$$

例如：

```text
control frequency = 30 Hz
inference latency = 0.4 s
```

则：


$$
d=\lceil 0.4 \times 30 \rceil=12
$$

也就是说，GPU 推理期间机器人又执行了约 12 个旧 action。

### 3.3 为什么最终仍然近似是 C-Q？

新 chunk 是在 inference 开始时刻的 observation 上预测的。

因此新 chunk 前 $d$ 个 action 在 inference 返回时已经对应“过去的时间”，会被视为 stale 并跳过。

新 chunk 实际可执行长度约为：


$$
C-d
$$

从它接管后，到 queue 再次下降到 $Q$，先执行：


$$
(C-d)-Q
$$

步；随后下一次 inference 又花 $d$ 步时间。

所以相邻两次 chunk 接管之间：


$$
N_{\text{exec}}
\approx
(C-d-Q)+d
$$

即：


$$
\boxed{N_{\text{exec}}\approx C-Q}
$$

这是 RTC 稳态下最重要的近似关系。

---

## 4. 一个具体例子

假设：

```text
chunk_size C = 50
queue_threshold Q = 30
control frequency = 30 Hz
inference latency = 400 ms
```

则：


$$
d=\lceil0.4\times30\rceil=12
$$

目标 chunk 更新周期：


$$
N_{\text{exec}}\approx50-30=20
$$

时间轴：

```text
Chunk A

0                    20              32
|--------------------|---------------|
执行 A                开始预测 B       B 返回
                     <--- 12 step --->

B 原始预测：
b0 ... b11 | b12 ... b49
^^^^^^^^^^^
已经 stale

B 从 b12 附近接管当前时间。
```

在稳定状态下，下一次 chunk 的接管周期仍约为 20 个 control step。

若控制频率为 30 Hz：


$$
20/30\approx0.667s
$$

即约每 0.67 秒发生一次新 chunk 接管。

---

## 5. queue_threshold 的真实 trade-off

### 5.1 queue_threshold 较大

例如：

```text
C = 50
Q = 40
```

则：


$$
N_{\text{exec}}\approx10
$$

特点：

- 更早开始下一次 inference；
- queue starvation 风险低；
- replanning 更频繁；
- 新 observation 更快影响机器人；
- **chunk 交替次数增加**；
- 如果 chunk boundary 本身不够平滑，真机上更容易看到高频顿挫。

### 5.2 queue_threshold 较小

例如：

```text
C = 50
Q = 20
```

则：


$$
N_{\text{exec}}\approx30
$$

特点：

- 一个 chunk 被执行得更深；
- chunk 切换次数减少；
- 对低动态、简单任务通常更平滑；
- 但留给下一次 inference 的旧 action 库存更少；
- 如果推理太慢，会出现 queue starvation / 等不到新 action 的风险。

因此核心 trade-off 是：

```text
Q 大
  -> 安全余量大
  -> replanning 高频
  -> chunk 切换多
  -> 可能更顿

Q 小
  -> chunk 执行更长
  -> 切换更少
  -> 更平滑
  -> 但更容易 action 库存不足
```

---

## 6. queue_threshold 的安全下界

为了避免 action queue 在下一 chunk 返回前耗尽，应满足：


$$
Q \gtrsim d_{\text{worst-case}}
$$

真机上不应只看平均 inference latency，更建议用 P95 / P99。

例如：

```text
control frequency = 30 Hz
P99 inference latency = 0.43 s
```

则：


$$
d_{p99}=\lceil0.43\times30\rceil=13
$$

可以加 3-5 step safety margin：


$$
Q_{\min}\approx16\sim18
$$

工程上可写为：


$$
\boxed{
Q_{\min}=d_{p99}+M
}
$$

其中 $M$ 用于覆盖：

- CUDA / GPU latency jitter；
- Python thread scheduling；
- camera pipeline 波动；
- control-loop jitter；
- 数据预处理/后处理抖动。

### 对应最大可安全执行长度

因为：


$$
N_{\text{exec}}\approx C-Q
$$

所以：


$$
N_{\text{exec,max}}
\approx
C-(d_{p99}+M)
$$

这给出了“希望减少 chunk 切换”和“必须给推理留足时间”之间的硬约束。

---

## 7. execution_horizon 到底是什么

`execution_horizon = H` 这个名字容易误导。

它**不是**：

```text
每个 chunk 实际执行 H 个 action
```

而是：

> 新 chunk 在前多少个 timestep 内仍然受到 previous chunk 的 consistency guidance。

可以把它更准确地理解为：

```text
prefix_guidance_horizon
transition_horizon
consistency_horizon
```

假设：

```text
execution_horizon = 10
```

概念上：

```text
new chunk

step:   0 1 2 3 4 5 6 7 8 9 | 10 ........
        <------ old chunk influence ------>

weight: ███████▓▓▒▒░░░░░░░░░ | 0 .........
```

它控制的是：**旧轨迹对新轨迹的影响最多持续多远。**

---

## 8. execution_horizon 为什么有用

假设旧 chunk 剩余轨迹和新 observation 预测出来的轨迹差别很大。

如果完全不做过渡：

```text
old  ---------------->
                    \
                     new
```

chunk 接管处可能出现：

- joint position jump；
- joint velocity jump；
- end-effector direction sudden change；
- gripper jitter；
- jerk 增大。

RTC guidance 让新 chunk 在一段 horizon 内先尊重旧轨迹，再逐渐释放：

```text
old  -------------------
                       \
                        \
                         ---------- new
```

因此：

### H 较小

```text
execution_horizon ↓
```

通常意味着：

- 新 observation 更快接管；
- responsiveness ↑；
- trajectory inertia ↓；
- 但 chunk boundary discontinuity / jerk 可能 ↑。

### H 较大

```text
execution_horizon ↑
```

通常意味着：

- transition 更缓；
- smoothness ↑；
- chunk boundary jerk ↓；
- 但新 trajectory 更久受到旧 trajectory 约束；
- reactivity ↓；
- 任务可能表现出“黏”“迟钝”“惯性大”。

因此：


$$
\boxed{
H \text{ 控制的是单次 chunk 切换的过渡时间尺度}
}
$$

而不是 chunk 的实际执行长度。

---

## 9. execution_horizon 与 inference_delay 的关系

一个比较自然的工作区间是：


$$
H \gtrsim d
$$

原因是：

- inference 期间旧 chunk 已经实际执行了约 $d$ 步；
- RTC 需要对这段时间区域进行一致性约束和时间对齐；
- 如果 $H$ 明显小于 $d$，consistency horizon 甚至覆盖不了 inference delay 对应的区域。

例如：

```text
d = 6
H = 12
```

可以理解成：

```text
0        6            12
|--------|-------------|
强一致性    渐进释放
██████████▓▓▓▒▒░░░░░░░
```

一个适合真机调参的经验关系是：


$$
\boxed{
d \lesssim H \lesssim N_{\text{exec}}
}
$$

但需要强调：

- `H < N_exec` 不是代码层面的硬约束；
- 它只是通常较自然的调参区域；
- 若 $H \gg N_{\text{exec}}$，系统可能在每次 replanning 后长期处于旧轨迹约束中，响应性下降。

---

## 10. queue_threshold 与 execution_horizon 是两个不同方向

这是整个 RTC 调参最容易混淆的地方。

### queue_threshold：控制“多久换一次”


$$
N_{\text{exec}}\approx C-Q
$$

它主要影响：

- chunk 切换频率；
- replanning cadence；
- queue starvation 风险；
- 对新 observation 的更新频率。

### execution_horizon：控制“换的时候过渡多久”

它主要影响：

- 单次 chunk boundary 的连续性；
- smoothness；
- jerk；
- responsiveness。

因此整体顿挫可以粗略理解为：


$$
\text{perceived jerk}
\sim
\text{switch frequency}
\times
\text{discontinuity per switch}
$$

其中：

```text
switch frequency
    <- 主要由 queue_threshold 控制

discontinuity per switch
    <- 主要由 execution_horizon
       和 max_guidance_weight 控制
```

因此仅仅降低 chunk 切换频率并不是唯一手段。

---

## 11. max_guidance_weight 的作用

`execution_horizon` 决定：

> previous chunk 影响新 chunk **持续多久**。

`max_guidance_weight` 决定：

> 这段时间内 previous chunk **拉新 chunk 有多强**。

可以理解为：

```text
H 大 + weight 大
-> 很保守地继承旧轨迹
-> 很平滑，但可能明显迟钝

H 小 + weight 大
-> 短时间强制对齐，然后快速释放

H 大 + weight 小
-> 长时间温和参考旧轨迹

H 小 + weight 小
-> RTC 很弱，接近普通异步 chunk prediction
```

---

## 12. 对低动态真机任务的调参策略

适用场景：

- 桌面 manipulation；
- 目标基本静止；
- 环境变化慢；
- 不需要高频 replanning；
- 主要痛点是 chunk 交替造成的顿挫。

### Step 1：测 inference latency 分布

记录：

```text
mean
P50
P95
P99
max
```

转换成 control steps：


$$
d=\lceil T f\rceil
$$

重点使用 $d_{p95}$ 或 $d_{p99}$，不要只用平均延迟。

### Step 2：确定 queue_threshold 安全下界


$$
Q_{\min}=d_{p99}+M
$$

建议从 3-5 step safety margin 起步。

### Step 3：在安全范围内尽量减小 Q

因为你的主要目标不是快速响应，而是减少 chunk 切换：


$$
Q \downarrow
\Rightarrow
N_{\text{exec}}\uparrow
\Rightarrow
\text{switch frequency}\downarrow
$$

例如 `C=50`：

| Q | 近似 N_exec |
|---:|---:|
| 45 | 5 |
| 40 | 10 |
| 35 | 15 |
| 30 | 20 |
| 25 | 25 |
| 20 | 30 |
| 15 | 35 |

前提：`Q` 仍需明显高于最坏 inference delay。

### Step 4：固定 Q 后调 H

如果仍然能看到 chunk boundary jerk，再调整：

```text
execution_horizon
max_guidance_weight
```

对于低动态任务，可以适当增大 `H`，换取更平滑过渡。

### Step 5：区分“切得太频繁”和“每次切得太硬”

建议记录：

- `actual_chunk_lifetime`
- `inference_delay`
- chunk switch timestamp
- $\Delta a_{\text{switch}}$
- $\Delta^2 a_{\text{switch}}$
- joint velocity
- joint acceleration / jerk

否则仅凭肉眼看到的“顿挫”，容易混淆：

1. chunk 切换频率太高；
2. 单次 chunk 接管不够平滑。

---

## 13. 推荐的参数思维模型

假设：

```text
chunk_size = 50
control = 30 Hz
P99 inference_delay = 10 step
```

针对低动态平滑任务，可以先试：

```text
queue_threshold = 15 ~ 20
```

得到：


$$
N_{\text{exec}}\approx30\sim35
$$

再尝试：

```text
execution_horizon = 10 ~ 15
```

形成：

```text
0       d=10       H=12~15                   N=30~35
|---------|-----------|-------------------------|
推理时序对齐   RTC平滑释放          新chunk自由执行
```

这个思路比把 `execution_horizon` 也设置到 30 更合理，因为后者可能让系统过度继承旧轨迹。

---

## 14. 一个非常重要的判断

如果实验目标是：

> “严格每 N 个 action 换一次 chunk”

那么 LeRobot 当前 RTC 的 `queue_threshold` 机制只能近似做到：


$$
N\approx C-Q
$$

而不能 hard guarantee。

真实系统更接近：


$$
N_{\text{actual}}
=
C-Q+\epsilon
$$

其中 $\epsilon$ 来自：

- latency estimation error；
- latency jitter；
- thread scheduling；
- control-loop jitter；
- queue merge timing；
- `ceil()` 离散化。

如果论文实验必须严格比较：

```text
5 / 10 / 20 / 30 actions per chunk
```

应在 `RTCInferenceEngine` 中增加显式的：

```text
target_execution_steps
```

并根据预测的 inference delay 提前启动下一轮 inference，而不是等执行满 N 步后才开始算。

更合理的触发逻辑应是：


$$
\text{launch step}
\approx
N_{\text{target}}-d_{\text{pred}}
$$

从而让新 chunk 尽量在第 $N_{\text{target}}$ 步附近 ready。

---

## 15. 最终结论

> **Summary**
> 参数职责
> **`queue_threshold` 决定“多久换一次 chunk”。**  
> 稳态近似：
>
> 

$$
> N_{\text{exec}}\approx chunk\_size-queue\_threshold
> 
$$

>
> **`execution_horizon` 决定“换 chunk 时过渡多久”。**  
> 它不会直接规定每个 chunk 实际执行多少 action。
>
> **`inference_delay` 决定推理期间物理时间推进了多少。**  
> `queue_threshold` 必须给它留出足够安全余量。
>
> **`max_guidance_weight` 决定过渡约束有多强。**

对于低动态、主要追求真机平滑的任务，推荐思路是：


$$
\boxed{
Q \text{ 在不 starvation 的前提下尽量小}
}
$$

同时：


$$
\boxed{
H \text{ 适度大于 inference delay，用于平滑 chunk boundary}
}
$$

即：

```text
减少 chunk 切换次数
        +
降低单次 chunk 切换的不连续
        =
更平滑的真机运动
```

## 16. 速查表

| 问题 | 优先调哪个参数 |
|---|---|
| chunk 换得太频繁 | ↓ `queue_threshold` |
| queue 偶尔耗尽/机器人等动作 | ↑ `queue_threshold` |
| chunk 切换瞬间明显 jerk | ↑ `execution_horizon` 或适度 ↑ guidance weight |
| 机械臂太“黏”、响应新目标太慢 | ↓ `execution_horizon` / ↓ guidance weight |
| 希望更高 replanning 频率 | ↑ `queue_threshold` |
| 希望一个 chunk 执行更久 | ↓ `queue_threshold` |
| 希望严格固定 N steps/chunk | 需修改 RTC engine，当前参数只能近似 |
