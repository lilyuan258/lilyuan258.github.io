---
title: RTC 中实时约束与 `d <= s <= H-d` 条件的理解
description: >-
  本文档总结关于 RTC（Real Time Chunking）中实时执行约束的讨论，重点澄清 s 、 d 、 H 的含义，以及为什么不能把 d <= H s
  直接解释成“开始推理后旧 chunk 还够用”。
pubDate: '2026-06-18'
updatedDate: '2026-06-18'
tags: []
draft: false
source: RTC_realtime_constraint_summary_obsidian.md
wordCount: 1482
readingTime: 3
---
<img src="../../notes-assets/rtc-1.jpg" alt="rtc_1.jpg" loading="lazy" width="533" />
<img src="../../notes-assets/rtc-2.jpg" alt="rtc_2.jpg" loading="lazy" width="532" height="709" />
# RTC 中实时约束与 `d <= s <= H-d` 条件的理解

本文档总结关于 RTC（Real-Time Chunking）中实时执行约束的讨论，重点澄清 `s`、`d`、`H` 的含义，以及为什么不能把 `d <= H-s` 直接解释成“开始推理后旧 chunk 还够用”。

说明：本文使用 Obsidian/MathJax 兼容写法。独立公式统一使用 `$$...$$`，行内公式尽量使用普通文本或 `$...$`。

## 1. 基本符号

设一个动作 chunk 的长度为：

$$
H
$$

当前正在执行的旧 chunk 记为：

$$
A^{old} = [a^{old}_0, a^{old}_1, \ldots, a^{old}_{H-1}]
$$

其中每个 $a_i$ 是一个控制周期要执行的动作。

设模型推理延迟为：

$$
d
$$

也就是说，从启动新 chunk 的推理，到新 chunk 可用，中间会经过 `d` 个控制步。

设 execution horizon / chunk switch step 为：

$$
s
$$

这里的 `s` 表示：系统计划在旧 chunk 的第 `s` 步切换到新 chunk。

注意：在 RTC 论文语境下，`s` 不是“开始推理的时刻”，而是“希望新 chunk 可用并开始切换的时刻”。这是最容易混淆的点。

## 2. 正确的 RTC 时间线

如果希望在第 `s` 步切换到新 chunk，而推理延迟为 `d` 步，那么新 chunk 的推理必须提前启动：

$$
\text{start inference at } s-d
$$

经过 `d` 个控制步后，新 chunk 在第 `s` 步可用：

$$
\text{new chunk ready at } s
$$

因此正确时间线是：

```text
step s-d                step s
   |----------------------|
   |   inference latency  |
   |----------------------|
start new inference     new chunk ready / switch
```

在这段推理期间，控制器仍然执行旧 chunk 中的动作：

$$
a^{old}_{s-d}, a^{old}_{s-d+1}, \ldots, a^{old}_{s-1}
$$

一旦新 chunk 生成完成，控制器从新 chunk 中尚未过期的动作开始执行。

## 3. 为什么实时性要求是 `d <= s`

推理从 `s-d` 开始。为了这个启动时刻合法，必须有：

$$
s-d \ge 0
$$

整理得到：

$$
d \le s
$$

这就是 RTC 中保证异步推理可以提前启动的下界条件。

更具体地说，在推理期间的第 `j` 个控制步，`j = 0, 1, ..., d-1`，控制器执行：

$$
a^{old}_{s-d+j}
$$

这些索引必须合法：

$$
0 \le s-d+j \le s-1 < H
$$

其中左侧合法性需要：

$$
d \le s
$$

所以，`d <= s` 的含义是：在计划切换点 `s` 之前，系统有足够的时间提前启动新 chunk 的推理，并且在推理期间旧 chunk 可以继续供给动作。

## 4. 新 chunk 的前 `d` 个动作为什么已经“过期”

新 chunk 是在第 `s-d` 步根据当时观测开始生成的。因此从生成模型的角度看，新 chunk 形式上是从时间 `s-d` 开始的一段动作序列：

$$
A^{new} = [a^{new}_0, a^{new}_1, \ldots, a^{new}_{H-1}]
$$

其中：

$$
a^{new}_0, \ldots, a^{new}_{d-1}
$$

对应的物理时间是：

$$
s-d, \ldots, s-1
$$

但是这些时间步在新 chunk 生成完成时已经过去了。它们实际上已经由旧 chunk 执行：

$$
a^{old}_{s-d}, \ldots, a^{old}_{s-1}
$$

因此，切换时真正可执行的新动作不是 $a^{new}_0$，而是：

$$
a^{new}_d
$$

时间线可以写成：

推理期间执行旧 chunk：

$$
a^{old}_{s-d}, \ldots, a^{old}_{s-1}
$$

新 chunk 生成完成后，从第 `d` 个新动作开始执行：

$$
a^{new}_d, a^{new}_{d+1}, \ldots
$$

因此整体切换关系是：

$$
a^{old}_{s-d}, \ldots, a^{old}_{s-1}
\quad \rightarrow \quad
 a^{new}_d, a^{new}_{d+1}, \ldots
$$

RTC 因此会把新 chunk 的前 `d` 个动作视为 frozen prefix，让它们与旧 chunk 中已经确定会执行的动作对齐。这不是为了执行这些动作，而是为了保证新 chunk 的生成过程与实际已经发生的轨迹一致。

## 5. 那么 `d <= H-s` 的作用是什么？

在论文语境中，常见条件是：

$$
d \le s \le H-d
$$

其中：

$$
s \le H-d
$$

等价于：

$$
d \le H-s
$$

这个条件不是主要用来说明“推理期间旧 chunk 是否够用”。因为推理不是从 `s` 开始，而是从 `s-d` 开始。

`d <= H-s` 更准确的含义是：切换点 `s` 不能太靠近旧 chunk 的末尾。这样，在新 chunk 生成出来并准备切换时，旧 chunk 与新 chunk 在未来区域仍有足够的重叠部分，可以用于 RTC 的 inpainting / soft masking 约束。

也就是说：

$$
d \le s
$$

主要保证实时调度可行。

$$
s \le H-d
$$

主要保证还有足够的后续重叠区域，用于连续性约束。

## 6. 前面误解的修正

前面曾把 `d <= H-s` 解释为：

> 从第 `s` 步开始推理后，旧 chunk 剩余 `H-s` 个动作，必须覆盖 `d` 步推理延迟。

这个解释只有在另一种坐标定义下才成立：即把 `s` 定义为“开始推理的时刻”。

但在 RTC 论文语境中，`s` 通常表示 execution horizon / switch step，即新 chunk 应该在第 `s` 步可用并开始切换。因此推理启动时刻是：

$$
s-d
$$

所以，在论文的定义下，实时可用性的关键条件应理解为：

$$
d \le s
$$

而不是：

$$
d \le H-s
$$

后者对应的是：

$$
s \le H-d
$$

它约束的是切换点不能太晚，以便保留足够重叠区域。

## 7. 总结

RTC 的时间调度可以浓缩为一句话：

> 如果希望在旧 chunk 的第 `s` 步切换到新 chunk，而模型推理延迟为 `d` 步，则必须在第 `s-d` 步启动新推理；推理期间继续执行旧 chunk，到第 `s` 步新 chunk 可用，然后从新 chunk 的第 `d` 个动作开始接管。

因此：

$$
d \le s
$$

保证推理可以提前启动，并且切换前总有旧动作可执行。

$$
s \le H-d
$$

保证切换点不能过晚，使旧 chunk 与新 chunk 之间仍有足够重叠区域进行 frozen prefix、inpainting 和 soft masking，从而提升动作连续性。
