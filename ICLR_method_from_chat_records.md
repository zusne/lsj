# 按顺序完整学习聊天记录后的 ICLR 方法方案与定理清单

## 0. 本次学习范围

我按照用户指定的顺序，对仓库中的 16 个 `.docx` 聊天记录进行了重新提取和顺序阅读。顺序如下：

| 顺序 | 文件名 | 提取文本规模 | 学习重点 |
|---:|---|---:|---|
| 1 | `1新论文1，2.docx` | 391,758 chars | 新论文 1/2 的形成、缺陷、代码实现争议、MP 谱正则与自由卷积诊断 |
| 2 | `新论文3_1.docx` | 169,311 chars | 新论文 3 的起点：拒绝全谱矫正，转向 BBP 谱刺 |
| 3 | `新论文3_2.docx` | 174,506 chars | 谱刺、Bellman 下降、闭环证明、下降引理与算法特异性 |
| 4 | `新论文3_3.docx` | 278,577 chars | BBP double-scaling、alignment、阈值、目标网络下 Bellman 线性化 |
| 5 | `新论文3_4.docx` | 287,379 chars | 单步谱刺加速、safe gamma、方向导数、闭环增长 |
| 6 | `新论文3_5.docx` | 199,087 chars | 证明版本反复修正，去除不自然假设，聚焦谱刺如何真实影响 TD |
| 7 | `新论文3_6.docx` | 217,475 chars | NTK、有限宽、奇异值与核特征值关系、残差项讨论 |
| 8 | `新论文3_7.docx` | 202,154 chars | Bellman error、TD operator、target network 与局部线性化 |
| 9 | `新论文3_8.docx` | 529,812 chars | 大量定理/附录草稿，C1 条件、对角化、row-wise margin |
| 10 | `新论文3_9.docx` | 451,907 chars | 对证明假设的逐步削弱，E[K]、有限宽 K(W)、残差 R(W) |
| 11 | `新论文3_10.docx` | 275,798 chars | 正文定理精简、附录证明展开、期望与概率版本区分 |
| 12 | `新论文3_11.docx` | 232,133 chars | 自然假设、简化对角改进证明、BBP 信息如何进入证明 |
| 13 | `新论文3_12docx.docx` | 235,358 chars | 简化附录、正文定理选择、rank-one NTK enhancement 与 C1 |
| 14 | `新论文3_13.docx` | 196,885 chars | 定理 1 与定理 2 如何衔接，E[K] 是什么，正文列哪些定理 |
| 15 | `新论文3_14.docx` | 217,830 chars | 明确拒绝强行把 tau 塞进 kappa，要求从 \hat v 出发推导 |
| 16 | `新论文3_15.docx` | 77,222 chars | 最终关键转向：不要强行写 rank-one NTK，改用 spike-induced tangent feature 与 TD curvature |

总的学习结论是：这些聊天记录不是单一方案的简单累加，而是一条不断纠错的研究路线。前期的 MP 全谱正则与 Pastur/free-convolution 想法提供了随机矩阵背景，但最终可投 ICLR 的主线应当是：**用 BBP 阈上谱刺诱导任务相关 tangent feature，从而改善 critic 的局部 TD/Bellman 优化几何**。

---

## 1. 全部聊天记录中的思想演化

### 1.1 新论文 1：MP 全谱匹配/奇异值分布矫正

最早的想法是让深度强化学习 critic 的权重奇异值分布逼近 Marchenko--Pastur 分布，试图用随机矩阵的“健康谱”稳定训练。后续聊天中逐步暴露出几个核心问题：

1. **方法目标不清楚**：为什么 MP 分布就是训练中最优的谱，缺少直接逻辑。
2. **实现成本高**：每次训练都做完整 SVD，会成为明显负担。
3. **目标谱不自适应**：不同任务、不同训练阶段的理想谱不应完全一样。
4. **和 RL/Bellman 改善关系弱**：谱匹配本身很难说明 TD error 或 Bellman contraction 为什么改善。
5. **工程细节反复出错**：例如 Q1/Q2 是否都正则、取哪一层、最后一层奇异值太少、倒数第二层遇到 ReLU、经验谱概率 `p_emp=s2/s2.sum()` 等问题。

因此，新论文 1 不能作为最终 ICLR 主方法，只能作为 baseline/ablation：**MP spectrum matching regularization**。

### 1.2 新论文 2：MP ⊞ SC / Pastur 自洽方程 / 训练后谱诊断

第二条路线把训练后的权重谱看作初始 MP 谱与训练扰动谱的自由卷积，例如 MP 与 semicircle/free additive noise 的组合。后续讨论中已经形成明确判断：

- 把 Pastur 方程嵌入每一步训练不现实；
- 每步估计扰动方差、求 Stieltjes transform、自洽迭代，训练成本过高；
- 但训练后比较经验谱与理论预测谱，可以作为有价值的随机矩阵诊断。

所以新论文 2 应降级为附录/诊断实验：**post-hoc free-probability spectral diagnostic**。它可以增强论文随机矩阵叙事，但不应作为主算法。

### 1.3 新论文 3 的约束：必须同时满足四点

从 `新论文3_1.docx` 开始，用户反复强调最终方法必须满足：

1. **真的使用随机矩阵理论**，而不是只借用“谱正则”这个外壳；
2. **不能做全谱 SVD 或主观指定理想谱**；
3. **必须和 RL/Bellman/TD 改善闭环**，不能只说谱更好；
4. **证明必须从方法特异性出发**，不能换成任意额外 loss 都成立。

这四点排除了“普通谱范数正则”“MP 全谱匹配”“主观选择前 k 个奇异值”等方案，也推动方法转向 BBP 谱刺。

### 1.4 BBP 谱刺想法的形成

后续聊天逐步形成一个核心直觉：

> 不要矫正整个奇异值谱，而是控制一个顶端 spectral spike，使其越过随机矩阵 bulk edge。根据 BBP 相变，阈上 spike 的奇异向量会与真实信号方向产生非消失对齐；阈下 spike 则仍然类似噪声。

这个思路解决了前面的问题：

- 不需要完整谱，只需要 top singular triplet；
- 不需要知道完整理想谱，只需要估计 bulk edge 和 spike gap；
- 随机矩阵理论不再是装饰，而是通过 BBP phase transition 给出“阈上才有信号对齐”的机制；
- 与 RL 的连接可以通过 spike direction 诱导的 tangent feature 和 TD curvature 建立。

### 1.5 早期谱刺证明的问题

中期大量聊天尝试证明：

\[
\mathbb E[K] = K_0 + \kappa(\theta)\tau^2 gg^\top.
\]

但用户多次指出关键问题：

1. 不能把 `K` 和 `E[K]` 混用；
2. 不能在 lemma 给出期望核分解，却在 theorem 中直接使用有限宽随机核；
3. 不能先证明一个不含 `tau` 的式子，最后强行令 `κ_0(θ)=κ(θ)τ²`；
4. 不能随意假设对角项大于非对角项，或强行假设 row-wise margin；
5. 不能为了正文好看隐藏掉关键残差 `R(W)` 而不说明概率界。

这些批评是最终方案必须吸收的：**最终理论不能以强行 rank-one NTK 分解为起点**。

### 1.6 最终关键转向：从 `\hat v` 出发，而不是从 `K=K0+rank-one` 出发

最后几份记录中，最重要的转向是：

> 不再直接假设或强行证明整个 NTK 出现 `κτ²ggᵀ`，而是从谱刺的 top right singular vector `\hat v` 出发，直接推导沿 `\hat u\hat v^T` 的一阶 perturbation 对 Q 网络输出的影响。

这得到 spike-induced tangent feature：

\[
\psi_i
=
\left.\frac{\partial Q_{W+\alpha \hat u\hat v^\top}(h_i)}{\partial \alpha}\right|_{\alpha=0}
=
S_i(W)\,\hat v^\top h_i.
\]

再用 BBP alignment：

\[
\hat v=\tau v^\star+\sqrt{1-\tau^2}v_\perp,
\qquad
h_i=T_i v^\star+h_i^\perp,
\]

得到

\[
\psi_i
=
\tau S_i(W)T_i
+
\sqrt{1-\tau^2}S_i(W)v_\perp^\top h_i^\perp.
\]

也就是

\[
\psi = \tau g + \sqrt{1-\tau^2}\xi.
\]

这是整套聊天记录中最稳定、最不容易被审稿人攻击的理论入口。

---

## 2. 最终方法：BBP-guided Spectral-Spike TD

### 2.1 方法名称

建议命名为：

**SST: Spectral-Spike TD**

更完整标题可以是：

**BBP-Guided Spectral Spikes for Stable Temporal-Difference Learning**

### 2.2 方法目标

最终方法不是“让谱更像 MP”，也不是“惩罚最大奇异值”。最终目标是：

> 在 critic/value 网络中产生一个受控、阈上、TD 有用的 top spectral spike，使其 top singular direction 通过 BBP 相变获得任务信号对齐，并由此产生新的有用 tangent feature，形成与当前 Bellman residual 对齐的有效更新方向，并直接降低固定 target 下的经验 Bellman loss。

### 2.3 网络与符号

考虑 critic/value 网络第一层权重：

\[
W\in\mathbb R^{m\times n}.
\]

设 top singular triplet 为：

\[
W\hat v=s_1\hat u,
\qquad
\|\hat u\|_2=\|\hat v\|_2=1.
\]

对一个 mini-batch 的 hidden/input feature `h_i`，critic 输出为 `Q_W(h_i)`，TD residual 为：

\[
\delta_i = Q_W(h_i)-y_i.
\]

### 2.4 TD-aware 谱刺方向

沿 top singular direction 的一阶扰动为：

\[
W(\alpha)=W+\alpha \hat u\hat v^\top.
\]

定义方向导数：

\[
\psi_i
=
\left.\frac{\partial Q_{W(\alpha)}(h_i)}{\partial \alpha}\right|_{\alpha=0}.
\]

经验 TD loss 的方向导数为：

\[
a_t
=
\left.\frac{\partial}{\partial \alpha}
\frac{1}{2B}\sum_{i=1}^B
\big(Q_{W+\alpha\hat u\hat v^\top}(h_i)-y_i\big)^2
\right|_{\alpha=0}
=
\frac{1}{B}\sum_{i=1}^B \delta_i\psi_i.
\]

这个量非常重要：它把“谱刺”与“TD/Bellman 下降”闭环起来。最终方法不应盲目增大 `s_1`，而应当在谱刺方向对 TD loss 有利或至少不冲突时激活谱刺。

### 2.5 损失函数

最终建议的 critic loss 为：

\[
\mathcal L_{\rm SST}
=
\mathcal L_{\rm TD}
+
\lambda_{\rm sp}\,G_t\,\mathcal L_{\rm spike}
+
\lambda_{\rm bulk}\,\mathcal L_{\rm bulk}.
\]

其中

\[
\mathcal L_{\rm TD}
=
\frac{1}{2B}\sum_{i=1}^B (Q_W(h_i)-y_i)^2.
\]

设 bulk edge 估计为 `b_t`，margin 为 `\Delta>0`，目标阈值：

\[
T_t=b_t+\Delta.
\]

谱刺项：

\[
\mathcal L_{\rm spike}(W)
=
\big[T_t-s_1(W)\big]_+^2.
\]

这个 loss 在 `s_1<T_t` 时推动 `s_1` 增大，越过 bulk edge；在 `s_1\ge T_t` 后自动关闭。

关键 gating：

\[
G_t
=
\mathbf 1\{s_1<T_t\}\cdot
\mathbf 1\{a_t\le -\zeta\},
\]

其中 `\zeta\ge 0` 是一个小阈值。也就是说，只有当：

1. spike 还没有越过目标阈值；
2. 增大 spike 的方向与 TD loss 下降方向一致；

才激活 spike pressure。

Bulk control 用于避免整个谱失控，例如：

\[
W_{\rm bulk}=W-s_1\hat u\hat v^\top,
\]

\[
\mathcal L_{\rm bulk}
=
\left[\frac{\|W_{\rm bulk}\|_F^2}{mn}-c_{\rm bulk}\right]_+^2,
\]

或对前若干非 top singular values 做轻量约束。它不是为了匹配 MP 全谱，而是为了防止除 top spike 外的 bulk 膨胀。

### 2.6 为什么这个方法比前一版更合理

相比上一版只写 `L_spike=[T-s_1]_+^2`，这里加入了 TD-aware gate `a_t≤-ζ`，它吸收了聊天记录中最关键的要求：

- 不是盲目促进最大奇异值；
- 不是和谱范数正则反着干；
- 谱刺必须与 Bellman/TD 下降闭环；
- 换成任意额外 loss 不一定成立，因为证明和算法都使用了特定方向 `\hat u\hat v^T` 及其 tangent feature `ψ`。

### 2.7 伪代码

```python
for each critic update:
    batch = replay.sample()
    td_loss, td_residuals = compute_td_loss_and_residuals(critic, target, batch)

    W = critic.first_layer.weight
    u1, s1, v1 = top_singular_triplet_by_power_iteration(W)

    psi = directional_tangent_feature(critic, batch.features, u1, v1)
    a_t = mean(td_residuals * psi.detach())

    bulk_edge = estimate_bulk_edge(W, excluding_top=True)
    T = bulk_edge + margin

    spike_loss = relu(T - s1) ** 2
    W_bulk = W - s1 * outer(u1, v1)
    bulk_loss = relu(mean_square(W_bulk) - target_bulk_energy) ** 2

    gate = float((s1 < T) and (a_t <= -zeta))

    loss = td_loss + lambda_sp * gate * spike_loss + lambda_bulk * bulk_loss
    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(critic.parameters(), max_norm)
    optimizer.step()
```

---

## 3. 理论路线的最终整理：不采用 TD operator，直接证明 Bellman loss 下降

### 3.1 最终选择与证明目标

最终理论不再使用

\[
A(\Phi)=\Phi^\top\Xi(I-\gamma P^\pi)\Phi,
\]

也不再通过 TD fixed point、非对称谱、条件数或

\[
|e_{\psi,t}|\le (1-\eta\beta_0)^t|e_{\psi,0}|
\]

证明方法有效。原因不是 TD operator 没有意义，而是这条路线需要局部线性化、稳定子空间、对称化、特征正交化以及“新增特征恰好改善最慢方向”等额外条件；这些条件会遮蔽本文真正想说明的谱刺机制。

正文理论只证明以下局部而清楚的结论：

> 固定当前 critic update 的 Bellman target。如果 top singular vector 已经处于 BBP 超临界对齐区间，并且其中的任务信号与当前 Bellman residual 构成下降相关，而正交部分只是中心化随机扰动，那么沿 top singular direction 的一个足够小更新，以高概率严格降低当前经验 Bellman loss。

证明闭环为

\[
\boxed{
\text{BBP 超临界对齐}
\Rightarrow
\text{任务相关 tangent feature}
\Rightarrow
\text{负方向导数}
\Rightarrow
\text{Bellman loss 严格下降}.}
\]

这一路线不修改 SST 算法。理论还会单独证明：现有 hinge spike regularizer 的梯度分量恰好沿 \(\hat u\hat v^\top\) 方向。

### 3.2 问题设置

对 minibatch \(\{(h_i,y_i)\}_{i=1}^N\)，定义固定 target 下的经验 Bellman loss：

\[
\mathcal B(W)
:=\frac{1}{2N}\sum_{i=1}^N\delta_i(W)^2,
\qquad
\delta_i(W):=Q_W(h_i)-y_i.
\]

分析两层 critic

\[
Q_W(h)=\sum_{k=1}^m o_k\varphi(w_k^\top h),
\qquad W=[w_1,\ldots,w_m]^\top.
\]

令 \((s_1,\hat u,\hat v)\) 是 \(W\) 的 top singular triplet，并定义

\[
D:=\hat u\hat v^\top,
\qquad \|D\|_F=1.
\]

理论研究一维路径

\[
W(\alpha)=W+\alpha D
\]

上的 loss 变化，而不分析 TD 参数递推。

### 3.3 四个最终假设

#### A1：Frozen target

在当前被分析的 critic update 中，\(y_i\) 对 \(W\) stop-gradient。这个假设只固定单次更新中的 target，不要求整个训练期间 target 不变化。

#### A2：Directional smoothness

存在 \(L_B<\infty\)，使充分小的 \(\alpha\) 满足

\[
\mathcal B(W+\alpha D)
\le
\mathcal B(W)
+\alpha\langle\nabla_W\mathcal B(W),D\rangle
+\frac{L_B}{2}\alpha^2.
\]

这只是标准 descent lemma 在谱刺方向上的局部版本，不要求全局凸性。

#### A3：BBP supercritical alignment

在谱刺超过 BBP 阈值后，top right singular vector 满足

\[
\hat v
=\tau v^\star+\sqrt{1-\tau^2}\,v_\perp,
\qquad
\tau\ge\tau_0>0,
\qquad
v_\perp\perp v^\star.
\]

\(v^\star\) 是潜在任务信号方向，\(v_\perp\) 是正交 bulk 方向。\(\tau\ge\tau_0\) 表示阈上对齐不会随维度消失。

#### A4：Useful signal and centered orthogonal fluctuation

谱刺诱导的 feature 将被证明可写为

\[
\psi_i=\tau g_i+\sqrt{1-\tau^2}\xi_i.
\]

任务信号与当前 residual 满足

\[
-\frac1N\sum_{i=1}^N\delta_i g_i\ge c_g>0.
\]

这不是预先假设整个谱刺方向已经是下降方向，而只要求可解释的任务信号部分是有用的。正交项 \(X_i:=\delta_i\xi_i\) 条件中心化且为 \(\sigma^2\)-sub-Gaussian；独立性可以替换为给出同型集中界的 martingale 或 mixing 条件。

### 3.4 正文引理：谱刺诱导任务相关 tangent feature

定义

\[
\psi_i
:=\left.\frac{\mathrm d}{\mathrm d\alpha}
Q_{W+\alpha D}(h_i)\right|_{\alpha=0}.
\]

把表示写成

\[
h_i=T_i v^\star+h_i^\perp,
\qquad (v^\star)^\top h_i^\perp=0.
\]

由链式法则，

\[
\psi_i=S_i(W)\hat v^\top h_i,
\qquad
S_i(W):=\sum_{k=1}^m o_k\hat u_k\varphi'(w_k^\top h_i).
\]

代入 BBP 分解得到

\[
\boxed{
\psi_i=\tau g_i+\sqrt{1-\tau^2}\xi_i,}
\]

其中

\[
g_i:=S_i(W)T_i,
\qquad
\xi_i:=S_i(W)v_\perp^\top h_i^\perp.
\]

这个引理的意义是从网络结构和 top singular vector 严格推出 \(\tau\) 如何进入输出变化，不再假设 NTK 被人为加上 rank-one 矩阵。

### 3.5 正文主定理：BBP 阈上谱刺直接降低 Bellman loss

对任意 \(\delta\in(0,1)\)，定义

\[
\varepsilon_N(\delta)
:=\sigma\sqrt{\frac{2\log(2/\delta)}{N}},
\]

以及净下降 margin

\[
\mu_N(\delta)
:=\tau_0c_g
-\sqrt{1-\tau_0^2}\,\varepsilon_N(\delta).
\]

**Theorem（Direct Bellman-loss decrease above the BBP threshold）.** 在 A1--A4 下，如果

\[
\mu_N(\delta)>0,
\]

那么以至少 \(1-\delta\) 的概率，对任意

\[
0<\alpha\le\frac{\mu_N(\delta)}{L_B},
\]

都有

\[
\boxed{
\mathcal B(W+\alpha\hat u\hat v^\top)
\le
\mathcal B(W)-\frac{\alpha\mu_N(\delta)}{2}.}
\]

该定理应是正文唯一的核心效果定理。它清楚表达：BBP 对齐提供任务信号，有限样本正交噪声只损失一个 \(O(N^{-1/2})\) margin，最终得到严格 Bellman-loss decrease。

### 3.6 正文推论：超临界收益为常数量级

如果

\[
N\ge
\frac{8\sigma^2(1-\tau_0^2)\log(2/\delta)}
{\tau_0^2c_g^2},
\]

则

\[
\mu_N(\delta)\ge\frac{\tau_0c_g}{2}.
\]

因而对

\[
0<\alpha\le\frac{\tau_0c_g}{2L_B},
\]

有

\[
\mathcal B(W+\alpha D)
\le
\mathcal B(W)-\frac{\alpha\tau_0c_g}{4}.
\]

这说明阈上对齐带来的保证是常数量级，而不是随维度消失的微小收益。

## 4. 附录中的完整证明链

### 4.1 Step 1：精确计算谱刺方向的输出导数

第 \(k\) 行权重沿 \(D\) 变化为

\[
w_k(\alpha)=w_k+\alpha\hat u_k\hat v.
\]

所以

\[
Q_{W+\alpha D}(h_i)
=\sum_{k=1}^m o_k
\varphi((w_k+\alpha\hat u_k\hat v)^\top h_i).
\]

对 \(\alpha\) 求导并令 \(\alpha=0\)：

\[
\begin{aligned}
\psi_i
&=\sum_{k=1}^m o_k\hat u_k
\varphi'(w_k^\top h_i)\hat v^\top h_i\\
&=S_i(W)\hat v^\top h_i.
\end{aligned}
\]

再将 \(\hat v\) 与 \(h_i\) 的信号—正交分解代入，即得到

\[
\psi_i=\tau g_i+\sqrt{1-\tau^2}\xi_i.
\]

### 4.2 Step 2：控制正交噪声

令 \(X_i=\delta_i\xi_i\)。A4 下的 sub-Gaussian concentration 给出

\[
\Pr\left(
\left|\frac1N\sum_{i=1}^NX_i\right|\ge t
\right)
\le
2\exp\left(-\frac{Nt^2}{2\sigma^2}\right).
\]

取

\[
t=\sigma\sqrt{\frac{2\log(2/\delta)}{N}},
\]

得到以至少 \(1-\delta\) 的概率

\[
\left|\frac1N\sum_i\delta_i\xi_i\right|
\le\varepsilon_N(\delta).
\]

### 4.3 Step 3：得到负方向导数

固定 target 后，

\[
\begin{aligned}
\langle\nabla_W\mathcal B(W),D\rangle
&=\left.\frac{\mathrm d}{\mathrm d\alpha}
\mathcal B(W+\alpha D)\right|_{\alpha=0}\\
&=\frac1N\sum_i\delta_i\psi_i\\
&=\tau\frac1N\sum_i\delta_i g_i
+\sqrt{1-\tau^2}\frac1N\sum_i\delta_i\xi_i.
\end{aligned}
\]

A4 和集中界给出

\[
\langle\nabla_W\mathcal B(W),D\rangle
\le
-\tau c_g+\sqrt{1-\tau^2}\varepsilon_N(\delta).
\]

函数

\[
f(\tau)=\tau c_g-\sqrt{1-\tau^2}\varepsilon_N(\delta)
\]

在 \([0,1]\) 上单调递增。由 \(\tau\ge\tau_0\)，

\[
\langle\nabla_W\mathcal B(W),D\rangle
\le-\mu_N(\delta).
\]

### 4.4 Step 4：应用一维 descent lemma

由 A2，

\[
\mathcal B(W+\alpha D)
\le
\mathcal B(W)-\alpha\mu_N(\delta)
+\frac{L_B}{2}\alpha^2.
\]

当

\[
0<\alpha\le\frac{\mu_N(\delta)}{L_B}
\]

时，

\[
\frac{L_B}{2}\alpha^2
\le\frac{\alpha\mu_N(\delta)}{2},
\]

因此

\[
\mathcal B(W+\alpha D)
\le
\mathcal B(W)-\frac{\alpha\mu_N(\delta)}{2}.
\]

### 4.5 Proposition：现有 hinge regularizer 产生同一方向

现有算法使用

\[
\mathcal L_{\rm spike}(W)=(T-s_1(W))_+^2.
\]

当 \(s_1(W)<T\) 且最大奇异值单重时，

\[
\nabla_Ws_1(W)=\hat u\hat v^\top,
\]

因此

\[
\nabla_W\mathcal L_{\rm spike}(W)
=-2(T-s_1(W))\hat u\hat v^\top.
\]

单独看正则项贡献，其梯度下降更新为

\[
W^+
=W+2\eta_{\rm sp}(T-s_1(W))\hat u\hat v^\top.
\]

令

\[
\alpha=2\eta_{\rm sp}(T-s_1(W)),
\]

便与主定理分析的 \(W+\alpha D\) 完全一致。这一命题只解释现有算法的梯度方向，不增加 gate、不改变 loss，也不修改算法。

### 4.6 Corollary：有限宽度鲁棒性

若实际 feature 为

\[
\psi^{(m)}=\psi+r_m
\]

且

\[
\left|\frac1N\sum_i\delta_i r_{m,i}\right|
\le\frac{\mu_N(\delta)}{2},
\]

则实际方向导数至多为

\[
\frac1N\sum_i\delta_i\psi_i^{(m)}
\le-\frac{\mu_N(\delta)}{2}.
\]

对

\[
0<\alpha\le\frac{\mu_N(\delta)}{2L_B},
\]

同样得到

\[
\mathcal B(W+\alpha D)
\le
\mathcal B(W)-\frac{\alpha\mu_N(\delta)}{4}.
\]

有限宽度误差只使 margin 损失常数因子，不会在误差小于信号 margin 时破坏下降结论。

## 5. 正文与附录的最终排版

### 5.1 正文只保留

1. 固定 target Bellman loss 的定义；
2. 四个假设的简短自然语言版本；
3. Spike-induced tangent feature lemma；
4. Direct Bellman-loss decrease theorem；
5. 一段三行 proof sketch；
6. 常数量级阈上收益的 corollary。

正文 proof sketch 可以写成：链式法则给出

\[
\psi=\tau g+\sqrt{1-\tau^2}\xi.
\]

集中性与信号有用性给出

\[
\langle\nabla\mathcal B,D\rangle\le-\mu_N.
\]

最后由 directional smoothness 得到

\[
\mathcal B(W+\alpha D)
\le\mathcal B(W)-\alpha\mu_N/2.
\]

### 5.2 附录保留

1. tangent feature 的逐行链式法则；
2. 条件 sub-Gaussian concentration lemma；
3. 负方向导数 certificate；
4. 主定理完整证明；
5. 样本量 corollary 的代数推导；
6. hinge spike loss 与 \(\hat u\hat v^\top\) 的连接命题；
7. finite-width robustness；
8. scope statement。

### 5.3 不再放入最终理论的内容

以下内容不再作为本文定理或证明的一部分：

- \(A(\Phi)=\Phi^\top\Xi(I-\gamma P^\pi)\Phi\)；
- TD fixed-point error recursion；
- \(H=(M+M^\top)/2\)；
- \(H\)-orthogonal projection；
- 新增 feature 改善整体 condition number；
- 非对称 TD operator 的 diagonalizable 假设；
- \(|e_{\psi,t}|\le(1-\eta\beta_0)^t|e_{\psi,0}|\)。

这些内容可以留作未来工作或相关讨论，但不应与当前直接 Bellman-loss 定理混在一起。

### 5.4 理论结论的准确边界

本文证明的是：

> 在一个固定 target 的 critic update 中，满足 BBP 对齐与信号条件时，谱刺正则项所产生的 top singular direction component 是经验 Bellman loss 的严格下降方向。

本文没有证明：

1. moving target 下每一次完整 optimizer step 都单调下降；
2. actor 与 critic 联合训练的全局收敛；
3. 整个训练轨迹始终处于 BBP 模型；
4. 总梯度中的其他分量永远不会抵消谱刺分量；
5. 所有任务上的最终 return 必然提升。

这些边界必须明确写出，避免把局部机制 theorem 夸大为全局 RL convergence theorem。

## 6. 实验设计

### 6.1 主实验

- Online RL：SAC + SST，在 MuJoCo / Gymnasium continuous control 上测试；
- 备用：TD3 + SST；
- Offline RL：TD3+BC / IQL / CQL + SST，在 D4RL 上测试。

### 6.2 Baselines

必须包含：

1. vanilla SAC/TD3/IQL/CQL；
2. spectral norm regularization；
3. Frobenius/weight decay；
4. orthogonal regularization；
5. MP spectrum matching regularizer，即新论文 1 baseline；
6. spike without TD-aware gate；
7. spike without bulk control；
8. random rank-one direction instead of `\hat u\hat v^T`。

### 6.3 Diagnostics

论文是否有说服力，关键不只看 return，还要证明机制存在：

1. `s_1` 与 bulk edge 的 gap；
2. gate 激活频率；
3. `a_t=(1/B)Σδ_iψ_i` 的分布；
4. top singular vector 与任务 proxy 的 alignment；
5. 经验方向导数 `mean(δψ)` 与理论下降 margin `μ_N`；
6. 实际一步 loss 变化 `B(W_after)-B(W_before)`；
7. 正交扰动项 `mean(δξ)` 随 batch size 的集中趋势；
8. Bellman residual/TD loss 曲线；
9. seed variance 和 critic divergence rate；
10. spike-only、bulk-only、random-direction ablation。

### 6.4 最关键的实验现象

如果方法正确，应观察到：

- SST 让 `s_1` 稳定越过 bulk edge，而非全谱膨胀；
- 有 gate 的 SST 比无 gate 的 spike 更稳定；
- random rank-one spike 不如 `\hat u\hat v^T`；
- 阈上时 `mean(δψ)` 更稳定为负，且实际单步 Bellman loss 变化与定理预测一致；
- 正交扰动随 batch size 增大而集中，支持 A4 的随机正交性建模；
- SST 改善 TD loss 和 Bellman residual；
- return 提升或至少稳定性/方差显著改善。

---

## 7. 最终论文贡献表述

建议在论文 introduction 中写成三点：

1. **Method**：提出 Spectral-Spike TD，一种 BBP-guided critic regularization，通过 TD-aware gate 创建受控 spectral outlier，同时约束 bulk 谱稳定。
2. **Theory**：在固定 target 的局部更新中，证明阈上 spectral spike 的 top singular direction 诱导 task-aligned tangent feature，并以高概率提供常数量级 Bellman-loss 下降 margin。
3. **Experiments**：在 online/offline deep RL 中验证 SST 的性能与稳定性，并通过 spike gap、alignment、方向导数、噪声集中和实际单步 loss 变化验证理论机制。

最稳妥的一句话 claim 是：

> SST does not regularize the entire spectrum toward a prescribed shape; instead, it uses the BBP transition to create a controlled, task-aligned spectral outlier whose induced tangent feature provides a certified local Bellman-loss descent direction.

---

## 8. 下一步最应该做什么

1. 先写两页英文 LaTeX method + theory skeleton；
2. 实现 `top_singular_triplet_by_power_iteration`；
3. 实现 `directional_tangent_feature ψ` 和 `a_t=mean(δψ)`；
4. 加入 gated spike loss 与 bulk loss；
5. 在 SAC 的两个 MuJoCo 任务上跑 5 seeds pilot；
6. 同时记录 spike gap、gate、`mean(δψ)`、正交噪声项、单步 Bellman-loss 变化和 return；
7. pilot 成功后再扩展 D4RL 和完整 baseline。

---
