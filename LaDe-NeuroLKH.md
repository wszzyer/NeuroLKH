# [LaDe](https://huggingface.co/datasets/Cainiao-AI/LaDe) 数据集 + [Neural](https://github.com/liangxinedu/NeuroLKH) LKH

P.S. 其实不一定用 LKH 类算法。能验 LaDe Pointer Network 比传统 Pointer Network 效果好，那也有一定意义。比如可以将 LaDe 结合 [GreedRL](https://github.com/wangqianlongucas/Cainiao-GreedRL)。

## 数据生成

LaDe 数据集中每个快递员每轮配送大概 10 ～ 20 个邮件

### 路网规模

取一部分路网，大概 10000 节点，否则 GNN 效果不好。这部分路网就是该数据集所用的固定路网。

将所有不位于该路网范围内的物品删除。

### 问题规模

对于 TSP 问题的大小，尝试 100 和 1000 两种规模。

其中 TSP100，LKH 算法一般能瞬间求出最优解，在该规模上与 NeuralLKH 比不出什么来。

对于 TSP1000，LKH 算法求解速度稍慢，我们需要在该规模上验证 NeuralLKH 比 LKH 算法求解速度更快。

每次选取 10～100 个快递员，将他们的快递合并到一起，形成一个 100~1000 点的 TSP 问题。

我们简单认为每个快递位于最近的路网节点上，然后计算任意两个快递之间的最段路的到 CostMatrix。形成一个具有 1000 个 TSP 问题的训练集。

如果带时间窗口，需要对不同快递员的时间窗口作出一定调整，比如将时间加一个偏移，否则无论如何也送不完。

**这里不能将所有的快递员全都合并到一起，如果这样的话就只有一个样本，没法有监督训练**

### 问题种类

纯 TSP 问题的结构过于简单，LKH 本身的**启发的**启发值已经很好用，可以瞬间求出结果。所以考虑带约束的复杂一点的 TSP 问题，发挥 Neural 的优势。考虑如下 4 种问题。

- TSP：原始 TSP 问题

- TSPTW：带时间窗口

- TSPPD：既有取货，也有送货

- TSPPDTW：带时间窗口的取送


### 边权

因为快递员不一定总是走空间上的最段路，可能要考虑交通工具和道路拥挤程度。如果要参考快递员的真实行为，最好是使用时间上的最段路。这里可以两种边权都做一下实验。

#### 时间上的最段路
LaDe 数据集提供了快递员的 GPS 轨迹，可以通过该轨迹算出每条道路的通行时间作为边权（如果道路经过次数太少，可以根据其同类别（小路、主干路）的道路的时间进行推测。

#### 空间上的最段路
直接算道路长度作为边权。

### 标签生成
使用 LKH3 多跑一点时间生成一个较好的标签轨迹。

## Method

快递员的行为也许对启发值的预测有帮助，我们将在神经网络中加入快递员的行为，有以下两种方式：

- 快递员行为作为神经网络输入
- 快递员行为作为预训练监督信号



### 1、快递员行为作为神经网络输入

#### 快递员行为的 OD 矩阵计算

虽然 LaDe 数据集提供了快递员的 GPS 路径，但是其粒度太细了，需要搭配路网才能使用，这里就不使用 GPS 路径了。

取而代之，我们使用粒度较粗的 OD。LaDe 数据集提供了快递员的取件顺序，对于相邻的两次取件，设其分别位于 $u$, $v$，则 $OD[u][v] += 1$。因为取送件这个行为相对于路网来说比较稀疏，最终的 OD 矩阵需要在相邻路网节点之间进行平滑。

矩阵 OD 即我们可以利用的快递员行为信息。后面将考虑如何使用 OD 矩阵。

每个 **有货物** 节点的 feature 为 $X_i = [latitude, longitude]$
每个 **无货物** 节点的 feature 为 $Y_j = [latitude, longitude]$

首先经过若干层 GNN， $X'_i = GNN(\{X, Y\})$，这一步将 **无货物** 节点的信息通过 GNN 融入到 **有货物** 节点中。之前的研究中因为没有路网，所以没有这一步。不过这一步也可能没有用。

然后经过若干层的GIN（一种可以融入边的特征的 GNN），$X''_i, E_{u,v} = GIN(\{X'\},CostMatrix,OD)$，这一步将 **距离矩阵** 和 **快递员行为** 的信息融入到 **有货物** 节点中。并得到任意两点之间边的表征 $E_{u,v}$，最终拿 $E_{u,v}$ 过一个 MLP 的到边的启发值。

### 2、快递员行为作为预训练监督信号

这里将快递员的配送顺序作为预训练的监督信号，即如果 (u,v) 在快递员的配送顺序中相邻，那么 (u,v) 的预测值应该为 1，否则 (u,v) 的预测值为 0。**这里的 u 和 v 在路网上不一定相邻。**

GNN 层与上述相同。

GIN 层不加入 OD 输入，即 $X''_i, E_{u,v} = GIN(\{X'\}, CostMatrix)$。$E_{u,v}$ 过一个 MLP 得到快递员最终有没有先配送 u 再配送 v。

如果该快递员实际走过从 u 到 v 的配送，则 E_{u,v} 应该预测为 1，否则 E{u,v} 预测为0，使用交叉熵损失。可能要做一些正负样本平衡。

预训练完后，再换一个 MLP ，并变为预测该边的启发值。（其实快递员的行为本身也算一种启发值，但是和 TSP 问题最优解作为启发值还是有差距）。

## Discussions & Potential risk

### 距离矩阵的完备性，正收益与负收益

对于原始 TSP 问题来说，距离矩阵已经完备的表述了这个问题，理论上只依赖距离矩阵可以求出最优解。但是神经网络不一定能准确捕获距离矩阵的 Pattern，所以通过加入快递员的行为来辅助神经网络来捕获 Pattern，但是这也同时加入了干扰。这种做法是正收益还是负收益，需要看实验结果。

### OD 矩阵和 CostMatrix 同质性

OD 矩阵和 CostMatrix 本质都是二维矩阵，其 Pattern 捕获难度应该相当。而且 CostMatrix 是真正需要遵循的矩阵，OD 矩阵不是本身应该遵循的矩阵。

### 启发值准确性对算法效果的影响

是不是启发值预测越准，LKH 的效果或速度就越好。启发值相对 Pointer-Network 有一个弊端，那就是启发值只考虑当前所在的点，不考虑已经走过的点，而 Pointer-Network 考虑已经走过的点。如果已经走过的点不相同，那在当前点预测的启发值多准都没用。