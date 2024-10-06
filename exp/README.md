# something important

## CandidateSet 生成方式

疑问0？alpha 生成 CandidateSet 的流程是啥来着？

看明白了如何生成之后，就可以
1、不需要提前运行 LKH 来生成候 CandidateSet 了。

### 对于不带时间窗口的对称问题 CVRP

疑问1？为什么需要这么处理？

疑问2？这么转化后最优解不会变差吗？

[选择 Special 节点](SRC/MTSP2TSP.c)
``` c
if (ProblemType != CTSP && Salesmen <= Dim && MTSPMinSize > 0 &&
    !AnyFixed) {
    HeapMake(Dim - 1);
    for (i = 1; i <= Dim; i++) {
        N = &NodeSet[i];
        if (N == Depot)
            continue;
        N->Rank = Distance(N, Depot);
        HeapLazyInsert(N);
    }
    Heapify();
    for (i = 1; i <= Salesmen; i++)
        HeapDeleteMin()->Special = i; // 选择离 Depot 最近的 Salesmen 个节点标记为 Special 节点。
    HeapClear();
    free(Heap);
    Heap = 0;
}
```

[连边](SRC/Forbidden.c)
``` c
if (Na->DepotId) {
    if ((Nb->DepotId && MTSPMinSize >= 1) || // Depot 之间不能连边
        (Nb->Special && // 是 Special 节点
            Nb->Special != Na->DepotId && // Special 节点只能与对应的 Depot 连边
            Nb->Special != Na->DepotId % Salesmen + 1))
        return 1;
}
```

### 对于带时间窗口的非对称问题 CVRPTW (WIP)



## TimeWindow 生成方式 (WIP)

稍为修改了 LKH 代码，需要重新编译。

## Discussions

### 最段路矩阵这种数据（dist[a, b] + dist[b, c] <= dist[a, c]) 会造成 alpha-measure 退化吗？