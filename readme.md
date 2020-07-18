# GCN and GAT 
## GCN
instruction:
```shell
python example.py > rungcnlog.txt
```

Dataloader:采用老师提供的代码util.py

算法：采用GCN，使用了论文提供的代码（gcn和gcnlayer）

batch读取和优化：自己实现

我的结果是loss: 0.5910 AUC: 0.7584 Prec: 0.4088 Rec: 0.7359 F1: 0.5256

这个结果和论文中的结果比较接近（毕竟是论文中提供的算法代码）

TODO:尝试不同的超参数

## GAT
```shell
python examplegat.py > rungatlog.txt
```

目前的结果是（50个epoch）loss: 0.5114 AUC: 0.8205 Prec: 0.4847 Rec: 0.7300 F1: 0.5826
