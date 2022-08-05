# Parts
## Optimizer
---
### SGD with momentum

> 本笔记主要借鉴[Summer Clover知乎回答](https://www.zhihu.com/question/395685065/answer/2535950728)

> 总的来说，Momentum很有效，直觉上很容易理解，但是其理论性质复杂。
> 但是Momentum并不总能提升SGD的效果。尤其在深度学习里，优化器的性能会受到很多其它超参数的影响，比如batch_size和weight decay.

#### 直观理解
SGD是一个轻球(无惯性)做梯度下降寻找loss极小值，而Momentum则是一个重球(有惯性)做梯度下降寻找loss极小值。

#### 有依据的比较
在DL中，Momentum相比SGD最主要的好处是**经常有更好的收敛性**.