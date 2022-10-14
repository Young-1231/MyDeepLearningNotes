# CNN
## SENet

> 参照[知乎文章](https://zhuanlan.zhihu.com/p/65459972)
### 主题思路
对于CNN网络来说，核心计算是卷积算子，其通过卷积核从输入feature map学习到新feature map。从本质上来讲，卷积是对一个局部区域进行特征融合，这包括空间上$H\times W$和channel$C$的特征融合
而对于卷积操作，很大一部分工作是提高感受野，即空间上融合更多特征融合，或者提取多尺度空间信息，而对于channel维度的特征融合，卷积操作基本上默认对输入特征图的所有channel进行融合. SENet的创新点子啊与关注channel之间的联系，希望模型可以自动学习到不同channel特征的weight.为此，SENet提出了Squeeze-and-Excitation(SE)模块。
SE模块首先对卷积得到的feature map进行Squeeze操作，得到channel级的全局特征，然后对全局特征进行Excitation操作，学习各个channel之间的关系，也得到不同channel的权重，最后乘以原来的特征图得到最终特征。