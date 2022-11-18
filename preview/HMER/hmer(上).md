# 简述手写数学公式识别(上)

## Introduction
数学表达式是描述包括数学，物理在内的许多领域所不可缺少的。同时，随着移动设备的普及，人们已经开始使用手写数学公式作为一种自然的输入模式。同时，随着手写笔，平板电脑和智能手机等移动设备的普及，人们也逐渐开始使用手写的数学符号作为输入。手写数学表达式识别(HandWritten MaHandwritten Mathematical Expression Recognition
, 下文简称为HMER)具有广泛的应用场景，如智能教育，人机交互，作业评分，论文写作的辅助。尽管目前的OCR系统取得了巨大的成功，但由于公式的复杂结构和多样化的个人书写习惯，HMER仍然是一个非常具有挑战性的问题。

---
## Datasets

### CROHME
CROHME数据集是手写数学公式识别中使用最广泛的公共数据集，产生于一个在线的手写数学公式识别的比赛(CROHME).CROHME数据集中训练集由以下三部分组成：CROHME 2014(986), CROHME 2016(1147), CROHME 2019(1199)。括号中为数学公式的数量。CROHME数据集中识别的symbol class数量为111,包括"eos"和"sos".

### HME100K 
HME100K是一个真实场景下的手写数学公式的数据集，其中训练集样本数量为74502, 测试集样本数量为24607.可识别的symbol class数量为249(此处存在小问题，CAN论文中所说symbol class数量为249.而笔者从公开版本的HME100K获取到的symbol class数量为247, 详情可见[issue19](https://github.com/LBH1024/CAN/issues/19)). 
HME100K作为真实场景下的数据集，具有包括颜色的多变、不同程度的模糊和复杂的背景等挑战。

关于CAN论文中针对HME100K数据集所使用的数据增强，以下为作者在[issue3](https://github.com/LBH1024/CAN/issues/3)中的回应
> Sorry the data augmentation code is for internal company use only and will not be released. In our experiments we mainly compare the results without using data augmentation since most previous methods didn't use it. Experiments with data augmentation are just to demonstrate that our method is still effective when data augmentation is used.

---
## Evaluation Metrics

### ExpRate(Expression recognition rate) 
表达式级别上的准确率

#### WER
WER是WAP中提出的一个word 级别上的评价指标, 然而在HMER领域后续的工作并未将WER作为评价工作的重要指标
* 将生成的LaTeX序列中的错误分为三类:   
    - substitution
    - deletion
    - insertion
* 最后得到$
WER = \frac{N_{sub}^W+N_{del}^W+N_{ins}^W}{N^W} = \frac{N_{sub}^W+N_{del}^W+N_{ins}^W}{N_{sub}^W+N_{del}^W+N_{cor}^W}
$其中
  - $N_{sub}^W$: the number of substitutions 
  - $N_{del}^W$: the number of deletions 
  - $N_{ins}^W$: the number of insertions 
  - $N_{cor}^W$: the number of corrects 
  - $N^W$: the number of words in the target


---
## Important Work
上部分介绍[WAP](http://home.ustc.edu.cn/~xysszjs/paper/PR2017.pdf)和[DWAP](https://arxiv.org/pdf/1801.03530.pdf)


### WAP
[WAP](http://home.ustc.edu.cn/~xysszjs/paper/PR2017.pdf)为HMER中引入encoder-decoder机制的经典之作之一。WAP具体为Watch, Attend, Parse。主要思想借鉴Image Caption领域中经典之作[Show, Attend&Tell](https://arxiv.org/pdf/1502.03044.pdf). 实际上两个网络结构也比较相似。
![](image/2022-11-18-00-24-46.png)
#### Watch
使用深度FCN作为encoder, 接受可变分辨率的公式图片作为输入，输出一个固定分辨率的高层的语义级别的feature map。将得到的大小为$H\times W\times C$的feature map空间维度展平，得到一个长度$L=H\times W$的序列,其中每个
分量均为$C$维，并且每个分量即为与原始图片某个区域对应的高层表征。
$
A=\{\mathbf{a}_1, \cdots, \mathbf{a}_L\}, \mathbf{a}_i\in \mathbb{R}^C
$

#### Attend
带有attention的encoder-decoder存在一个问题**lack of coverage**。(coverage可理解为在decoder在每一时间步进行解码时，需要知道过去每一时间步内注意力所关注过的局部区域综合起来的整体信息)。否则就会出现以下问题
* over-parsing: 某一个局部区域重复被注意力所关注
* under-parsing: 某一个局部区域一直没能被注意力所关注

WAP针对上述问题使用的解决方案是在普通的additive attention中引入一个coverage vector(更多细节可见[Modeling Coverage···](https://arxiv.org/pdf/1601.04811.pdf))，来持续记录过去注意力的对齐信息，具体使用了对过去所有时间步的注意力概率的和（与只是用上一时间步的注意力概率相比更加全局）。而带有coverage vector的注意力评分函数即为下式
$
\begin{aligned}
& \beta_t = \sum\limits_{l}^{t-1}\alpha_l \\ & F = Q \ast \beta_t \\ 
& e_{ti} = v_a^T\tanh(W_ah_{t-1}+U_aa_i +U_ff_i)
\end{aligned} 
$

其中，$\beta_t$为过去注意力概率的求和. 
$f_i$即是与annotation$a_i$对应的coverage vector。注意到coverage vector 是通过一层卷积运算后得到的（之所以使用卷积层，作者给出的解释是$a_i$对应的coverage vector应该和空间上与之相邻区域的注意力概率相关）
而t时刻的注意力权重(注意力概率)自然为$\text{Softmax}(e_{t})$, 具体可写成如下形式：
$
\alpha_{ti}=\frac{\exp(e_{ti})}{\sum_{k=1}^{L}\exp(e_{tk})}
$ 

#### Parse
采用带有Attend部分中所述的带有coverage attention机制的GRU作为parser
解码阶段在t时刻的context vector $c_t$由下式给出 
$
\mathbf{c}_t=\sum\limits_{i=1}^{L} \alpha_{ti}\mathbf{a}_i
$
以下为GRU中隐状态$\mathbf{h}_t$计算过程
$
\begin{aligned}
&\mathbf{z}_t = \sigma(\mathbf{W}_{yz}\mathbf{E}\mathbf{y}_{t-1} + \mathbf{U}_{hz}\mathbf{h}_{t-1}+\mathbf{C}_{cz}\mathbf{c}_t) \\ 
&\mathbf{r}_t = \sigma(\mathbf{W}_{yr}\mathbf{E}\mathbf{y}_{t-1} + \mathbf{U}_{hr}\mathbf{h}_{t-1}+\mathbf{C}_{cr}\mathbf{c}_t)
\\ 
&\tilde{\mathbf{h}}_t = \tanh(\mathbf{W}_{yh}\mathbf{E}\mathbf{y}_{t-1} + \mathbf{U}_{rh}(\mathbf{r}_t\otimes\mathbf{h}_{t-1})+\mathbf{C}_{cz}\mathbf{c}_t) \\
& \mathbf{h}_t = (1-\mathbf{z}_t)\otimes \mathbf{h}_{t-1} + \mathbf{z}_t \otimes \tilde{\mathbf{h}}_{t}
\end{aligned} 
$

最终，输出word的概率分布的计算过程如下
$
p(\mathbf{y}_t|\mathbf{x},\mathbf{y}_{t-1})=g(\mathbf{W}_o(\mathbf{E}\mathbf{y}_{t-1}+\mathbf{W}_h\mathbf{h}_t+\mathbf{W}_c\mathbf{c}_t))
$

其中,$g$为softmax函数,$\mathbf{W}_o, \mathbf{W}_h, \mathbf{W}_c$为linear projection matrix,$\mathbf{E}$为embedding matrix

---
### DWAP

Zhang等人在WAP的基础上提出[DWAP](https://arxiv.org/pdf/1801.03530.pdf)， 主要改进如下：

#### DenseEncoder 
![](image/2022-11-18-00-24-08.png)
采用DenseNet作为encoder 
DenseNet主要思想为将先前多个层输出的feature map在channel维度上拼接后作为当前层的输入。
令$H_l(\cdot)$表示第$i$个卷积层的卷积操作，那么第$i$层的输出即可表示为
$
{\bf{x}}_l=H_l([{\bf{x}}_0;{\bf{x}}_1;\cdots,{\bf{x}}_{l-1}])
$


#### Multi-Scale Attention with Dense Encoder
历史方法中存在以下问题:
* CNN中的池化操作会减小feature map的分辨率。手写的数学符号的尺寸差别较大。所以提取到的feature map的精细细节对HMER很重要。然而低分辨率的特征图则会损失掉这些细节。

针对此问题，作者在encoder中使用二分支，同时提供高分辨率和低分辨率的feature map。低分辨率feature map具有更大的感受野，提供了更加全局的语义信息，高分辨率feature map则具有更加精细的细节。

主分支产生低分辨率的feature map($H\times W\times C$)。在最后一个池化层之前引出另一个分支，输出一个高分辨率的feature map($2H\times 2W\times C'$)。
同WAP中方法，将feature map沿空间维度展平。得到低分辨率表征$A$, 高分辨率表征$B$

$
\begin{aligned}
&A = \{\mathbf{a}_1,\cdots,\mathbf{a}_L\} \quad \mathbf{a}_i \in \mathbb{R}^C \\
&B = \{\mathbf{b}_1,\cdots,\mathbf{b}_{4L}\} \quad \mathbf{b}_i \in \mathbb{R}^{C'}
\end{aligned}
$

其中，$L=H\times W$
采用两个单一尺度的coverage attention model（单尺度下的coverage attention model与WAP相同） 来分别生成低分辨率和高分辨率下的context vector，并将这两个不同分辨率的context vector拼接起来作为多尺度context vector。
decode计算过程如下（GRU相关计算过程与WAP中基本相同，故此处略去）

$
\begin{aligned}
&\hat{s_t}={\rm{GRU}}(y_{t-1},s_{t-1}) \\ &{\bf{cA_t}} =f_{catt}({\bf{A}, \hat{{\bf{s}}}}_t) \\ &{\bf{cB_t}}=f_{catt}({\bf{B}, \hat{s}}_t) \\ &{\bf{c}}_t = [{\bf{cA}}_t;{\bf{cB}}_t] \\ &{\bf{s}}_t = {\rm{GRU}}({\bf{c}}_t, \hat{{\bf{s}}}_t)
\end{aligned}
$

${\bf{s}}_{t-1}$表示上一时间步的解码器状态, $\hat{\bf{s}}_t$是当前解码器状态的预测值, ${\bf{cA}}_t$ 是t时刻得到的低分辨率context vector, 与之类似的，${\bf{cB}}_t$是高分辨率context vector
