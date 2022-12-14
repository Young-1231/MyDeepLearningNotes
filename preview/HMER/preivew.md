# Preview
## HMER


### Introduction
数学表达式是描述包括数学，物理在内的许多领域所不可缺少的。同时，随着移动设备的普及，人们已经开始使用手写数学公式作为一种自然的输入模式。同时，随着手写笔，平板电脑和智能手机等移动设备的普及，人们也逐渐开始使用手写的数学符号作为输入。手写数学表达式识别具有广泛的应用场景，如智能教育，人机交互，作业评分，论文写作的辅助。尽管目前的OCR系统取得了巨大的成功，但由于公式的复杂结构和多样化的个人书写习惯，HMER仍然是一个非常具有挑战性的问题。


### Datasets

#### CROHME
CROHME数据集是手写数学公式识别中使用最广泛的公共数据集，产生于一个在线的手写数学公式识别的比赛(CROHME).CROHME数据集中训练集由以下三部分组成：CROHME 2014(986), CROHME 2016(1147), CROHME 2019(1199)。括号中为数学公式的数量。CROHME数据集中识别的symbol class数量为111,包括"eos"和"sos".

#### HME100K 
HME100K是一个真实场景下的手写数学公式的数据集，其中训练集数量为74502, 测试集数量为24607.可识别的symbol class数量为249(此处存在问题，cite(CAN)论文中数量为249.而笔者从公开的HME100K版本获取到的symbol class数量为247). 
HME100K具有的意义是提供了真实场景,会具有复杂多变的背景（例如，颜色，并存在模糊的情况）。
HME100K的挑战与真实场景下文本识别(STN)
* Complex background
* Multiple colors, irregular fonts, different writing, stylesdifferent sizes, and diverse orientations
* Distorted by nonuniform illumination, low resolution, and motion blurring

CAN论文中针对HME100K使用的数据增强手段为TAl公司内部具有商用意义，作者并未公开。

### Evaluation Metrics

#### ExpRate(Expression recognition rate) 
表达式级别上的准确率

#### WER
WER是cite{WAP}中提出的一个word 级别上的评价指标, 然而在HMER领域后续的工作并未将WER作为评价工作的重要指标
* 将生成的LaTeX序列中的错误分为三类:   
    - substitution
    - deletion
    - insertion
* 最后得到$$
WER = \frac{N_{sub}^W+N_{del}^W+N_{ins}^W}{N^W} = \frac{N_{sub}^W+N_{del}^W+N_{ins}^W}{N_{sub}^W+N_{del}^W+N_{cor}^W}
$$其中
  - $N_{sub}^W$: the number of substitutions 
  - $N_{del}^W$: the number of deletions 
  - $N_{ins}^W$: the number of insertions 
  - $N_{cor}^W$: the number of corrects 
  - $N^W$: the number of words in the target


### Important Work
简要介绍以下工作


#### WAP
HMER中引入encoder-decoder机制的经典之作之一。WAP具体为Watch, Attend, Parse。主要思想借鉴Image Caption领域中经典之作\cite{SAT}, Show, Attend, Tell. 实际上两个网络结构也比较相似。

##### Watch
使用深度FCN作为encoder, 接受可变分辨率的公式图片作为输入，输出一个固定分辨率的高层的语义级别的feature map。将得到的大小为$H\times W\times C$的feature map空间维度展平，得到一个长度$L=H\times W$的序列,其中每个
分量均为$C$维，并且每个分量即为与原始图片某个区域对应的高层表征。
$$
A=\{a_1,\cdots,a_L\}, a_i\in \mathbb{R}^C
$$

##### Attend
带有attention的encoder-decoder存在一个问题\cite{coverage attention} **lack of coverage**。(coverage可理解为在decoder在每一时间步进行解码时，需要知道过去每一时间步内注意力所关注过的局部区域综合起来的整体信息)。否则就会出现以下问题
* over-parsing: 某一个局部区域重复被注意力所关注
* under-parsing: 某一个局部区域一直没能被注意力所关注

WAP针对上述问题使用的解决方案是在普通的additive attention中引入一个coverage vector，来持续记录过去注意力的对齐信息，具体使用了对过去所有时间步的注意力概率的和（与只是用上一时间步的注意力概率相比更加全局）。带有coverage的additive attention的数学描述如下：
$$
\begin{aligned}
& \beta_t = \sum\limits_{l}^{t-1}\alpha_l \\ & F = Q \ast \beta_t \\ 
& e_{ti} = v_a^T\tanh(W_ah_{t-1}+U_aa_i +U_ff_i)
\end{aligned} 
$$
其中，$\beta_t$为过去注意力概率的求和. 
$f_i$即是与annotation$a_i$对应的coverage vector。注意到coverage vector 是通过一层卷积运算后得到的（之所以使用卷积层，作者给出的解释是$a_i$对应的coverage vector应该和空间上与之相邻的注意力概率相关）
而t时刻attention weight即由下式给出
$$
\alpha_{ti}=\frac{\exp(e_{ti})}{\sum_{k=1}^{L}\exp(e_{tk})}
$$ 

##### Parse
采用带有Attend所述的带有coverage attention机制的GRU
t时刻的context vector $c_t$ $$
\mathbf{c}_t=\sum\limits_{i=1}^{L} \alpha_{ti}\mathbf{a}_i
$$
简要介绍GRU中隐状态$\mathbf{h}_t$计算过程
$$
\begin{aligned}
&\mathbf{z}_t = \sigma(\mathbf{W}_{yz}\mathbf{E}\mathbf{y}_{t-1} + \mathbf{U}_{hz}\mathbf{h}_{t-1}+\mathbf{C}_{cz}\mathbf{c}_t) \\ 
&\mathbf{r}_t = \sigma(\mathbf{W}_{yr}\mathbf{E}\mathbf{y}_{t-1} + \mathbf{U}_{hr}\mathbf{h}_{t-1}+\mathbf{C}_{cr}\mathbf{c}_t)
\\ 
&\tilde{\mathbf{h}}_t = \tanh(\mathbf{W}_{yh}\mathbf{E}\mathbf{y}_{t-1} + \mathbf{U}_{rh}(\mathbf{r}_t\otimes\mathbf{h}_{t-1})+\mathbf{C}_{cz}\mathbf{c}_t) \\
& \mathbf{h}_t = (1-\mathbf{z}_t)\otimes \mathbf{h}_{t-1} + \mathbf{z}_t \otimes \tilde{\mathbf{h}}_{t}
\end{aligned} 
$$

最终，输出word的概率分布即由下式给出$$
p(\mathbf{y}_t|\mathbf{x},\mathbf{y}_{t-1})=g(\mathbf{W}_o(\mathbf{E}\mathbf{y}_{t-1}+\mathbf{W}_h\mathbf{h}_t+\mathbf{W}_c\mathbf{c}_t))
$$其中,$g$为softmax函数,$\mathbf{W}_o\in \mathbb{R}^{K\times m}, \mathbf{W}_h\in \mathbb{R}^{m\times n}, \mathbf{W}_c\in \mathbb{R}^{m\times D}$,$\mathbf{E}$为embedding matrix


#### DWAP
Zhang等人在WAP的基础上提出DWAP， 主要改进如下：

##### DenseEncoder 
采用DenseNet\cite(DenseNet)作为encoder 
DenseNet主要思想为将先前多个层输出的feature map在channel维度上拼接后作为当前层的输入。
令$H_l(\cdot)$表示许多卷积层组合成的卷积块，那么第$l$个卷积块的输出即可表示为
$$
{\bf{x}}_l=H_l([{\bf{x}}_0;{\bf{x}}_1;\cdots,{\bf{x}}_{l-1}])
$$


##### Multi-Scale Attention with Dense Encoder
历史方法中存在以下问题:
* CNN中的池化操作会减小feature map的分辨率。手写的数学符号的尺寸差别较大。所以提取到的feature map的精细细节对HMER很重要。然而低分辨率的特征图则会损失掉这些细节。

针对此问题，作者在encoder中使用二分支，同时提供高分辨率和低分辨率的feature map。低分辨率feature map具有更大的感受野，提供了更加全局的语义信息，高分辨率feature map则具有更加精细的细节。

主分支产生低分辨率的feature map($H\times W\times C$)。在最后一个池化层之前引出另一个分支，输出一个高分辨率的feature map($2H\times 2W\times C'$)。
同WAP中方法，将feature map沿空间维度展平。得到低分辨率表征$A$, 高分辨率表征$B$
$$
\begin{aligned}
&A = \{\mathbf{a}_1,\cdots,\mathbf{a}_L\} \quad \mathbf{a}_i \in \mathbb{R}^C \\
&B = \{\mathbf{b}_1,\cdots,\mathbf{b}_{4L}\} \quad \mathbf{b}_i \in \mathbb{R}^{C'}
\end{aligned}
$$其中，$L=H\times W$

采用两个单一尺度的coverage attention model（同WAP） 来分别生成低分辨率和高分辨率下的context vector，并将这两个不同分辨率的context vector拼接起来作为多尺度context vector。
$$
\begin{aligned}
&\hat{s_t}={\rm{GRU}}(y_{t-1},s_{t-1}) \\ &{\bf{cA_t}} =f_{catt}({\bf{A}, \hat{{\bf{s}}}}_t) \\ &{\bf{cB_t}}=f_{catt}({\bf{B}, \hat{s}}_t) \\ &{\bf{c}}_t = [{\bf{cA}}_t;{\bf{cB}}_t] \\ &{\bf{s}}_t = {\rm{GRU}}({\bf{c}}_t, \hat{{\bf{s}}}_t)
\end{aligned}
$$
${\bf{s}}_{t-1}$表示上一时间步的解码器状态, $\hat{\bf{s}}_t$是当前解码器状态的预测值, ${\bf{cA}}_t$ 是t时刻的低分辨率的context vector$t$, 同理${\bf{cB}}_t$是高分辨率的context vector

#### ABM

#### CAN
CAN针对的是大部分现有的方法采用的都是encoder-decoder架构，但是encoder-decoder方法并不能保证取得的准确率，尤其是当公式的结构比较复杂或者序列长度较长时。
针对encoder-decoder方法所存在的问题，本文作者引入了symbol counting，并设计了一个弱监督的multi-scale counting module, 可以非常灵活的嵌入到其它网络结构中去。
作者认为引入symbol counting主要有以下两个好处:
* symbol counting和HMER两个任务实际上是互补的
* counting的输出结果同时也可以作为一个额外的全局信息输入，可以提高识别的准确率。
实际上，CAN即是向DWAP中嵌入了MSCM。
本文所设计的counting module实际上是一个弱监督，并不需要额外的标注信息。



##### Multi-scale Counting Module
输入为2D特征图$\mathcal{F}\in \mathbb{R}^{H\times W\times 684}$, 首先经过一个$1\times 1$卷积来改变通道数，得到**transformed feature**$\mathcal{T}\in \mathbb{R}^{H\times W\times 512}$, 并加上固定的位置编码$\mathcal{P}\in \mathbb{R}^{H\times W\times 512}$。
在decoding过程中，给定时间步$t$, 下式为attention weight($\alpha_t\in \mathbb{R}^{H\times W}$)的计算过程
$$
e_t = w^T\tanh(\mathcal{T}+\mathcal{P}+W_a\mathcal{A}+W_hh_t)+b\\
\alpha_{t,ij}=\exp(e_{t,ij})/ \sum\limits_{p=1}^{H}\sum\limits_{q=1}^W e_{t,pq}
$$
将得到的attention weight $\alpha_t$ 和 feature map$\mathcal{F}$做spatial-wise product得到context vector $\mathcal{C}\in\mathbb{R}^{1\times 256}$.在之前的HMER方法中，$y_t$仅仅是由context vector$\mathcal{C}$, hidden state$h_t$, embedding$E(y_{t-1})$。最终$p(y_t)$即由下式给出$$
p(y_t)={\text{softmax}}(w_o^T(W_c\mathcal{C}+W_v\mathcal{V}+W_th_t+W_eE)) + b_o\\
y_t \sim p(y_t)
$$

#### BTTR



### Conclusion
HMER作为Vision2Language，实际也是一个cross-modality任务，以下将从inner-modality和connection between vision and language
* LaTeX和自然语言的区别
    - LaTeX是mark up language 


### Experiment

#### DWAP


#### CAN
采用的是\cite{CAN}开源的代码，超参数等相关设置沿用


##### CROHME

