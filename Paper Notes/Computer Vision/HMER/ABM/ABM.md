# HMER
## ABM

---
>>> 本论文主要创新点为: 1.在decoder部分，引入了mutual learning 实现了一个bi-directional mutual learning 模块2. 提出一个新的attention aggregation 模块


### 特征提取
此部分与DWAP基本一致，均是采用了DenseNet作为encoder,最后将得到的高层特征转化为一个语义向量$$
\textbf{a} = \{a_1, a_2, \cdots, a_M\}
$$, 其中，$a_i \in \mathbb{R}^D. M = H\times W$

### Attention Aggregation Module
同样是采用了带有coverage vector的attention。
值得注意的是，本文中所使用的多尺度机制是受到inception启发(采用并行多分支网络，采用kernel size不同的卷积), 具有更大的感受野，更加能够利用好全局的空间信息。数学描述为下: $$
A_s = U_s\beta_t, A_l = U_l\beta_l
$$(此式中$\beta_t$即为过去时间步的attention score之和，初始化为0 $\beta_t = \sum\limits_{l=1}^{t-1}\alpha_l$)
当前时间步$t$下attention score即由下式给出$$
\alpha_t = v_a^T\tanh(W_{\hat{h}}\hat{h}_t+U_fF+W_sA_s+W_lA_l)
$$ 而context vector也是采用与DWAP相同的计算方法

### Bi-directional Mutual Learning Module
> 此部分为本文核心

利用双向decoder来实现图片到$\LaTeX$序列的转化, L2R和R2L，互相学习得到的解码信息。双向decoder的结构是一样的，唯一区别就是在解码方向上。
以下是双向decoder的一些训练细节：
每一时间步$t$的预测由下式给出: 
* L2R: $p(\overrightarrow{y}_t|\overrightarrow{y}_{t-1})=W_o\max(W_yE\overrightarrow{y}_{t-1}+W_h h_t +W_t  c_t)$
* R2L: $p(\overleftarrow{y_t}|\overleftarrow{y}_{t-1})=W_o' \max(W_y'E'\overleftarrow{y}_{t-1}+W_h'h_t'+W_t'c_t')$
Mutual Learning的实现: 去掉L2R和R2L branch输出中开始标志$<eos>和结束标志<sos>$, 再将R2L branch的输出反向, 