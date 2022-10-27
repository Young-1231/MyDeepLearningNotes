# HMER
## WAP Series

### 

### 评估指标
* ExpRate


* WER
    * word level上的评价指标
    * 将得到的word sequences中的错误分为三类:   
        - substitution
        - deletion
        - insertion
    * 最后得到 WER = $$
WER = \frac{N_{sub}^W+N_{del}^W+N_{ins}^W}{N^W} = \frac{N_{sub}^W+N_{del}^W+N_{ins}^W}{N_{sub}^W+N_{del}^W+N_{cor}^W}
$$其中, $N_{sub}^W$: the number of substitutions $N_{del}^W$: the number of deletions $N_{ins}^W$: the number of insertions $N_{cor}^W$: the number of corrects $N^W$: the number of words in the target

### Architecture

#### Watch
使用FCN来接收variable size的input image
同时利用FCN来保持起feature map与input image中local region的关系(obtain the level of correspondence)
使用FCN同时也有利于Parser有选择地对与图片的给定区域赋予相应的attention weight
FCN encoder的输入是一个三维矩阵`shape` $H\times W\times D$.将这个三维矩阵沿着高宽维度上展平，得到$L=H\times W$个elements，其中每个element都是$D$维，最后得到和图片中local region相关的表征$$
a=\{\bf{a_1},\cdots,b_L\}, a_i\in \mathbb{R}^D
$$

#### Attend
对于每一个由parser生成的word,整体图片未必都在提供有用的信息。从此意义上说，对于给定时间步$t$,原始图片中的一部分不应参与到当前$c_t$的计算。所以parser需要利用attention mechanism来知道对于生成下一时间步的word,model需要'focus'图片的哪一部分(即对于正确的annotation$\bf{a_i}$施加更大的attention weight)
##### Additive Attention
对于以上需求，我们利用additive attention来对此种关系进行建模。将上一时间步的hidden state$h_{t-1}$作为query,将$\bf{a}_i$作为key,attention scoring function即可描述为$$
e_{ti}={\bf{v}_a^\top}\tanh(W_ah_{t-1}+U_a{\bf{a_i}})
$$
而attention weight即由下式给出$$
\alpha_{ti}=\frac{\exp(e_{ti})}{\sum_{k=1}^{L}\exp(e_{tk})}
$$ 用$n'$表示attention dimension;$v_a\in \mathbb{R}^{n'}
$,$W_a\in \mathbb{R}^{n'\times n}$, $U_a\in \mathbb{R}^{n'\times D}$.从而,即可给出当前时间步的context vector $c_t$ $$
\mathbf{c}_t=\sum\limits_{i=1}^{L} \alpha_{ti}\mathbf{a}_i
$$
最终，输出word的概率分布即由下式给出$$
p(\mathbf{y}_t|\mathbf{x},\mathbf{y}_{t-1})=g(\mathbf{W}_o(\mathbf{E}\mathbf{y}_{t-1}+\mathbf{W}_h\mathbf{h}_t+\mathbf{W}_c\mathbf{c_t}))
$$其中,$g$为softmax function作用于vocabulary中的所有word,$\mathbf{W}_o\in \mathbb{R}^{K\times m}, \mathbf{W}_h\in \mathbb{R}^{m\times n}, \mathbf{W}_c\in \mathbb{R}^{m\times D}$,$\mathbf{E}$是随机初始化的可学习的参数
#### Parse 

##### 计算context vector
为解决需从variable-size input images生成同样是variable-length output的LaTeX sequence.所以我们需要得到一个fixed size的中间vector, 借由seq2seq中的理念，我们需要得到一个context vecto来连接encoder和decoder
模型接受ME image, 输出$y$为LaTeX序列，可表示为$$y=\{y_1,\cdots,y_C\}, y_i\in\mathbb{R}^K$$,其中$K$为vocab_size, $C$为生成的sequence的length
contect vector$c_t$由encoder输出的表征的weighted sum产生。
而我们利用GRU对此种关系进行建模（得到预测word的概率分布），此种关系用数学语言可描述为$$
p(y_t|y_1,\cdots,y_{t-1},x)=f(y_{t_1}, h_t, c_t)
$$其中，function$f$将在后面给出，$x$为input ME image
> GRU的过程省略，见GRU章节
> embedding matrix $E\in \mathbb{R}^{m\times K}$