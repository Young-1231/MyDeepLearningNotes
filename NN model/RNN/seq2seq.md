# NN model
## RNN
### seq2seq

#### Encoder
encoder将长度可变的输入序列转换为形状固定的context vector.
考虑由一个序列组成的样本(batch_size=1).输入序列为$x_1,\cdots,x_T$.其中$x_t$是输入文本序列中的第$t$个token.在时间步$t$,RNN将token $x_t$的输入向量$\bf{x_t}$和$\bf{h_{t-1}}$(即上一时间步的隐状态)转化为$\bf{h_t}$(即当前步的隐状态).$$
{\bf{h_t}}=f(\bf{x_t}, \bf{h_{t-1}})
$$
encoder通过选定的函数$q$,将所有时间步的隐状态转换为context vector$$
{\bf{c}} = q (\bf{h_1}, \cdots, \bf{h_T})
$$
比如, 当选择$q({\bf{h}}_1,\cdots,{\bf{h}}_T)={\bf{f}}_T$, context vector 仅仅是输入序列在最后时刻的隐状态${\bf{h}}_T$

#### Decoder
encoder输出的context vector ${\bf{c}}$对整个输入序列$x_1,\dots,x_{T}$进行编码。来自train dataset的输出序列$y_1,y_2,\cdots,y_{T'}$,对于每个时间步$t'$(与输入序列或编码器的时间步$t$不同), decoder输出$y_{t'}$的概率取决于先前的输出子序列$y_1,\cdots,y_{t'-1}$和context vector${\bf{c}}$, 即$P(y_{t'}|y_1,\cdots,y_{t'-1},{\bf{c}})$