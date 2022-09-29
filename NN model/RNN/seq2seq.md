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
