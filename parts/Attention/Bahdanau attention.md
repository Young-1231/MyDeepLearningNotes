# Attention
## Bahdanau Attention

### Model
本章中遵循seq2seq中相同的符号表达。这个新的基于注意力的模型与seq2seq中的模型相同，只是context vector ${\bf{c}}$在任一decoder time step $t'$ 都会被 $c_{t'}$替换。Suppose the input sequence has $T$ tokens. decoder time step $t'$ 的 context vector 是attention pooling 的输出
$$ 
{\bf{c}}_{t'} = \sum\limits_{t=1}^{T}\alpha({\bf{s}}_{t'-1},{\bf{h}}_t) {\bf{h}}_t
$$
其中, 时间步$t'-1$的decoder hidden state${\bf{s}}_{t'-1}$是query, encoder隐状态${\bf{h}}_t$既是key, 也是value. attention weight $\alpha$是使用additive attention scoring function 来计算的。
