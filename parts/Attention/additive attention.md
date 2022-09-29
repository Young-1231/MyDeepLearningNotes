# Parts
## Attention

### Additive Attention
一般来说，当query和key是不同长度的vector时，我们可以使用Additive Attention作为attention scoring function.给定$q\in \mathbb{R}^q$和$k\in \mathbb{R}^k$,additive attention scoring function 为:$$
a(q, k)=w_v^\top \tanh{(W_q q+W_kk)}\in\mathbb{R}
$$
其中learnable parameters是$W_q \in \mathbb{R}^{h\times q},W_k \in \mathbb{R}^{h\times k}$和$w_v\in \mathbb{R}^h$.将query和key连结起来后输入到一个MLP中，包含一个隐藏层，num of layers是一个hyperparameter $h$.通过$\tanh$作为activation function,并且禁用偏置项。