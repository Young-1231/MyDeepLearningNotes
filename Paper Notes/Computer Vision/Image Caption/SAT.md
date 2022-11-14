# Image Caption
## SAT

### Soft and Hard attention
#### stochastic hard attention
我们将位置变量$s_t$理解为模型在生成第$t$个word时所需要集中注意力的位置，那么$s_{t,i}$即是一个独热的指示变量，当第$i$个位置被用来提取特征时被设置为1.如果将注意力位置视为一个中间隐变量，我们可以引入一个以$\{\alpha_i\}$为参数的多重分布，并将$\hat{z_t}$作为随机变量$$
p(s_{t,i}=1\vert s_{j<t}, {\bf{a}})= \alpha_{t,i} \\
\hat{\bf{z}}_t = \sum\limits_{i}s_{t,i}{\bf{a}}_i
$$
据此，我们可以定义一个新的目标函数$L_s$
#TODO: 