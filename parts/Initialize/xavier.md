# Parts 
## Initialization

---
### Xavier Initialize

#### 基本思想

* **方差一致性**
    * 保持激活值的方差一致或梯度的方差一致，这样有利于优化。基于该基本思想，作者假设每层的输入位于激活函数的线性区域，且具有零点对称的分布
    * **保持激活值的方差一致**
        - 首先，对于每个卷积层，其相应结果为: $$
Y_l = W_l X_l + b_l \tag{1}
$$
        - 其中, $X_l$是每层的输入参数，其shape为$(k\times k\times c)$; $W_l$是该层的卷积核参数，其shape为$(d\times k\times k\times c)$; $b_l$是偏置，通常初始化为$0$ 从公式(1)中，可以得知， $X_l=f(Y_{l-1})$和$c_t=d_{l-1}$, 其中$f(\cdot)$表示激活函数
        - 假设每层的$W_l$是相互独立且同分布；每层的$X_l$是相互独立且同分布; $W_l$和$X_l$是相互独立的。因此，令$b_l=0$, 可以得到: $$
Var[y_l]=n_lVar[w_l x_l] \tag{2}
$$其中, $y_l,w_l$和$x_l$表示$Y_l, W_l$和$X_l$的每个随机变量，$n_l =k^2c$表示神经元个数。 令$w_l$具有零均值，且关于零点对称分布；$x_l$也具有零均值， 且位于激活函数的线性区域, $x_l =f(y_{l-1})=y_{y-1}$, 所以: $$
Var[y_l] = n_lVar[w_lx_l] \\ = n_lVar[w_l]Var[x_l]  \\ = n_lVar[w_l]Var[y_{l-1}] \tag{3}
$$
        - 因此，对于$L$层的输出，有$$
Var[y_L] = Var[y_1]\left(\prod_{l=2}^{L}n_l Var[w_l]\right) \tag{4}
$$ 
        - 所以，公式(4)是初始化设计的关键，如果每层的$n_lVar[w_l]$过大或者过小，都会导致最后输出的值会指数型的增加或减少。因此，对于所有层，都需要满足: $$
n_lVar[w_l]=1 \tag{5}
$$
        - 所以，每层权重的初始化服从零均值，方差为$\frac{1}{n_l}$的分布
    * 保持梯度的方差一致
        - 对于反向梯度传播而言，每个卷积层的梯度为 $$
\nabla X_l = \hat{W}_l\nabla Y_l \tag{6}
$$其中, $\nabla X_l$和$\nabla Y_l$分别表示梯度； $\nabla Y_l$的shape为$(k\times k \times d)$; $\hat{W}$的shape为$(c\times k\times k\times d)$, 这里的$\hat{W}$和$W$可以变换形状进行转化。同样，这里假设$w_l$和$\nabla y_l$相互独立，$\nabla x_l$具有零均值，$w_l$关于零点对称分布。在公式(6)中, 可以得到: $$
\nabla y_l = f'(y_1)\nabla x_{l+1} \tag{7}
$$
由于假设位于线性激活区域, $f'(y_l)=1$， 所以: $$
\begin{aligned}
Var[\nabla x_l]&=\hat{n}_l Var[\hat{w}_l \nabla y_l] \\ &= \hat{n}_l Var[\hat{w}_l]Var[\nabla y_l] \\ &=\hat{n}_l Var[\hat{w}_l]Var[\nabla x_{l+1}]
\end{aligned}
$$其中, $\hat{n}_l=k^2 d$, 与上述的$n_l=k^2 c$不一样，但都表示神经元个数.将$L$层结果堆积起来，得到 $$
Var[\nabla x_2] = Var[\nabla x_{L+1}]\left(\prod_{l=2}^L \hat{n}_l Var[w_l]\right) \tag{9}
$$同样，为了使公式(8)不会过大或者过小(过大或者过小会导致梯度爆炸或者梯度弥散)，让每层梯度均满足: $$
\hat{n}_l Var[w_l]=1 \tag{10}
$$所以， 每层权重的初始化服从零均值，方差为$\frac{1}{\hat{n}_l}$的分布