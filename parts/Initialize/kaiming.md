# Parts
## Initialization

### Kaiming Initialize

#### Background
Xavier初始化方法中，其假设激活函数是线性的。但随着ReLU成为常见的激活函数，其不是线性的，所以Xavier初始化会对ReLU系列的激活函数失效。因此，kaiming在基于激活函数是ReLU的基础上，推导出He初始化方法。

#### 保持激活值的方差一致
假设$w_l$具有零均值，$w_l$与$x_l$相互独立，所以$Var[w_l]=E[w_l^2]-E^2[w_l]=E[w_l^2]$.而激活函数是ReLU，有$x_l=\max(0,y_{l-1})$，所以$x_l$不具备零均值，且$E[x_l^2]=\frac{1}{2}Var[y_{l-1}]$,因此可得
$$
\begin{aligned}
Var[y_l]&=n_lVar[w_lx_l] \\ &=n_l(E[(w_lx_l)^2]-E^2[w_lx_l]) \\ &=n_l(E[w_l^2]E[x_l^2]-E^2[w_l]E^2[x_l]) \\ &=n_lE[w_l^2]E[x_l^2] \\ &=\frac{1}{2}n_lVar[w_l]Var[y_{l-1}]
\end{aligned}
$$

#### fan_in&fan_out
> fan(in dictionary): disperse or radiate from a central point to cover a wide area: 从一个中心点分散或辐射到一个大范围。
> 
而在DeepLearning中 **central point** 对应一个layer。
例如在linear层中, input layer$\in \mathbb{R}^{m}$, output_layer$\in \mathbb{R}^{n}$，这包含一个weight matrix$\in \mathbb{R}^{n\times m}$
把layer作为fan所对应的 **central point**
那么: 
* 该层的input就是**fan_in**
* 该层的output就是**fan_out**
在此情况下,fan_in = m, fan_out = n

##### PyTorch中计算fan_in和fan_out
* 方法一: 同时计算fan_in 和 fan_out
```python
m, n = 4, 6
linear = nn.Linear(m, n)
torch.nn.init._calculate_fan_in_and_fan_out(
        linear.weight)
```
* 方法二：单独计算fan_in和 fan_out
``` python
torch.nn.init._calculate_correct_fan(
                linear.weight,
                mode='fan_in')
```
### 原论文notes
[Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/pdf/1502.01852.pdf)

#### Abstract

> First: propose PReLU which generalizes the traditional rectifier unit.
> Second: derive a robust initialization method that considers the rectifier nonlinearity

#### Introduction

##### The conclusion of the development of CNN
* building more powerful models
* designing effective strategies against overfitting
    - nn are becoming more capable of fitting training data, because of increased complexity(like increased depth, enlarged width and the use of smaller strides, new nonlinear activation and sophisticated layer designs)
    - achieve better generalization, using effective regularization, aggressive data augmentation and large-scale data
* The property of ReLU
    - expedites convergence of the training procedure and leads to better solutions than conventional sigmoid like units.
    - ReLU is not a symmetric function. So, the mean response of ReLU is always $\geqslant 0$
    - even assuming the inputs/weights are subject to symmetric distributions, the distributions of response can still be asymmetric. 
    - the properties mentioned above influence the theoretical analysis of convergence and empirical performance

#### Approach 

##### PReLU

> Definition: $$
f(y_i) = \begin{cases}
y_i, &\text{if}\quad y_i > 0 \\ a_i y_i, &\text{if}\quad y_i \leqslant  0
\end{cases} \tag{1}
$$

$y_i$is the input of the nonlinear activation $f$ on the $i$th channel.$a_i$ is the coefficient controlling the scope of the negative part.when $a_i$ in $(1)$ is a learnable parameter, we refer $(1)$ as PReLU. $(1)$ is equivalent to $f(y_i)=\max(0,y_i) + a_i\min(0,y_i)$
PReLU introduces a very small number of extra parameters, whose number is equal to the total number of channels.Moreover, $a_i$ could be channel-shared, so that $(1)$ would be $f(y_i)=\max(0,y_i)+a\min(0,y_i)$.

> Optimization  PReLU can be trained using bp and optimized simultaneously with other layers. We can easily get the update formulations of $\{a_i\}$
 
The gradient of $a_i$ for one layer is  $$
\frac{\partial \varepsilon}{\partial a_i} = \sum\limits_{y_i}\frac{\partial \varepsilon}{\partial f(y_i)} \frac{\partial f(y_i)}{\partial a_i}
$$ $\varepsilon$ represents the objective function. The term$\frac{\partial \varepsilon}{\partial f(y_i)}$ is the gradient propagated from the deeper layer. The gradient of the activation is as flows$$
\frac{\partial f(y_i)}{\partial a_i} = \begin{cases}
0, &\text{if}\quad y_i > 0 \\ y_i, &\text{if}\quad y_i \leqslant 0
\end{cases}
$$


##### Initialization of Filter Weights for Rectifiers

##### TODOlist
- [ ] im2col