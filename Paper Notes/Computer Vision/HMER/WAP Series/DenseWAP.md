# HMER
## WAP series

### Encoder

#### DenseEncoder
> The main idea of DenseNet is to use the concatenation of output feature maps of preceding layers as the input of succeeding layers.

let $H_l(\cdot)$ denote the convolution of many conv layers, then the output layer $l$ is presented as:$$
{\bf{x}}_l=H_l([{\bf{x}}_0;{\bf{x}}_1;\cdots,{\bf{x}}_{l-1}])
$$

> More detail could be found in the DenseNet part

### Decoder
> employ GRU as the decoder because it is an improved version of simple RNN which can alleviate the vanishing and exploding gradient problems

$$
{\bf{h}}_t= \rm{GRU}({\bf{x}}_t, {\bf{h}}_{t-1})
$$

Te output of CNN encoder is a 3D array of size $H\times W\times C$, and we can get a array of $C$-dim annotation with length $L=H\times W$, $$
{\bf{A}} = \{a_1,\cdots, a_L\}, a_i \in \mathbb{R}^C
$$
The GRU decoder is employed to generate a corresponding $\LaTeX$ string. 

The context vector $c_t$ is computed via weighted summing the variable-length annotations $a_i$: $$
c_t = \sum\limits_{i=1}^{L} \alpha_{ti}a_i
$$
The probability of each predicted symbol is computed by the context vector $c_t$, current decoder state $s_t$ and previous target symbol $y_{t-1}$ using the following equation: $$
p(y_t|y_{t-1}, X)=g(W_oh(Ey_{t-1})+W_ss_t + W_cc_t)
$$ where $X$ denotes input mathematical expression images. $g$ denotes a softmax activation function.

### Multi-Scale Attention with Dense Encoder
* Multi-Scale Dense Encoder: To implement the multi-scale attention model, we first extend the single-scale dense encoder into multi-scale dense encoder. Dense encoder has another multi-scale branch that produces high-resolution annotations $B$. The multi-scale branch is extended before the last pooling layer of the main branch has a higher resolution. The high-resolution annotation is a 3D array of size $2H\times 2W\times C'$, which can be represented as variable-length grid of $4L$ elements. $$
B = \{b_1,\cdots, b_{4L}\}, b_i \in \mathbb{R}^{C'}
$$

### Multi-Scale Attention Model:
> adopt **two unidirectional** GRU layers to calculate the decoder state $s_t$ and the multi-scale context vector $c_t$ that are both used as input to calculate the probability of predicted symbol. Employ two different single-scale coverage based attention model to generate the low-resolution context vector and high-resolution context vector by attending to low resolution annotations and high-resolution annotations respectively.

$$
\begin{aligned}
&\hat{s_t}={\rm{GRU}}(y_{t-1},s_{t-1}) \\ &{\bf{cA_t}} =f_{catt}({\bf{A}, \hat{{\bf{s}}}}_t) \\ &{\bf{cB_t}}=f_{catt}({\bf{B}, \hat{s}}_t) \\ &{\bf{c}}_t = [{\bf{cA}}_t;{\bf{cB}}_t] \\ &{\bf{s}}_t = {\rm{GRU}}({\bf{c}}_t, \hat{{\bf{s}}}_t)
\end{aligned}
$$ ${\bf{s}}_{t-1}$ represents the previous decoder state, $\hat{\bf{s}}_t$ is the prediction of current decoder state, ${\bf{cA}}_t$ is the low-resolution context vector at decoding step $t$, similarly ${\bf{cB}}_t$ is the high-resolution context vector. The multi-scale context vector ${\bf{c}}_t$ is the concatenation and it performs as the input during the computation of current decoder state ${\bf{s}}_t$