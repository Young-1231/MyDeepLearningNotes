# Parts
## Attention
### Attention model with coverage vector

#### Introduction
* Lacking  coverage might result in the following problems in conventional NMT:
    - Over-translation: some words are unnecessarily translated for multiple times
    - Under-translation: some words are mistakenly translated untranslated

> Propose a coverage mechanism to the intermediate representations of NMT models. Those coverage vectors, when entering into attention model, can help adjust the future attention and significantly improve the alignment between the source and target sentences. 

#### Background

##### Attention-based NMT
![](image/2022-10-03-10-49-07.png)
Given an input sentence $x=\{x_1,\cdots,x_{T_x}\}$ and previous generated words $\{y_1,\cdots,y_{i-1}\}$,the probability of next $y_i$ is $$
P(y_i|y_1,\cdots,y_{i-1},x) = g(y_{i-1},t_i,s_i)
$$
where $t_i$ is a decoding state for time step $i$, computed by $$
t_i = f(t_{i-1}, y_{i-1}, s_i)
$$
$s_i$ is a distinct source representation for time $i$, calculated as a weighted sum of the source annotations $h$:
$$
s_i = \sum\limits_{j=1}^{T_x} \alpha_{i,j}\cdot h_j
$$

##### Coverage Model for NMT
a coverage set is maintained to keep track of which source words have been translated("covered") in the past. Take an input sentence $x=\{x_1,x_2,x_3,x_4\}$  as an example, the initial 