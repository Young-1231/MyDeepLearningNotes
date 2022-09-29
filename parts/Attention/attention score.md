# Parts
## Attention
---

### 背景
我们对query和key之间的关系建模.我们可以其中的一部分视为 注意力评分函数(attention scoring function)，然后把这个函数的输出结果输入到softmax函数中进行运算。通过上述步骤，我们将得到与key对应的value的概率分布(即注意力权重).最后，注意力汇聚的输出就是基于这些注意力权重的值的weighted sum.
用数学语言描述，假设有一个$q\in \mathbb{R}^q$和$m$个"key-value"对$(k_1,v_1),\cdots,(k_m,v_m)$,其中$k_i\in \mathbb{R}^k,v_i\in \mathbb{R}^v$。attention pooling函数$f$就被表示为成value的weighted sum$$
f(q, (k_1,v_1),\cdots,(k_m,v_m))=\sum\limits_{i=1}^{m}\alpha (q, k_i)v_i \in \mathbb{R}^v
$$
其中$q$和$k_i$的注意力权重(标量)是通过attention scoring function$a$将两个向量映射成标量，再经过softmax运算得到的
$$
\alpha(q, k_i)=softmax(a (q, k_i))= \frac{\exp{(a(q,k_i))}}{\sum\limits_{j=1}^{m}\exp{(a(q,k_j))}} \in\mathbb{R}
$$