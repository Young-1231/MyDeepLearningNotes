# Image Caption

## MMT

### Introduction
heart
* modeling language: fully-attentive layer vs rnn 
* follow multi-level & extensive (need to see more details) fashion

new design of the fully-attentive layer(two key novelty)
* encode image regions and their relationships in a multi-level way ("can learn and encode a prior knowledge by using **persistent memory vectors**") #TODO: the computation of the memory vectors 
* learned gating mechanism when generating the sentence , also in a multi-level way

#### Contribution
> 1. encoding and decoding layers are connected in a mesh-like structure, weighted through a learnable gating mechanism(Q: gating mechanism is actually the **mesh-like** structure)
> 2. using persistent memory vectors as a way to introduce prior knowledge 
> 3. in some fashionable datasets

### Related Work
#TODO: more papers need to be read further


### Methodology
both encoder and decoder are made of stacks of fully-attentive layers

#### Encoder
> process regions from the input image and devising relationships between them

Q: how to devise relationships  between regions 

##### Memory-Augmented Encoder

$S(X)=\text{Attention}(W_qX, W_kX, W_vX)$

> attention weights depend solely on the **pair-wise similarities** between linear projections of the input set itself
> There are, the self-attention operator can be seen as a way of encoding pair-wise relationships



#### Decoder
> reads from the output of each encoding layer to generate output caption word by word

Q: how to compute the probability distribution over the vocab in the transformer


#### Connection