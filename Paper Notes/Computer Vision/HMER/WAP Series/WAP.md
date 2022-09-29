# HMER
## WAP Series

### 

### 评估指标
* ExpRate


* WER
    * word level上的评价指标
    * 将得到的word sequences中的错误分为三类:   
        - substitution
        - deletion
        - insertion
    * 最后得到 WER = $$
WER = \frac{N_{sub}^W+N_{del}^W+N_{ins}^W}{N^W} = \frac{N_{sub}^W+N_{del}^W+N_{ins}^W}{N_{sub}^W+N_{del}^W+N_{cor}^W}
$$其中, $N_{sub}^W$: the number of substitutions $N_{del}^W$: the number of deletions $N_{ins}^W$: the number of insertions $N_{cor}^W$: the number of corrcts $N^W$: the number of words in the target
