# Parts
## Optimizer
### SGD(with Momentum) vs Adam
---

#### 参考资料
> [Summer Clover知乎回答](https://www.zhihu.com/question/42115548/answer/1636798770)

#### 
SGD(with Momentum)依然常常是实践效果更好的那个方法。
在理论和实践上，Adam家族里用了自适应学习率的优化器都不善于寻找flat minima.而flat minima对于generalizaion是很重要的。所以Adam训练得到的training loss可能会更低，但test performance常常却更差。这是很多任务里避免自适应学习率的最重要的原因。
同时，我们对SGD的理论算是比较了解，而Adam代表的自适应有话题是一种很heuristic,理论机制也很不清楚的方法。

#### 为什么SGD和Adam会各有所长呢？
在CV中用Adam之类的自适应优化器，得到的结果很有可能会离SGD的baseline差好几个点。主要原因是，**自适应优化器容易找到sharp minima,泛化表现常常比SGD显著地差**.(处在flat minima处的梯度期望比较低，导致学习率会增大，参数更新的时候很容易跳出该区域。)

而训练Transformer一类的模型，Adam优化得更快其好。主要原因是，**NLP任务的loss landscape有很多“悬崖峭壁”，自适应学习率更能处理这种极端情况，避免梯度爆炸。**
也有一些例外。虽然GAN一般是视觉任务，但是Adam还是成为了最流行的优化器。主要原因还是在于GAN的训练是不太稳定的，它的loss landscape和正常的视觉任务很不同。我们对于训练GAN的追求是能稳定就足够好了，flat minima对GAN的意义还不是很明确。

#### 关于Adam的两个误解
**1. 使用Adam不需要调节初始学习率**
尽管Adam默认的学习率0.001被广泛使用，但是在Adam比SGD表现好的领域，恰好都是重新调节的。比如训练GAN是大家一般使用学习率0.0002,而不是0.001.而训练Transformer会需要比0.001更大的初始学习率，默认设置是学习率0.2+NOAM Scheduler.调节学习率会结果影响很大，可以说是优化器最重要的超参数。

**2. Adam不需要learning rate decay**

