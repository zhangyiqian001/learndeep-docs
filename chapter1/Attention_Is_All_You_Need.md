# [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
# 摘要

我们提出了一种新的简单的网络架构，Transformer，完全基于注意力机制，完全取消RNN和CNN。在两个机器翻译任务上的实验表明，这些模型在质量上更优越，同时更并行，需要的训练时间明显更少。

## 模型架构

![image.png](https://tc-cdn.flowus.cn/oss/c89fada6-2875-4fcc-ae96-59645b89a028/image.png?time=1743045300&token=fb2d575fc6fbf74da887992906af2d9a24b29a2f775b571d40a982d01f1efde1&role=free)

变压器遵循这种整体架构，对编码器和解码器使用堆叠的self-attention和点积的全连接层，分别如图1的左半部分和右半部分所示

**编码器**：该编码器由N=6个相同的层组成。每个层都有两个子层。第一种是一个多头自注意机制，第二种是一个简单的、位置上完全连接的前馈网络。我们在两个子层周围使用残差连接，然后进行层归一化。也就是说，每个子层的输出是LayerNorm（x +子层(x)），其中子层(x)是由子层本身实现的函数。为了方便这些剩余的连接，模型中的所有子层以及嵌入层都会产生尺寸$$d_{model}$$= 512的输出。

**解码器**：解码器也由N = 6个相同的层组成。除了每个编码器层中的两个子层外，解码器还插入第三个子层，该子层对编码器堆栈的输出执行多头自注意机制。与编码器类似，我们在每个子层周围使用剩余连接，然后进行层归一化。我们还修改了解码器堆栈中的自注意子层，以防止位置关注后续的位置。这种掩蔽，加上输出嵌入被一个位置偏移，确保了对位置i的预测只能依赖于小于i的位置的已知输出

## 注意力模型

注意函数可以描述为将查询和一组键值对映射到输出，其中查询、键、值和输出都是向量。输出是作为值的加权和计算的，其中分配给每个值的权重是由查询与相应键的兼容性函数计算的

### Scaled Dot-Product

![image.png](https://tc-cdn.flowus.cn/oss/d211845b-1ced-43f9-966d-32fc09381542/image.png?time=1743045300&token=0dc5012b1d3bc8c2366e740f44814ffa0acce21f0420e0e9b8044435a8f3e5e2&role=free)

输入由$$d_k$$维度的查询(Q)和键(K)，以及$$d_v$$维度的值(V)组成。

本文$$d_{model}$$=512，$$h$$=8，故$$d_k$$=64

两种最常用的注意函数是：

- dot-product attention：除了$$\frac{1}{\sqrt{d_k}}$$，与本文的计算方式一致。**防止输入softmax值过大，导致偏导趋近0，除以根号d类似归一化**

- additive attention：使用单隐层的前馈网络

虽然这两种方法在理论复杂性上相似，但在实践中，点积注意力在实践中更快，更节省空间，因为它可以使用高度优化的矩阵乘法代码来实现。

![image.png](https://tc-cdn.flowus.cn/oss/4b6043d2-6a39-493f-a493-e1870a34aee2/image.png?time=1743045300&token=fc1097cea532228d44d55dcbaf721befd5bd62b125fe9846335f11c299211736&role=free)

### Multi-Head Attention

![image.png](https://tc-cdn.flowus.cn/oss/0a79bb22-f4d7-411d-87f4-60655515eb00/image.png?time=1743045300&token=312e9266e63021b03e68539a52adff1d7762e9711b81383b2af313792aacc0a4&role=free)

我们发现，将查询、键和值h次与不同的学习线性投影分别线性投影到$$d_k$$、$$d_k$$和$$d_v$$维是有益的。

多头注意允许模型共同关注来自不同位置的不同表示子空间的信息。用一个注意力头，平均可以抑制这一点。

在这项工作中，我们使用了h = 8 的注意层。对于每一个模型，我们都使用dk = dv = dmodel/h = 64。由于每个头部的维数降低，其总计算代价与全维的单头注意相似。

![image.png](https://tc-cdn.flowus.cn/oss/15c671d1-e388-4ef3-a31e-a33fd3a7e074/image.png?time=1743045300&token=2b8f31d73cbd569b8e3072fba30b588841d85b7ae1ebb8cfa6c407cb496b76fd&role=free)



该变压器以三种不同的方式使用多头注意力：

![image.png](https://tc-cdn.flowus.cn/oss/7b8c19e6-a600-4040-ac94-458e2ff9b475/image.png?time=1743045300&token=8142ddb8cfb0754ce413c77bff3078b24241b3e0e1e904335766b573ee130bcb&role=free)

在“编解码器注意”层中，Q来自解码器层，K、V来自编码器层

![image.png](https://tc-cdn.flowus.cn/oss/75efc3b3-76e4-495e-9901-cc20be4f4f15/image.png?time=1743045300&token=d8eca492a846cc65eafccf2299c53e1b762342c5d08ecdd32dfd23704b13ab31&role=free)

在“编码器“层中，所有的K、Q、V都来自同一个位置，在这种情况下，是编码器中上一层的输出。编码器中的每个位置都可以处理编码器上一层中的所有位置。

![image.png](https://tc-cdn.flowus.cn/oss/62915777-6256-4e49-bd7f-85ba1feb9633/image.png?time=1743045300&token=6bdcbcf734a1e3d48c405c6b78880c9d52e9d11fe3b1ff64df72bbc5879edcc4&role=free)

在“解码器“层中，解码器中的自注意层允许解码器中的每个位置关注解码器中的所有位置，直到并包括该位置。我们需要防止解码器中的信息向左流，以保持自回归特性。我们通过mask（设置为−∞）softmax输入中对应于非法连接的所有值来在缩放点积注意内部实现这一点

## Position-wise Feed-Forward Networks

除了注意子层外，我们的编码器和解码器中的每个层都包含一个完全连接的前馈网络，这由两个线性变换组成，中间有一个ReLU激活。

![image.png](https://tc-cdn.flowus.cn/oss/94367c7d-b8a2-4b49-b34f-6ed42f4ed03c/image.png?time=1743045300&token=ad53673186db600f19e69edba5c62491267f78d7a493a27f4d107b45c4a4a8c7&role=free)

## Embeddings and Softmax

与常见模型相同

## Positional Encoding

为了使模型利用序列顺序，我们必须注入一些关于序列中令牌的相对位置或绝对位置的信息。

![image.png](https://tc-cdn.flowus.cn/oss/0dfb9669-f8cd-46cf-a877-ed54aba23695/image.png?time=1743045300&token=e4fe5e5f3b75d2fece1d1042c287a0561c1a6f5ab841b5f6f8b0ddd65e816d27&role=free)

其中pos是位置，i是维度。也就是说，位置编码的每个维度都对应于一个正弦曲线。我们选择这个函数是因为我们假设它允许模型容易地学习相对位置，因为对于任何固定偏移k，$$PE_{pos+k}$$可以表示为$$PE_{pos}$$的线性函数

这种方法它可能允许模型推断比训练中遇到的更长序列。

# QA
# Reference

