2010.02502v4

# 摘要

Denoising diffusion probabilistic models（DDPMs）在没有对抗性训练的情况下实现了高质量的图像生成，但它们需要多次模拟马尔可夫链才能生成样本。为了加速采样，我们提出了denoising diffusion implicit models（DDIMs），这是一种更有效的迭代隐式概率模型，具有与DDPMs相同的训练过程。在DDPMs中，生成过程被定义为一个特定的马尔可夫扩散过程的反向过程。我们通过一类非马尔可夫扩散过程来推广DDPMs，从而得到相同的训练目标。这些非马尔可夫过程可以对应于确定性的生成过程，从而产生能够快速产生高质量样本的隐式模型。我们的经验证明，与DDPMs相比，DDIMs可以快速产生10×到50×的高质量样本，使我们能够在样本质量上权衡计算，直接在潜在空间中执行有语义意义的图像插值，并以极低的误差重建观测结果。

## 导言

## BACKGROUND

来自数据分布的给定样本$q(x_0)$，学习模型分布$p_\theta(x_0)$

DDPMs是潜在变量模型：

![image.png](DDIM:DENOISING+DIFFUSION+IMPLICIT+MODELS+2dc2998f-bd42-4b37-a8ca-da08243c3e7c/image.png)

目标函数（maximizing a variational lower bound）： p的分布 - q的分布差最小，log最大

![image.png](DDIM:DENOISING+DIFFUSION+IMPLICIT+MODELS+2dc2998f-bd42-4b37-a8ca-da08243c3e7c/image 1.png)

前向加噪过程：

![image.png](DDIM:DENOISING+DIFFUSION+IMPLICIT+MODELS+2dc2998f-bd42-4b37-a8ca-da08243c3e7c/image 2.png)

前向加噪过程：（简化后的形式）

![image.png](DDIM:DENOISING+DIFFUSION+IMPLICIT+MODELS+2dc2998f-bd42-4b37-a8ca-da08243c3e7c/image 3.png)

$x_0$加噪后，去噪的输入$x_t$：

![image.png](DDIM:DENOISING+DIFFUSION+IMPLICIT+MODELS+2dc2998f-bd42-4b37-a8ca-da08243c3e7c/image 4.png)

目标函数带入4，简化结果：

![image.png](DDIM:DENOISING+DIFFUSION+IMPLICIT+MODELS+2dc2998f-bd42-4b37-a8ca-da08243c3e7c/image 5.png)

## VARIATIONAL INFERENCE FOR NON-MARKOVIAN FORWARD PROCESSES

### NON-MARKOVIAN FORWARD PROCESSES

让我们考虑一个由实向量索引的推理分布的族Q：

![image.png](DDIM:DENOISING+DIFFUSION+IMPLICIT+MODELS+2dc2998f-bd42-4b37-a8ca-da08243c3e7c/image 6.png)

经过变换后：

![image.png](DDIM:DENOISING+DIFFUSION+IMPLICIT+MODELS+2dc2998f-bd42-4b37-a8ca-da08243c3e7c/image 7.png)



### GENERATIVE PROCESS AND UNIFIED VARIATIONAL INFERENCE OBJECTIVE

这是给出$x_t$对$x_0$的预测，后向过程：（预测$x_0$）

![image.png](DDIM:DENOISING+DIFFUSION+IMPLICIT+MODELS+2dc2998f-bd42-4b37-a8ca-da08243c3e7c/image 8.png)

然后，我们就可以用一个固定的先验来定义生成过程$p_θ(x_T ) = N (0, I)$：（DDPM模型为）

![image.png](DDIM:DENOISING+DIFFUSION+IMPLICIT+MODELS+2dc2998f-bd42-4b37-a8ca-da08243c3e7c/image 9.png)

新目标函数：

![image.png](DDIM:DENOISING+DIFFUSION+IMPLICIT+MODELS+2dc2998f-bd42-4b37-a8ca-da08243c3e7c/image 10.png)

## SAMPLING FROM GENERALIZED GENERATIVE PROCESSES

以$L_1$为目标，我们不仅学考虑的马尔可夫推理过程的生成过程，还学习了我们描述的由σ参数化的许多非马尔可夫正向过程的生成过程。因此，我们基本上可以使用预先训练好的DDPM模型作为新目标的解决方案，并专注于寻找一个生成过程，通过改变σ来更好地根据我们的需求生成样本。

### 4.1 DENOISING DIFFUSION IMPLICIT MODELS

从等式中的$p_θ(x_{1:T})$（10），我们可以通过以下方式从一个样本$x_t$中生成一个样本$x_{t−1}$：

![image.png](DDIM:DENOISING+DIFFUSION+IMPLICIT+MODELS+2dc2998f-bd42-4b37-a8ca-da08243c3e7c/image 11.png)

其中$\epsilon_t∼N(0,I)$是独立于$x_t$的标准高斯噪声。当$σ_t = \sqrt{(1−α_{t−1})/(1−α_t)} \sqrt{1−α_t/α_{t−1}}$时，正向过程为马尔可夫，生成过程为DDPM。

**由于12式子不遵循马尔可夫过程，我们可以从$\{1,...,T\}$序列中采样出序列 L = $\{1,...,S\}$ L的长度远小于T。**

**我们注意到另一个特殊情况，当所有t的$σ_t = 0$；给定$x_{t−1}$和$x_0$，正向过程变得确定性，除了t = 1；在生成过程中，随机噪声$\epsilon_t$之前的系数为零。由此得到的模型成为一个隐式概率模型，其中样本是通过一个固定的程序（从$x_T$到$x_0$）从潜在变量生成的。我们将其命名为去噪扩散隐式模型（DDIM，发音/d：Im/），因为它是一个用DDPM目标训练的隐式概率模型（尽管正向过程不再是一个扩散）。**

### 4.2 ACCELERATED GENERATION PROCESSES

在前面几节中，生成过程被认为是反向过程的近似值；由于正向过程有T个步骤，生成过程也被迫对T个步骤进行采样。但是，由于只要$q_σ(x_t|x_0)$是固定的，去噪目标L1就不依赖于特定的正向过程，因此我们也可以考虑长度小于T的正向过程，这样无需训练不同的模型，即可加速相应的生成过程。

让我们考虑正向过程不是定义在所有潜在变量$x_{1:T}$上，而是在一个子集$\{x_{τ_1},…,x_{τ_S} \}$，其中τ是长度为$[1,…,T]$的递增子序列。特别是，我们定义了$x_{τ_1},…,x_{τ_S} $上的顺序正向过程使$q(x_{τ_i}|x_0)=N(\sqrt{α_{τ_i}}x_0,(1−α_{τ_i}) I)$ 匹配“边线”（见图2）。生成过程现在根据反向(τ)对潜在变量进行采样，我们称之为（采样）轨迹。当采样轨迹的长度远小于T时，由于采样过程的迭代特性，我们可以显著提高计算效率。

使用类似于第3节中的论点，我们可以证明使用使用L1目标训练的模型，因此在训练中不需要改变。我们展示了对等式中的更新只有轻微的变化（12）需要获得新的、更快的生成过程，这适用于DDPM、DDIM，以及等式中考虑的所有生成过程 (10).我们在附录C.1中包含了这些细节。

原则上，这意味着我们可以用任意数量的前进步骤来训练一个模型，但在生成过程中只从其中一些步骤中采样。因此，训练后的模型可以考虑比（Ho et al.，2020）或甚至连续时间变量t（Chen等人，2020）中所考虑的更多的步骤。我们将这方面的实证调查留为未来的工作。

![image.png](DDIM:DENOISING+DIFFUSION+IMPLICIT+MODELS+2dc2998f-bd42-4b37-a8ca-da08243c3e7c/image 12.png)



