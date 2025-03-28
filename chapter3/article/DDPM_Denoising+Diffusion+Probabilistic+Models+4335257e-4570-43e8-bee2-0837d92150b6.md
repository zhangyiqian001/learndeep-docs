2006.11239

[https://github.com/hojonathanho/diffusion](https://github.com/hojonathanho/diffusion.)

# 摘要

我们使用扩散概率模型呈现了高质量的图像合成结果，这是一类潜在变量模型的想法来自非平衡热力学。

## 导言

各种深度生成模型最近在各种数据模式中显示出高质量的样本。生成对抗网络（GANs）、自回归模型、流和变分自编码器（VAEs）已经合成了引人注目的图像和音频样。

扩散概率模型（为了简洁起见，我们称之为“扩散模型”）是一个参数化的马尔可夫链，使用变分推理进行训练，在有限时间后产生与数据相匹配的样本。这条链的转换被学习为逆转扩散过程，扩散过程是一个马尔可夫链，它逐渐向相反采样方向的数据增加噪声，直到信号被破坏。当扩散包含少量的高斯噪声时，将采样链转换设置为条件高斯噪声就足够了，允许一个特别简单的神经网络参数化。

![image.png](DDPM：Denoising+Diffusion+Probabilistic+Models+4335257e-4570-43e8-bee2-0837d92150b6/image.png)

扩散模型定义起来很简单，训练效率也很高，但据我们所知，还没有证据表明它们能够生成高质量的样本。我们这篇论文证明，扩散模型实际上能够生成高质量的样本，有时比在其他类型的生成模型上发表的更好的结果更好。

## 背景

扩散模型是$p_θ(x_0) := \int p_θ(x_{0:T} ) dx_{1:T}$的潜在变量模型

其中，$x_1，...，x_T$是与数据$x_0∼q(x_0)$具有相同维度的隐变量

## Diffusion models and denoising autoencoders

扩散模型可能是一类有限的潜在变量模型，但它们在实现过程中允许大量的自由度。我们必须选择正向过程的方差$β_t$和反向过程的模型体系结构和高斯分布参数化。为了指导我们的选择，我们在扩散模型和去噪分数匹配之间建立了一个新的明确的联系（第3.2节），这导致了扩散模型的一个简化的、加权的变分界目标（第3.4节）。最终，我们的模型设计是由简单性和实证结果证明的（第4节）。我们的讨论按公式式分类

![image.png](DDPM：Denoising+Diffusion+Probabilistic+Models+4335257e-4570-43e8-bee2-0837d92150b6/image 1.png)

![image.png](DDPM：Denoising+Diffusion+Probabilistic+Models+4335257e-4570-43e8-bee2-0837d92150b6/image 2.png)

简化后的目标函数（模型使用的）：

![image.png](DDPM：Denoising+Diffusion+Probabilistic+Models+4335257e-4570-43e8-bee2-0837d92150b6/image 3.png)

其中$\overline{\alpha}_t$是给定好的一组序列，$\{\overline{\alpha}_0 ... \overline{\alpha}_T\}$，生成方式如下代码：

```Shell
betas = torch.Tensor(
    make_beta_schedule(schedule="linear", n_timestep=n_timestep)
)
# [0.9999, 0.9994, 0.9985, 0.9971, 0.9953, 0.9931, 0.9905, 0.9874, 0.9839,0.9800]
alphas = 1.0 - betas

def make_beta_schedule(
    schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3
):
    if schedule == "linear":
        betas = (
            torch.linspace(
                linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64
            )
            ** 2
        )

    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(
            linear_start, linear_end, n_timestep, dtype=torch.float64
        )
    elif schedule == "sqrt":
        betas = (
            torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
            ** 0.5
        )
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()
   
```

因为我们的简化目标（14）放弃了等式中的权重（12），它是一个加权变分界，与标准变分界[18,22]相比，它强调了重建的不同方面。特别是，我们在第4节中的扩散过程设置使简化目标简化为对应于小t的降低重量损失项。这些项训练网络用非常小的噪声去噪数据，因此降低其权重是有益的，这样网络就可以在更大的t项下专注于更困难的去噪任务。我们将在我们的实验中看到，这种重新加权会导致更好的样本质量。



## 注：

T必须>1000，希望$x_t$要服从正态分布。

