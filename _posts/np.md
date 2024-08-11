---
layout: post
title: 神经过程
tags: mathjax
math: true
date: 2024-10-02 15:32 +0800
---
#论文/神经过程 #论文/PDE   
# Random Grid Neural Processes for Parametric Partial Differential Equations

![[Pasted image 20240810001554.png]]

<font color="#9bbb59">引用或注释</font>[来源] <font color="#f79646">暂疑</font> 摘抄或重述或翻译 <font color="#4bacc6">想法</font>
## 1 Introduction
<font color="#4bacc6">非常清爽凝练的introduction。</font>

**研究内容**：The central objective of this paper is to introduce accurate and uncertainty aware physics informed probabilistic deep learning methods that go beyond fixed grid approaches to parametric PDEs, allowing for a synthesis with noisy data measured on arbitrary grids for forward and inverse problems.

**总体方法**：we propose a new framework for jointly learning <span style="background:#fff88f">probabilistic mappings</span> of forward and inverse problems of parametric PDEs using ideas from <font color="#9bbb59">spatial statistics</font> 

<font color="#9bbb59">spatial statistics</font> [Cressie, N. and Moores, M. T. Spatial statistics. arXiv preprint arXiv:2105.07216, 2021] : connects neural Gaussian process models of PDE solution fields with physical parameters through conditioning on random domain partitions. 通过<font color="#f79646">condition on </font>【随机域分割】建立【神经高斯过程建模的PDE解空间】与【物理参数】的关联。<font color="#4bacc6">作者是否要建立参数到解空间的映射（用NP）</font>

**技术要点**：
1. a physics driven variational inference framework based on random grids;<font color="#f79646">如何体现“物理驱动”，推导思路</font>
2. new kernels for the learning of Gaussian random fields;解用<font color="#9bbb59">高斯随机场</font>建模。<font color="#f79646">什么样的核？</font>
<font color="#9bbb59">高斯随机场</font>：<font color="#f79646">平面或空间的随机过程？</font>
3. a new grid invariant architecture to enable learning through random collocation。<font color="#f79646">什么架构？体现random grid吗？</font>
后面看这三点如何实现。

**实验对象**：
1. 1D nonlinear Poisson PDE 
2. spatio-temporal Burgers equation
3. incompressible Navier-Stokes lid-driven cavity flow
后面看具体方程形式、从xx映射到xx。

【<font color="#f79646">?</font>】Furthermore, we demonstrate how to correctly incorporate sparse, noisy observations of sample solution fields to improve the predictive capabilities of the model.<font color="#4bacc6">乍一看很像添加噪声样本提升模型泛化能力。</font>

**对比方法**：PDDLVM，modified DeepONets，PhysicsInformed Parametric Fourier Feature Networks
<font color="#f79646">本文方法正问题、逆问题都能解决？</font>
看后面的实验部分

**相关工作**：
主要列举的是学一组参数化PDE的方法，后面列了一些学一个PDE的方法。  

1. 监督/半监督算子学习。
	-  思路：完全或部分基于由传统方法得到的没有噪声的PDE解数据集，或添加物理驱动损失。
	- 代表：FNO，Physics Informed Neural Operators and DeepONets
	- 缺点： 
		1. 依赖传统方法来生成数据集，解决新问题受限。<font color="#f79646">The limitations include the need to recompute models from scratch for new parameter instances and rely on CPU-driven operations that do not parallelize easily.</font>
		2. 从一个到另一个函数空间的映射中学习算子，固定grid，当“<font color="#f79646">domain geometry</font>”，初始条件、边界条件等难以在函数空间中定义时，学习能力受到限制。
	- 回应：关注“sets of scalar coefficients” 标量参数集。  
	
2. 物理+VAE。
	- 思路：将<span style="background:#fff88f">观测空间</span>与<font color="#f79646">discovered physical latent space</font>联系起来
	- 代表：
		1. [Zhong, W. and Meidani, H. PI-VAE: Physics-informed variational auto-encoder for stochastic differential equations. Computer Methods in Applied Mechanics and Engineering, 403:115664, 2023. ISSN 0045-7825.]
		2. [Takeishi, N. and Kalousis, A. Physics-integrated variational autoencoders for robust and interpretable generative modeling. Advances in Neural Information Processing Systems, 34:14809–14821, 2021.]
		3. [Tait, D. J. and Damoulas, T. Variational autoencoding of PDE inverse problems. arXiv preprint arXiv:2006.15641, 2020.]
		4. [Glyn-Davies, A., Duffin, C., Akyildiz,  ̈ O. D., and Girolami, M. Φ-DVAE: Learning Physically Interpretable Representations with Nonlinear Filtering. arXiv preprint arXiv:2209.15609, 2022.]
	- 本文特色：将<span style="background:#fff88f">解场</span>与物理隐空间联系起来，同时可能用观测空间的一些数据来增强这种映射。<font color="#f79646">“Furthermore, we are interested in methods that can perform inference in the absence of data and are only supplemented/improved by data.”</font>  
	
3. 物理+NP。现有工作【Yang, Y. and Perdikaris, P. Conditional deep surrogate models for stochastic, high-dimensional, and multi-fidelity systems. Computational Mechanics, 64(2):417–434, 2019b.】为纯数据驱动。  

4. 其他概率建模，学习单个解实例的方法。只进行了列举。

没有提到之前看过的<font color="#9bbb59">ProbConserv</font>[D. Hansen, D. C. Maddix, S. Alizadeh, G. Gupta, and M. W. Mahoney, “Learning Physical Models that Can Respect Conservation Laws,” presented at the ICLR2023, Jan. 2024, p. 133952]这篇文章。简单回顾一下![[Pasted image 20240810231603.png]]
![[Pasted image 20240810232222.png]]
其中$\sigma_G$ 是很关键的参数，相当于引入噪声，如果=0，完全等同于数据驱动了。

后面看作者对已有工作的不足做出了哪些回应。以及有没有我们可以做的部分。

## 2 背景

该章对正问题和逆问题进行定义。  

### 2.1 正问题
$$
\begin{align}\mathcal{G}^{\mathbf{w}}_{\mathbf{z}}(u)(x)=0, &\space x\in \Omega\\\mathcal{B}_{\mathbf{w}}(u)(x)=0, &\space x\in \partial\Omega\end{align}
$$
$\mathcal G$是微分算子，$\mathbf z$ 是逆问题要求解的参数，$\mathbf w$是其他参数，$x$是自变量，$u$是要求解的变量，$\mathcal B$是边界算子。

此外，限制边界条件为迪利克雷条件，用了以下手段来处理：$$u(x)=B(x)+D(x)N(x)$$B(x)是符合边界条件的任意函数，D在边界微分处=0，
扫了一眼他引用的两篇文章，没有直接出现这个公式，<font color="#f79646">没有解释D(x)和N(x),不知道什么意思</font>，后面也没出现。
[Sukumar, N. and Srivastava, A. Exact imposition of boundary conditions with distance functions in physicsinformed deep neural networks. Computer Methods in Applied Mechanics and Engineering, 389:114333, 2022]这篇文章乍眼一看非常新颖扎实且有用。![[Pasted image 20240811011447.png]]与这个公式似乎相关的段落如上。
### 逆问题

$$h_\beta:u(x),\mathbf w \rightarrow \mathbf z$$
可以使用优化或<font color="#9bbb59">概率公式</font>来完成。逆仿真器逼近问题依赖于一个经过训练的正仿真器，并且不依赖于经典的数值求解器。
<font color="#9bbb59">概率公式</font>[Vadeboncoeur, A., Akyildiz,  ̈ O. D., Kazlauskaite, I., Girolami, M., and Cirak, F. Deep probabilistic models for forward and inverse problems in parametric PDEs. arXiv preprint arXiv:2208.04856, 2022.]


