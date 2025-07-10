# IDGI: A Framework to Eliminate Explanation Noise from Integrated Gradients (CVPR 2023)

---

## 1. 核心思想

**IDGI（Integrated Directional Gradients for Interpretation）** 旨在减少集成梯度（Integrated Gradients, IG）方法中的解释噪声。  
其核心区别在于：**用归一化梯度方向替换原始的插值方向**，即在归因积分时，不再按输入的插值方向（$d = x_j - x_{j-1}$）累加梯度，而是用当前点的归一化梯度方向作为积分方向。

---

## 2. 算法流程

### 2.1 原始 IG 算法

Integrated Gradients 的基本归因计算方式为：
积分梯度定义：


$$
\text{IntegratedGradients}_i(x) = (x_i - x'_i) \times \int _{\alpha=0}^{1} \frac{\partial F(x' + \alpha \times (x - x'))}{\partial x_i} \, d\alpha
$$

数值近似：

$$
\text{IntegratedGradients} _ i(x)=  \sum_{k=1}^{m} \frac{\partial F(x' + \frac{k}{m} \times (x - x'))}{\partial x_i} \times \frac{1}{m}
$$ 

---
---

### 2.2 IDGI 变化

**IDGI 的改进点：**  
- **将插值方向 $d_j$ 替换为当前点的归一化梯度方向**

具体地，IDGI 的归因积分变为：

$$
\text{IDGI}_ i(x) = \sum_{j=1}^{m} g_j \cdot \tilde{d}_j
$$

或者

$$
\text{IDGI}_ i(x) = \sum_{j=1}^{m}  \frac{g_j g_j}{\| g_j\| \|g_j \|} \times  f(x_{j+1} )-f(x_j )
$$

其中，
- $g_j$：在第 $j$ 个插值点的梯度
- $\tilde{g}_j = \frac{g_j}{\|g_j\|}$：归一化后的梯度方向
- $\frac{f(x_{j+1} )-f(x_j )}{\|g_j\|} $: 步长的大小



---

## 3. 公式对比

| 方法    | 方向 $d_j$                  | 归因积分公式                                      |
|---------|----------------------------|--------------------------------------------------|
| IG      | $x_{j} - x_{j-1}$          | $\text{IG}_ i(x) = \sum_{j=1}^{m} g_j \cdot d_j$  |
| IDGI    | $\frac{g_j}{\|g_j\|}$      | $\text{IDGI}_ i(x) = \sum_{j=1}^{m} g_j \cdot \tilde{d}_j$ |

---

## 4. 优势与意义

- **消除噪声：** 通过自适应地选择归因方向，减弱了因插值方向带来的噪声，提高了解释的稳定性和一致性。
- **更自然的归因路径：** 归因方向始终与梯度方向一致，更符合模型本身的决策流。

---

## 5. 总结

**IDGI** 提供了一种更鲁棒的归因分析方法，通过动态调整归因方向，有效消除了集成梯度中的噪声，使显著性图更加真实、可解释。

---










# HDR-GAN: HDR Image Reconstruction From Multi-Exposed LDR Images With Large Motions

---

## 1. 简介

**HDR-GAN** 论文提出了一种基于生成对抗网络（GAN）的高动态范围（HDR）图像重建方法，能够从多张不同曝光的低动态范围（LDR）图像中恢复高质量 HDR 图像，特别是在存在大幅度运动（large motions）的情况下，表现优异。该方法旨在解决传统 HDR 重建中由于运动导致的鬼影（ghosting）等伪影问题。

---

## 2. 原理与方法

### 2.1 网络结构

- **生成器（Generator）**：采用 U-Net 结构，结合多尺度特征融合与注意力机制。它对输入的多张不同曝光的 LDR 图像进行特征提取、对齐、融合，生成最终的 HDR 图像。
- **判别器（Discriminator）**：用于判别生成的 HDR 图像与真实 HDR 图像的差异，提升生成图像的真实性。
  

### 2.2 关键技术

- **特征对齐模块**：通过注意力机制和可变形卷积，有效对齐不同曝光下的运动区域，减少鬼影。
- **多尺度融合**：在不同尺度上融合特征，增强网络对运动和细节的鲁棒性。
- **GAN 框架**：利用对抗学习提升 HDR 重建的视觉质量。

---

## 可微分色调映射损失（Differentiable Tonemapping Loss）

在 HDR-GAN 论文中，**损失的计算方式**是将生成的 HDR 图像通过一个**可微分的色调映射器（differentiable tonemapper）**进行处理，然后与目标图像进行比较。这种色调映射器常采用 **μ-law（μ-律）**，它在音频处理中常用于动态范围压缩，同样适用于图像的动态范围压缩。
本质： 通过此函数进行映射，再计算L1损失。更符合人眼的直觉

---

### μ-law 色调映射公式

μ-law 色调映射的数学表达式如下：

$$
T(x) = \frac{\log(1 + \mu \cdot x)}{\log(1 + \mu)}
$$

其中：

- $T(x)$：色调映射后的输出值
- $x$：输入的 HDR 像素值，通常归一化到 $[0, 1]$ 区间
- $\mu$：控制压缩程度的参数，通常 $\mu = 5000$ 或其他较大值

---

### 损失函数示例

通过 μ-law 色调映射后的损失函数可以写为：

$$
L_{\text{tonemap}} = \frac{1}{N} \sum_{i=1}^{N} \left| T\left(I_{HDR}^{gen}(i)\right) - T\left(I_{HDR}^{gt}(i)\right) \right|
$$

其中：

- $I_{HDR}^{gen}$：生成的 HDR 图像
- $I_{HDR}^{gt}$：真实的 HDR 图像
- $T(\cdot)$：μ-law 可微分色调映射函数
- $N$：像素总数

---
## PatchGAN判别器与对抗损失的高维超球距离计算

在 HDR-GAN 中，判别器 $D$ 采用 PatchGAN 结构（包含五层卷积），并引入了一种基于高维超球距离的对抗损失，以提升训练的稳定性并防止模式崩溃（mode collapse）。

---

### 1. PatchGAN 判别器

PatchGAN 判别器的输出可以被重塑为一个 $n$ 维向量：

$$
q = D(\cdot) \in R^n
$$

---

### 2. 逆立体投影到超球面

通过逆立体投影（inverse stereographic projection），将 $q$ 投影到单位超球面 $S^n$ 上的点 $p$：

$$
p = \left( \frac{2q}{\|q\|_2^2 + 1}, \frac{\|q\|_2^2 - 1}{\|q\|_2^2 + 1} \right)
$$

- 其中 $\|q\|_2$ 表示 $q$ 的 $L_2$ 范数。

---

### 3. 超球面上的距离度量

在超球面 $S^n$ 上，两个投影点 $p$ 和 $p'$ 的距离定义为：

$$
d_s(p, p') = \arccos \left( \frac{ \|q\|_2^2 \|q'\|_2^2 - \|q\|_2^2 - \|q'\|_2^2 + 4 q q' + 1 }{ (\|q\|_2^2 + 1)(\|q'\|_2^2 + 1) } \right)
$$

- 其中 $q'$ 是另一张图片的判别器输出，$q q'$ 表示内积。

---

### 4. 对抗损失（Adversarial Loss）

在该 GAN 框架下，判别器的优化目标是**最小化生成图像与真实图像在超球面映射特征上的距离**，参考点为超球面北极 $N = [0, ..., 0, 1]^T \in \mathbb{R}^n$：

$$
L_{adv} = \min_G \max_D \; E_{I_{real}} \left[ d_s(p_{real}, N) \right] - E_{I_{gen}} \left[ d_s(p_{gen}, N) \right]
$$

- $p_{real}$：真实图像通过判别器及逆立体投影后的点
- $p_{gen}$：生成图像通过判别器及逆立体投影后的点
- $N$：超球面北极


# Full-Gradient Representation for Neural Network Visualization (NeurIPS 2019)

---

## 1. 论文核心思想

该论文指出，**现有的可视化方法**（如梯度、积分梯度等）在计算显著性图时**忽略了偏置项（bias）对模型输出的影响**。实际上，偏置项在深度神经网络的决策中同样起着重要作用。为此，作者提出了一种**Full-Gradient Representation（全梯度表示）**，将**输入梯度和所有层的偏置项影响**一起整合到显著性图中，从而获得更全面、准确的可解释性。

---

## 2. 显著性图的两部分组成

### 2.1 输入梯度

第一部分是**模型输出对输入的梯度**，即：

$$
S_{input} = x \odot \frac{\partial f(x)}{\partial x}
$$

- $x$：输入
- $f(x)$：模型输出
- $\odot$：逐元素乘法

---

### 2.2 偏置项的全梯度

第二部分是**所有层中偏置项的梯度影响**。对于每一层 $l$ 和每个通道（卷积核） $c$，计算该层偏置项 $b_{l,c}$ 对输出的梯度：

$$
S_{bias}^{(l, c)} = b_{l,c} \cdot \frac{\partial f(x)}{\partial b_{l,c}}
$$

将所有层、所有通道的 $S_{bias}^{(l, c)}$ 通过上采样（例如反卷积或插值）**映射回输入空间大小**，得到 $S_{bias}^{(l, c)\uparrow}$。最后，将所有层、所有通道的偏置显著性图进行累加：

$$
S_{bias} = \sum_{l} \sum_{c} S_{bias}^{(l, c)\uparrow}
$$

---

## 3. Full-Gradient 显著性图

最终的**全梯度显著性图**由输入梯度显著性和偏置项显著性两部分相加：

$$
S_{full-grad} = S_{input} + S_{bias}
$$

---

## 4. 主要贡献与优势

- **全面性**：显著性图考虑了输入和所有偏置项的影响，提供更完整的解释。
- **适用性强**：适用于各类前馈神经网络，包括卷积神经网络（CNN）。
- **通用性**：可与现有的显著性可视化方法结合，提升解释能力。

---

## 5. 参考公式总结

- 输入梯度部分：
  $$
  S_{input} = x \odot \frac{\partial f(x)}{\partial x}
  $$
- 偏置项部分：
  $$
  S_{bias} = \sum_{l} \sum_{c} b_{l,c} \cdot \frac{\partial f(x)}{\partial b_{l,c}} \uparrow
  $$
- 全梯度显著性图：
  $$
  S_{full-grad} = S_{input} + S_{bias}
  $$

其中 $\uparrow$ 表示将偏置项的显著性分数上采样（映射）到输入空间大小。

---

## 偏置项显著性映射的梯度回流（反卷积）

在 Full-Gradient Representation 中，**偏置项的显著性分数需要映射到与输入同样的空间**。这可以通过**梯度回流（反卷积）**的方式实现：即计算每一层偏置项 $b_{l, c}$ 对输入 $x$ 的梯度。

---

### 公式表示

对于第 $l$ 层第 $c$ 个通道的偏置项 $b_{l,c}$，其对输入 $x$ 的梯度为：

$$
M_{bias}^{(l,c)}(x) = b_{l,c} \cdot \frac{\partial f(x)}{\partial b_{l,c}} \cdot \frac{\partial b_{l,c}}{\partial x}
$$

但由于 $b_{l,c}$ 仅在其对应层有效，实际映射到输入空间时，关注的是其对输入的影响，因此通常这样表达：

$$
M_{bias}^{(l,c)}(x) = b_{l,c} \cdot \frac{\partial f(x)}{\partial x} \Bigg|_{b_{l,c}}
$$

更常见的实现是，将 $b_{l,c}$ 的梯度通过反向传播（反卷积）回传到输入空间，记为：

$$
M_{bias}^{(l,c)}(x) = b_{l,c} \cdot \frac{\partial f(x)}{\partial x} \Bigg|_{\text{via } b_{l,c}}
$$

将所有层、所有通道的结果累加，得到最终的偏置项显著性图：

$$
S_{bias}(x) = \sum_{l} \sum_{c} M_{bias}^{(l,c)}(x)
$$

---

### 解释

- $M_{bias}^{(l,c)}(x)$ ：第 $l$ 层第 $c$ 个偏置项通过梯度回流（反卷积）映射到输入空间的显著性分数。
- $b_{l,c}$ ：第 $l$ 层第 $c$ 个通道的偏置项。
- $\frac{\partial f(x)}{\partial x} \Big|_{b_{l,c}}$ ：模型输出对输入的梯度，限定于 $b_{l,c}$ 的通路。

---

**说明**：  
这种通过梯度回流（反卷积）将偏置项的影响映射到输入空间的方法，使得**每个输入位置都能反映偏置项对最终输出的贡献**，实现了显著性图的空间一致性和可解释性。

---



