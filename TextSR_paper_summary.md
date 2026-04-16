# TextSR: Diffusion Super-Resolution with Multilingual OCR Guidance

本文介绍了论文 [TextSR (arXiv:2505.23119v1)](https://arxiv.org/html/2505.23119v1) 的核心理论梳理，包括模型结构、数据构造和训练方案，以及用于工程复现的注意点和详细的 TODO List。

---

## 1. 模型结构 (Model Structure)

TextSR 采用了一种基于跨注意力 (Cross-Attention) 融入多语言先验（UTF-8）的 **多模态扩散模型 (Multimodal Diffusion Model)** 用以针对场景文本级别图像超分辨率进行增强 (STISR)。

- **文本区域定位与提取**:
  整体框架不直接对包含大规模背景的全图做文字超分。而是使用现成的文本检测器提取文本边界框，通过区域切片（仿射变形裁剪）提取单一文本行。然后利用 OCR 提取多语言文本内容（UTF-8编码串）。
- **多语言文本表征 (Multilingual Text Representations)**:
  - 为了能够统一处理中、英、日、法等超过5种语言，模型直接采用 **UTF-8 字节格式**将字符转换为对应 Token（共 256 个 UTF-8 Token + PAD/EOS/UNK 等）。
  - **特征提取**: 使用了预训练的 **ByT5-Base** 文本编码器进行字形关联。为了避免模型学习到与翻译而非字形相关的高层语义特征，**只选用了编码器的前 2 层**，其输出特征表达为 $M \times d_\tau$ (序列长度 $\times$ 1536 维度的嵌入层)。
- **扩散模型与去噪生成 (DDPM U-Net)**:
  - 基本架构：以 32 通道为起点的 5 对残差编码解码 U-Net (降采样/上采样)，通道随尺度缩放为 `[1, 2, 4, 8, 8]`。
  - **条件注入 (Conditioning)**：图片条件 $c_I$（低分辨率图像）直接与各个 Timestep 的 Nosie （加噪空间）在通道维度拼接之后输入给 U-Net。文本特征条件 $c_T$ 通过**交叉注意力 (Cross-Attention)** 选择性注入，且仅注入到分辨率为 $3\times30\times256$ 和 $1\times10\times256$ 的特征图级。
  - **残差域生成**：为了避免由于大模型和外部先验引起的“生成幻觉”偏离原有文本样式，**扩散过程生成目标并非常规的高分辨率原图，而是 HR 与 LR 的残差差异图像特征。**

---

## 2. 数据构造 (Data Construction)

- **训练素材来源**: 综合使用了 7 个公开的大规模文本检测和转录 (TDT) 数据集：HierText, TextOCR, VinText, DenseText, ICDAR19 LSVT, ICDAR19 MLT, TotalText。
- **降质量模型 (Degradation Process)**: 纯净高清原图需要搭配低分辨率图片才能进行训练监督。在此没有使用简单的模糊降质，而是使用了 **Real-ESRGAN** Pipeline 高阶退化策略，将每张训练图像**重复复制20次并施加随机的复杂降质操作**。
- **尺度对齐与清洗**: 
  - 通过目标检测 Ground Truth 参数提取纯粹的独立文本行，过滤掉高度不在 16 像素 ~ 512 像素之间的异常数据。
  - 最终得到了 **1800万对 (18M Crops)** 小片文本训练数据。
  - 所有送入扩散模型的小片训练数据在长宽比限制下**全部被强制 Resize/Padding 成了 $48 \times 480$** 的统一图像尺寸进入网络流动。

---

## 3. 训练方案 (Training Scheme)

- **框架与超参**: 基于 JAX，于 TPUv5 上训练；Batch Size 设置为 1024，学习率为 3e-4。
- **无分类器引导与双重条件 (CFG with Dual Conditions)**:
  为了让模型不完全依靠精确识别的文字强行渲染，而是能学会“纯图片去噪”和“融合先验去噪”，在生成 Batch 过程中加入了 **条件随机丢弃 (Dropout of Txt Condition)** 机制。
  - 形成纯视觉条件网络：$\epsilon_\theta(x_t, t, c_I, \emptyset)$
  - 形成图文联合条件网络：$\epsilon_\theta(x_t, t, c_I, c_T)$
- **推理中的自适应控制策略 (Inference Techniques)**:
  采用 5 步 (5-steps) 的 DDIM 采样。
  1. **分类器引导平衡控制**: 引入参数 $\omega$ 调节权重。根据实际输入源OCR的可靠程度改变：公式等效为 `pred_noise = img_only_noise + omega * (img_text_noise - img_only_noise)`。高质量 OCR / Ground-Truth 使用高 $\omega$ ($\omega=3.0$)，当使用不精准的外部检测模型引导时设置 $\omega < 1.0$ 保证不过度干扰。
  2. **迭代 OCR 调节机制 (Iterative OCR Conditioning)**: 不完全依赖原始模糊图片 $c_I$ 使用 OCR。官方强烈建议使用 $R=1$ 配置：**先让没有文本先验辅助的 $\epsilon_\theta (c_I, \emptyset)$ 预跑一遍超分出较好的临时图，对临时图跑 OCR，用识别出的精准内容作为 $c_T$，最终再由大模型图文模型全量生图。**

---

## 4. 复现注意点 (Implementation Notes)

> [!WARNING]
> **冻结 ByT5 编码器**: 文本的字符特征抽取必须直接调用预训练 ByT5-Base 模型并且**保持冻结 (Frozen)** 不参与反向传播，仅截取其 1~2 层使用。避免破坏现成的多语言字符字形空间。

> [!CAUTION]
> **防止生成幻觉**: 很多传统 MLLM-based SR 模型（如 SUPIR）倾向把相似的线段修复成不相关的字符甚至不存在的噪点花纹。扩散阶段**必须预测原图与模糊图之间的差值 (Residual Images)** 作为 Ground Truth 处理。

> [!TIP]
> **局部推理到全图的映射融合 (Blending)**: TextSR 本质只是修复图像里的**字区域**，并不重构原图背景。在全图推理中：
> 1. 先用标准的 Real-ESRGAN 将原背景图 $I$ 进行放大超分。
> 2. 对每个 $48 \times 480$ 且已经放大的文本 Patch，用逆仿射变换 (`cv2.invertAffineTransform`) 匹配回对应区域坐标。
> 3. 由于边缘光照或环境可能撕裂，论文提出将 TextSR 的结果通过采用 Sigma 为 3.0 的高斯**低通滤波器 (LPF)** 过滤融合底色，仅以叠加高频残差信息的方式粘合到 Real-ESRGAN 的结果底图上。（公式 $\widetilde{T}_i^\prime = g(T_i) + LPF(f(T_i)) - LPF(g(T_i))$）。

---

## 5. TODO List

为了成功复现该论文方法，建议按照以下阶段推进代码：

- `[ ]` **Phase 1: 数据流搭建 (Data Engineering)**
  - 下载目标子数据集：HierText, TextOCR 等共七大开源数据集。
  - 使用 OpenCV 和真值回归参数编写边界框到小尺度水平文本的裁剪代码 (`cv2.getAffineTransform`)。
  - 集成基础版 Real-ESRGAN 退化管道，通过随机插值，生成 20 条随机劣化低质量配现代图。
  - 实现 Dataset 组件并支持 DataLoader 可以对数据按 $48 \times 480$ 进行 Resize 与 Padding，保证 Batch 的张量形状规整。
- `[ ]` **Phase 2: 架构搭建 (Architecture Definition)**
  - 对接 HuggingFace 获取 ByT5-Base，将输入文本 UTF-8 Tokenizer 开发至截取第一、二层隐层状态特征。
  - 编写 DDPM 核心组件：支持输入 `concat(Image Latent, LR Condition Image Channel)` 的带 ResBlock 层级降采样的 U-Net。
  - 实现 **Cross-Attention 层**，通过特定维度下采样层接收来自 ByT5 的多模序列输入，将其注入到特征网络中去。
- `[ ]` **Phase 3: 优化与训练流程 (Training loop)**
  - 设定 JAX (或者可替代的 PyTorch 分布式框架) 随机丢弃 (Dropout) Text 条件。
  - 构造 Loss：监督目标指向 HR 与 LR 的算术插值差异 (Res target)。配置批量为 $\sim$ 1000 级的大规模梯度累加优化器 (Adam 等)。
- `[ ]` **Phase 4: 推理管线与测试组装 (Inference Pipeline & Integration)**
  - 开发带 `omega` 双条件调节系数的 5-Steps DDIM 倒推器。
  - 编写 **Iterative OCR Refinement** 函数调用链：利用第三方工具 (如 EasyOCR / PaddleOCR) 对零参考图像第一阶段输出进行实时推断来反哺第二轮最终生成。
  - 实现后处理工具函数：支持双重逆向形变 `warpAffine`、高斯低通滤波算法以及将文本图块合成到底层背景框架之上的复合渲染逻辑。
