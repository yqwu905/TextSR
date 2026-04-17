# TextSR 运行报告

## 环境现状

| 项目 | 状态 |
|------|------|
| Python | 3.9 (系统自带) |
| PyTorch | 2.8.0 |
| CUDA | 不可用（Mac 本地，CPU only） |
| 核心依赖 | torch / torchvision / transformers / omegaconf / einops / cv2 / lmdb 全部已安装 ✓ |
| 可选依赖 | paddleocr / realesrgan / wandb 未安装 |

> **注意**：本机（Mac CPU）只适合验证代码正确性，训练和评估需在 GPU 服务器运行。

---

## 一、需要下载的内容

### 1. ByT5-base 文本编码器（约 580MB）

- **来源**：HuggingFace Hub，`google/byt5-base`
- **时机**：首次调用 `build_textsr_from_config()` 时自动下载，无需手动操作
- **缓存位置**：`~/.cache/huggingface/hub/`

如网络受限，提前手动缓存：

```bash
HF_ENDPOINT=https://hf-mirror.com python3 -c "
from transformers import AutoTokenizer, T5EncoderModel
AutoTokenizer.from_pretrained('google/byt5-base')
T5EncoderModel.from_pretrained('google/byt5-base')
print('Done')
"
```

### 2. TextZoom 数据集（约 10GB）

数据为 LMDB 格式，包含 train/test 三难度分组（easy / medium / hard）。

| 下载方式 | 地址 |
|----------|------|
| BaiduYun | https://pan.baidu.com/s/1KSNLv4EY3zFWHpBYlpFoDA（提取码：`m6uk`） |
| Google Drive | https://github.com/WenjiaWang0312/TextZoom README 中的链接 |

下载后组织为：

```
data/TextZoom/
  train/
    easy/     data.mdb  lock.mdb
    medium/   data.mdb  lock.mdb
    hard/     data.mdb  lock.mdb
  test/
    easy/     data.mdb  lock.mdb
    medium/   data.mdb  lock.mdb
    hard/     data.mdb  lock.mdb
```

**自动下载（需访问 Google Drive）**：

```bash
python3 scripts/download_textzoom.py --split all
```

**手动下载后验证**：

```bash
python3 scripts/download_textzoom.py --check
```

数据统计：训练集约 17,367 对，测试集约 3,021 对；LR 约 32×128，HR 约 64×256（2× SR）。

### 3. Real-ESRGAN 权重（可选，约 67MB）

仅在推理时使用 `--blend_esrgan` 选项才需要，首次调用会自动从 GitHub Releases 下载。

```bash
pip3 install realesrgan basicsr
```

---

## 二、训练前准备

### Step 1：安装 PaddleOCR

PaddleOCR 是文本条件的来源，训练和评估都依赖它。

```bash
# GPU 服务器（推荐）
pip3 install paddlepaddle-gpu
pip3 install paddleocr

# CPU 环境
pip3 install paddlepaddle
pip3 install paddleocr
```

### Step 2：预计算 OCR 标注缓存

训练时实时跑 OCR 会使数据加载成为瓶颈（慢约 10×）。建议提前计算并缓存到 JSON 文件。

```bash
# 计算训练集（最重要）
python3 scripts/prepare_annotations.py \
    --data_root ./data/TextZoom \
    --output_dir ./data/ocr_annotations \
    --split train \
    --use_gpu

# 同时计算测试集（评估时也会用）
python3 scripts/prepare_annotations.py \
    --data_root ./data/TextZoom \
    --output_dir ./data/ocr_annotations \
    --split test \
    --use_gpu
```

输出：
- `data/ocr_annotations/train_ocr.json`
- `data/ocr_annotations/test_ocr.json`

脚本支持断点续传，中断后重新运行会跳过已完成的样本。

**OCR 缓存在代码中的适配情况**：

- **训练集**（`TextZoomDataset`）：`_get_text()` 先查缓存，命中则直接返回；未命中且样本有 GT label 时直接用 GT label（训练集有标注，速度更快且更准确）；仅当 GT label 也为空时才实时跑 OCR。
- **测试集**（`TextZoomEvalDataset`）：先查缓存，命中则返回；未命中则实时跑 OCR；OCR 也失败时 fallback 到 GT label（仅用于保证不返回空字符串，不影响评估指标）。
- 配置文件中 `ocr_annotation_dir: ./data/ocr_annotations` 已与 `train.py` / `evaluate.py` 正确对接。

> 总结：训练集预计算价值有限（GT label 兜底），**测试集预计算收益更大**（避免评估时每张图实时 OCR 拖慢速度）。

---

## 三、启动训练

### 主要配置参数（`configs/textzoom_small.yaml`）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `batch_size` | 16（per GPU） | 显存不足时调小 |
| `num_epochs` | 100 | |
| `learning_rate` | 3e-4 | |
| `fp16` | true | 混合精度，需 GPU |
| `save_interval` | 5 | 每 N epoch 保存一次 checkpoint |
| `text_drop_prob` | 0.1 | CFG 训练时随机丢弃文本的概率 |
| `ddim_steps`（推理）| 5 | 越多越慢但质量更好 |
| `cfg_weight` | 2.0 | 文本引导强度 ω |

### 单 GPU 训练

```bash
python3 train.py --config configs/textzoom_small.yaml
```

### 多 GPU 训练（DDP）

```bash
torchrun --nproc_per_node=4 train.py --config configs/textzoom_small.yaml
```

### 从 checkpoint 续训

```bash
python3 train.py \
    --config configs/textzoom_small.yaml \
    --resume checkpoints/epoch_0009.pt
```

### 启用 WandB 日志

```bash
pip3 install wandb && wandb login
python3 train.py --config configs/textzoom_small.yaml --wandb
```

TensorBoard 日志默认自动开启（无需额外参数），保存在 `outputs/logs/`。

### Checkpoint 结构

每 5 epoch 保存两个文件：

```
checkpoints/
  epoch_0004.pt        # 完整训练状态（含 optimizer / scheduler / ema）
  epoch_0004_ema.pt    # 纯 EMA 权重（推理用这个）
```

---

## 四、推理

推理使用 EMA 权重（`epoch_XXXX_ema.pt`）效果更好。

### 单张图片

```bash
python3 inference.py \
    --config configs/textzoom_small.yaml \
    --checkpoint checkpoints/epoch_0099_ema.pt \
    --input path/to/lr_image.png \
    --output outputs/sr_image.png \
    --cfg_weight 2.0 \
    --ddim_steps 5 \
    --iter_rounds 1
```

### 批量处理目录

```bash
python3 inference.py \
    --config configs/textzoom_small.yaml \
    --checkpoint checkpoints/epoch_0099_ema.pt \
    --input_dir data/TextZoom/test/easy/ \
    --output_dir outputs/easy_sr/
```

### 关键推理参数

| 参数 | 含义 | 建议值 |
|------|------|--------|
| `--cfg_weight` | 文本引导强度 ω（0=无文本，2=论文设置） | 2.0 |
| `--ddim_steps` | DDIM 去噪步数，越多质量越好但越慢 | 5–20 |
| `--iter_rounds` | 迭代 OCR 精化轮数 R（0=直接用 LR OCR，1=先无文本 SR 再 OCR） | 1 |
| `--blend_esrgan` | 混合 Real-ESRGAN 低频成分（需安装 realesrgan） | 可选 |

**迭代精化流程（`--iter_rounds 1`）**：
1. 先做无文本引导的 SR（`cfg_weight=1.0`）
2. 对中间结果跑 OCR 得到更准确的文字
3. 用 OCR 文字再做一次有文本引导的 SR

---

## 五、评估

```bash
python3 evaluate.py \
    --config configs/textzoom_small.yaml \
    --checkpoint checkpoints/epoch_0099_ema.pt \
    --output_dir outputs/eval/ \
    --cfg_weight 2.0 \
    --ddim_steps 5 \
    --iter_rounds 1
```

输出格式（word-level accuracy，与论文 Table 1 对齐）：

```
  easy    : 73.5% (1234/1678)
  medium  : 61.2% (789/1289)
  hard    : 48.7% (26/54)
  overall : 68.1% (2049/3021)
```

详细记录保存在 `outputs/eval/results.json`，SR 图片保存在 `outputs/eval/{easy,medium,hard}/`。

**只评估 bicubic 基线**（无需加载模型）：

```bash
python3 evaluate.py --config configs/textzoom_small.yaml --baseline_only
```

---

## 六、验证安装

```bash
# 快速检查依赖和模型初始化
python3 scripts/verify_install.py

# 完整 forward pass 测试（不需要数据）
python3 scripts/test_forward.py

# 完整检查（含 ByT5 下载，约 580MB）
python3 scripts/verify_install.py --full
```
