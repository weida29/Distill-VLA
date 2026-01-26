# Data Processing Pipeline

将 LIBERO RLDS 数据集处理为 Grounding DINO 训练格式的完整流程。

## Pipeline 概览

```
RLDS Dataset
     │
     ▼ Step 1: extract_keyframes.py
Keyframes (images + prompts)
     │
     ▼ Step 2: prepare_vlm_input.py
VLM Input (base64 JSONL)
     │
     ▼ Step 3: vlm_annotation.py
BBox Annotations (JSONL)
     │
     ▼ Step 4: convert_to_grounding_dino.py
ODVG Dataset (for GDINO training)
```

## 快速开始

### 1. 修改配置

编辑 `scripts/config.sh`，设置以下路径：

```bash
# 项目根目录
export PROJECT_ROOT="/path/to/VLA-Adapter"

# RLDS 数据集目录
export RLDS_DATA_DIR="${PROJECT_ROOT}/data/modified_libero_rlds"

# VLM API 配置 (vLLM server)
export OPENAI_API_BASE_URL="http://127.0.0.1:18000/v1"
export OPENAI_MODEL="Qwen3-VL-30B-A3B-Instruct"
```

### 2. 运行完整 Pipeline

```bash
cd vla-scripts/data_process/scripts

# 运行所有步骤
./run_all.sh

# 或者跳过 VLM 标注（如果已有 bbox.json）
./run_all.sh --skip-vlm
```

### 3. 分步运行

```bash
# Step 1: 从 RLDS 提取关键帧
./step1_extract_keyframes.sh

# Step 2: 准备 VLM 输入
./step2_prepare_vlm_input.sh

# Step 3: VLM 标注（需要先启动 VLM 服务器）
# vllm serve Qwen3-VL-30B-A3B-Instruct --port 18000 -tp 8
./step3_vlm_annotation.sh

# Step 4: 转换为 Grounding DINO 格式
./step4_convert_to_gdino.sh
```

## 目录结构

```
data_process/
├── scripts/                    # Bash 脚本
│   ├── config.sh              # 配置文件 (修改路径)
│   ├── step1_extract_keyframes.sh
│   ├── step2_prepare_vlm_input.sh
│   ├── step3_vlm_annotation.sh
│   ├── step4_convert_to_gdino.sh
│   └── run_all.sh             # 运行所有步骤
│
├── extract_keyframes.py        # Step 1: 提取关键帧
├── prepare_vlm_input.py        # Step 2: 准备 VLM 输入
├── convert_to_grounding_dino.py # Step 4: 转换为 ODVG 格式
│
├── vlm_annotator/              # Step 3: VLM 标注
│   └── vlm_annotation.py
│
├── extract_libero_images.py    # (旧版) 提取图像
├── frames_to_video.py          # 工具: 帧转视频
└── visualize_bbox.ipynb        # 工具: 可视化 bbox
```

## 输出结构

```
data_processed/
├── keyframes/                  # Step 1 输出
│   ├── libero_spatial_no_noops/
│   │   ├── episode_00000/
│   │   │   ├── prompt.txt
│   │   │   ├── frame_00000.jpg
│   │   │   └── ...
│   │   └── ...
│   └── metadata.json
│
├── vlm_input.jsonl             # Step 2 输出
├── bbox.json                   # Step 3 输出
│
└── grounding_dino_dataset/     # Step 4 输出 (最终数据集)
    ├── images/
    ├── train.jsonl
    ├── val.jsonl
    └── meta.json
```

## 配置说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `RLDS_DATA_DIR` | RLDS 数据集路径 | `data/modified_libero_rlds` |
| `SUBSETS` | 要处理的子集 | 全部 4 个 LIBERO 子集 |
| `SAMPLE_RATE` | 帧采样率 (每 N 帧取 1 帧) | 5 |
| `VLM_FRAME_NUM` | VLM 推理使用的帧数 | 6 |
| `VLM_MAX_WORKERS` | VLM 并行工作数 | 8 |
| `VAL_RATIO` | 验证集比例 | 0.1 |

## 常见问题

### Q: VLM 服务器怎么启动？

```bash
# 使用 vLLM 启动
vllm serve Qwen3-VL-30B-A3B-Instruct --port 18000 -tp 8

# 或使用其他 OpenAI 兼容的 API
```

### Q: 如何只处理部分子集？

修改 `config.sh` 中的 `SUBSETS` 变量：

```bash
export SUBSETS="libero_spatial_no_noops"  # 只处理一个子集
```

### Q: 如何使用已有的 bbox.json？

```bash
./run_all.sh --skip-vlm
```

### Q: 如何可视化标注结果？

使用 `visualize_bbox.ipynb` notebook 查看标注效果。

