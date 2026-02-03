# Open-GroundingDino 推理指南

本文档说明如何使用训练好的 Open-GroundingDino 模型进行推理。

## 快速开始

### 方法 1: 使用推理脚本（推荐）

```bash
cd visual_teacher/Open-GroundingDino

# 基本用法
bash infer_libero.sh \
    checkpoints/open_gdino_finetuned/checkpoint.pth \
    path/to/image.jpg \
    "pick up the bowl"

# 自定义阈值和输出目录
bash infer_libero.sh \
    checkpoints/open_gdino_finetuned/checkpoint.pth \
    path/to/image.jpg \
    "basket" \
    --box_threshold 0.4 \
    --text_threshold 0.3 \
    --output_dir my_outputs
```

### 方法 2: 直接使用 Python 脚本

```bash
cd visual_teacher/Open-GroundingDino

python tools/inference_on_a_image.py \
    --config_file config/cfg_odvg.py \
    --checkpoint_path checkpoints/open_gdino_finetuned/checkpoint.pth \
    --image_path path/to/image.jpg \
    --text_prompt "pick up the bowl" \
    --output_dir outputs \
    --box_threshold 0.3 \
    --text_threshold 0.25
```

## 参数说明

### 必需参数

- `checkpoint_path`: 训练好的模型检查点路径
  - 例如: `checkpoints/open_gdino_finetuned/checkpoint.pth`
  - 训练完成后，检查点通常保存在 `checkpoints/open_gdino_finetuned/` 目录下

- `image_path`: 要推理的图片路径
  - 支持常见图片格式: `.jpg`, `.png`, `.jpeg` 等

- `text_prompt`: 文本提示（描述要检测的物体或动作）
  - 例如: `"pick up the bowl"`, `"basket"`, `"black bowl and plate"`

### 可选参数

- `--box_threshold`: 边界框置信度阈值（默认: 0.3）
  - 值越高，检测越严格，只保留高置信度的检测框
  - 建议范围: 0.2 - 0.5

- `--text_threshold`: 文本匹配阈值（默认: 0.25）
  - 控制文本与检测框的匹配程度
  - 建议范围: 0.2 - 0.4

- `--output_dir`: 输出目录（默认: `outputs`）
  - 推理结果会保存在此目录
  - 包含两个文件:
    - `raw_image.jpg`: 原始图片
    - `pred.jpg`: 带有检测框和标签的图片

## 使用示例

### 示例 1: 检测单个物体

```bash
bash infer_libero.sh \
    checkpoints/open_gdino_finetuned/checkpoint.pth \
    data_processed/open_gdino_dataset/images/episode_00156_frame_00000.jpg \
    "ketchup"
```

### 示例 2: 检测多个物体

```bash
bash infer_libero.sh \
    checkpoints/open_gdino_finetuned/checkpoint.pth \
    data_processed/open_gdino_dataset/images/episode_00156_frame_00000.jpg \
    "ketchup. basket."
```

### 示例 3: 使用完整指令作为提示

```bash
bash infer_libero.sh \
    checkpoints/open_gdino_finetuned/checkpoint.pth \
    data_processed/open_gdino_dataset/images/episode_00156_frame_00000.jpg \
    "pick up the ketchup and place it in the basket"
```

### 示例 4: 批量推理（使用循环）

```bash
# 创建输出目录
mkdir -p inference_results

# 遍历图片目录
for img in data_processed/open_gdino_dataset/images/*.jpg; do
    filename=$(basename "$img")
    bash infer_libero.sh \
        checkpoints/open_gdino_finetuned/checkpoint.pth \
        "$img" \
        "detect all objects" \
        --output_dir "inference_results/${filename%.*}"
done
```

## 输出说明

推理完成后，在输出目录中会生成：

1. **raw_image.jpg**: 原始输入图片
2. **pred.jpg**: 带有检测框和标签的图片
   - 每个检测框用不同颜色的矩形框标出
   - 标签显示在框的上方，格式为: `物体名称(置信度)`
   - 例如: `ketchup(0.85)`, `basket(0.92)`

## 常见问题

### Q1: 找不到检查点文件

**问题**: `Error: Checkpoint not found at ...`

**解决**: 
- 确认训练已完成
- 检查点通常保存在 `checkpoints/open_gdino_finetuned/` 目录
- 检查点文件名可能是 `checkpoint.pth` 或 `checkpoint_epoch_X.pth`

### Q2: 检测结果为空

**可能原因**:
1. 阈值设置过高，尝试降低 `--box_threshold` 和 `--text_threshold`
2. 文本提示与训练数据不匹配
3. 图片内容与训练数据差异较大

**解决**:
```bash
# 降低阈值
bash infer_libero.sh ... --box_threshold 0.2 --text_threshold 0.15
```

### Q3: 检测框太多（误检）

**解决**: 提高阈值
```bash
bash infer_libero.sh ... --box_threshold 0.4 --text_threshold 0.3
```

### Q4: 使用训练时的配置文件

如果训练时使用了自定义配置（如 `cfg_libero_train.py`），需要修改推理脚本中的 `CONFIG` 变量：

```bash
# 编辑 infer_libero.sh，修改 CONFIG 路径
CONFIG="${SCRIPT_DIR}/config/cfg_libero_train.py"
```

或者直接使用 Python 脚本并指定配置文件：

```bash
python tools/inference_on_a_image.py \
    --config_file config/cfg_libero_train.py \
    ...
```

## 高级用法

### 使用 token_spans 进行精确匹配

如果需要检测文本提示中的特定短语，可以使用 `--token_spans` 参数：

```bash
python tools/inference_on_a_image.py \
    --config_file config/cfg_odvg.py \
    --checkpoint_path checkpoints/open_gdino_finetuned/checkpoint.pth \
    --image_path image.jpg \
    --text_prompt "pick up the black bowl and place it on the plate" \
    --token_spans "[[[10, 15]], [[35, 40]]]" \
    --box_threshold 0.3 \
    --output_dir outputs
```

`token_spans` 格式说明：
- `[[[10, 15]]]`: 检测 "black bowl"（从位置 10 到 15）
- `[[[35, 40]]]`: 检测 "plate"（从位置 35 到 40）

## 注意事项

1. **模型配置**: 确保使用的配置文件与训练时一致
2. **BERT 路径**: 如果训练时使用了自定义 BERT 路径，推理时也需要相同配置
3. **GPU 内存**: 推理时模型会加载到 GPU，确保有足够的显存
4. **文本提示**: 使用与训练数据相似的文本格式会获得更好的效果

## 相关文件

- `tools/inference_on_a_image.py`: 推理脚本
- `config/cfg_odvg.py`: 模型配置文件
- `infer_libero.sh`: 便捷推理脚本
- `train_libero.sh`: 训练脚本（参考检查点路径）




