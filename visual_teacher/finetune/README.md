# Grounding DINO 微调训练

本目录包含 Grounding DINO 模型的微调训练代码。

## 目录结构

```
finetune/
├── __init__.py
├── odvg_dataset.py          # ODVG 格式数据集加载器
├── train_grounding_dino.py  # 训练主脚本
└── README.md
```

## 预训练权重下载

在开始训练前，需要下载 Grounding DINO 预训练权重到 `visual_teacher/pretrained_ckpt/` 目录：

### Swin-T 版本（推荐，轻量）
```bash
# 下载链接
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -O ../pretrained_ckpt/groundingdino_swint_ogc.pth
```

### Swin-B 版本（更大，精度更高）
```bash
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth -O ../pretrained_ckpt/groundingdino_swinb_cogcoor.pth
```

## 数据集格式

训练数据使用 ODVG (Object Detection Visual Grounding) 格式，每行一个 JSON：

```json
{
  "filename": "images/episode_00016_frame_00000.jpg",
  "height": 256,
  "width": 256,
  "grounding": {
    "caption": "bowl. drawer.",
    "regions": [
      {"phrase": "bowl", "bbox": [0.488, 0.594, 0.608, 0.714]},
      {"phrase": "drawer", "bbox": [0.192, 0.5, 0.388, 0.714]}
    ]
  }
}
```

其中：
- `bbox` 格式为 `[x1, y1, x2, y2]`，归一化到 0-1
- `caption` 是用于 grounding 的文本，每个 `phrase` 应该是 `caption` 的子串

## 使用方法

### 基础训练

```bash
cd visual_teacher/finetune

# 使用默认配置训练
python train_grounding_dino.py

# 指定参数
python train_grounding_dino.py \
    --train-jsonl ../../data_processed/grounding_dino_dataset/train.jsonl \
    --val-jsonl ../../data_processed/grounding_dino_dataset/val.jsonl \
    --image-dir ../../data_processed/grounding_dino_dataset \
    --pretrained ../pretrained_ckpt/groundingdino_swint_ogc.pth \
    --batch-size 4 \
    --epochs 50 \
    --lr 1e-4
```

### 冻结 Backbone 训练（推荐小数据集）

```bash
python train_grounding_dino.py --freeze-backbone --lr 1e-4 --epochs 30
```

### 从检查点恢复训练

```bash
python train_grounding_dino.py --resume checkpoints/grounding_dino_finetuned/checkpoint_epoch_10.pth
```

## 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch-size` | 2 | 批量大小 |
| `--epochs` | 50 | 训练轮数 |
| `--lr` | 1e-4 | 学习率 |
| `--weight-decay` | 1e-4 | 权重衰减 |
| `--lr-drop` | 40 | 学习率下降的 epoch |
| `--freeze-backbone` | False | 是否冻结 backbone |
| `--image-size` | 800 | 输入图像大小 |

## 输出

训练输出保存在 `checkpoints/grounding_dino_finetuned/`：

```
grounding_dino_finetuned/
├── logs/                    # TensorBoard 日志
├── best_model.pth          # 最佳验证损失模型
├── checkpoint_epoch_N.pth  # 定期保存的检查点
└── final_model.pth         # 最终模型
```

## TensorBoard 查看训练曲线

```bash
tensorboard --logdir checkpoints/grounding_dino_finetuned/logs
```

## 依赖

- torch >= 1.10
- torchvision
- transformers
- pillow
- tensorboard

## 注意事项

1. **显存需求**：Swin-T 版本在 batch_size=2 时约需 12GB 显存
2. **小数据集**：建议使用 `--freeze-backbone` 并适当增大学习率
3. **数据增强**：当前实现包含随机水平翻转和多尺度训练



