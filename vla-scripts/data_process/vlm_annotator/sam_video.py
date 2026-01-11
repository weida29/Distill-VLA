"""
SAM2 视频追踪脚本 - 根据首帧 BBox 追踪物体到整个视频

基于 sam_video.py 适配到 VLA-Adapter 项目

输入: 
    - data_processed/bbox.json (VLM 标注结果，支持多 bbox 格式)
    - data_processed/videos/ (视频文件)
输出: 
    - data_processed/annotations/sam_results/*.pkl

支持多 GPU 自动并行处理，兼容新旧两种 JSON 格式:
    - 新格式: labels (数组) + bboxes_2d (数组) - 支持多物体追踪
    - 旧格式: label (单值) + bbox_2d (单值) - 单物体追踪

使用示例:
    # 单 GPU 处理
    python sam_video.py --gpus 0

    # 多 GPU 并行处理 (自动分配任务)
    python sam_video.py --gpus 0,1,2,3

    # 使用所有可用 GPU
    python sam_video.py --gpus all

    # 并行处理 + 可视化前 5 个结果
    python sam_video.py --gpus 0,1 --visualize --num_visualize 5
"""

import json
import pickle
import argparse
import os
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np

try:
    import imageio.v3 as iio
except ImportError:
    print("Please install imageio: pip install imageio[pyav]")
    sys.exit(1)

try:
    from ultralytics.models.sam import SAM2VideoPredictor
except ImportError:
    print("Please install ultralytics: pip install ultralytics")
    sys.exit(1)

try:
    import cv2
    from PIL import Image
except ImportError:
    cv2 = None
    Image = None


def visualize_tracking_results(
    result_data: dict,
    output_path: Path,
    num_frames: int = 8,
    figsize: tuple = (20, 10)
):
    """
    可视化单个视频的追踪结果（支持多物体）
    
    Args:
        result_data: SAM2 追踪结果字典（包含 video_path, labels, frame_results 等）
        output_path: 输出图片路径
        num_frames: 显示的帧数
        figsize: 图片大小
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    video_path = Path(result_data.get('video_path', ''))
    frame_results = result_data.get('frame_results', [])
    labels = result_data.get('labels', [])
    prompt = result_data.get('prompt', '')
    
    if not frame_results:
        print(f"  No tracking results for {video_path}")
        return
    
    # 颜色列表（用于区分不同物体）
    colors = ['lime', 'red', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
    
    # 选择要显示的帧索引（均匀分布）
    total_results = len(frame_results)
    if total_results <= num_frames:
        selected_indices = list(range(total_results))
    else:
        selected_indices = np.linspace(0, total_results - 1, num_frames, dtype=int).tolist()
    
    n_cols = min(4, len(selected_indices))
    n_rows = (len(selected_indices) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    axes = axes.flatten()
    
    stride = result_data.get('stride', 1)
    
    # 读取视频帧并绘制
    for plot_idx, result_idx in enumerate(selected_indices):
        result = frame_results[result_idx]
        frame_idx = result.get('frame_idx', result_idx * stride)
        
        # 读取对应帧
        try:
            frame = iio.imread(str(video_path), index=frame_idx, plugin="pyav")
        except Exception as e:
            print(f"  Failed to read frame {frame_idx}: {e}")
            continue
        
        ax = axes[plot_idx]
        ax.imshow(frame)
        
        # 绘制 bbox（每个物体用不同颜色）
        bboxes = result.get('bbox', [])
        if len(bboxes) > 0:
            for obj_idx, bbox in enumerate(bboxes):
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = bbox[:4]
                    color = colors[obj_idx % len(colors)]
                    rect = patches.Rectangle(
                        (x1, y1), x2 - x1, y2 - y1,
                        linewidth=2, edgecolor=color, facecolor='none'
                    )
                    ax.add_patch(rect)
                    
                    # 添加标签
                    if obj_idx < len(labels):
                        ax.text(x1, max(y1 - 5, 10), labels[obj_idx],
                               color='white', fontsize=8, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.8))
        
        # 绘制 mask（如果有）
        masks = result.get('mask', [])
        if masks:
            for obj_idx, mask_points in enumerate(masks):
                if len(mask_points) > 0:
                    mask_array = np.array(mask_points)
                    if len(mask_array.shape) == 2 and mask_array.shape[1] == 2:
                        color = colors[obj_idx % len(colors)]
                        ax.fill(mask_array[:, 0], mask_array[:, 1], 
                               alpha=0.3, color=color, edgecolor=color, linewidth=1)
        
        ax.set_title(f'Frame {frame_idx}', fontsize=10)
        ax.axis('off')
    
    # 隐藏多余的子图
    for i in range(len(selected_indices), len(axes)):
        axes[i].axis('off')
    
    # 添加总标题（包含物体标签和 prompt）
    video_name = video_path.stem
    labels_str = ', '.join(labels) if labels else 'N/A'
    title = f'SAM2 Tracking: {video_name}\nObjects: {labels_str}'
    if prompt:
        title += f'\nPrompt: {prompt[:80]}{"..." if len(prompt) > 80 else ""}'
    fig.suptitle(title, fontsize=12, y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved visualization: {output_path}")


def process_item(
    predictor, 
    bbox_data_item: dict, 
    device_id: int, 
    index: int,
    video_dir: Path
) -> dict:
    """
    处理单个视频条目，执行 SAM2 推理并返回结果列表
    
    Args:
        predictor: SAM2VideoPredictor 实例
        bbox_data_item: 包含 video_path 和 bbox 的字典
        device_id: GPU ID
        index: 数据索引
        video_dir: 视频目录
    
    Returns:
        包含追踪结果的字典，包括元数据和每帧结果
    """
    video_path_str = bbox_data_item.get('video_path', '')
    
    # 构建实际视频路径
    # video_path 格式: /tmp/VLA-Adapter/data_processed/keyframes/libero_10_no_noops/episode_00001
    # 实际视频路径: data_processed/videos/libero_10_no_noops/episode_00001.mp4
    
    if video_path_str:
        path_parts = Path(video_path_str).parts
        # 找到 episode 名称和 subset 名称
        episode_name = path_parts[-1]  # episode_00001
        subset_name = path_parts[-2]   # libero_10_no_noops
        video_path = video_dir / subset_name / f"{episode_name}.mp4"
    else:
        print(f"GPU {device_id} | Skipping index {index}: No video_path")
        return None
    
    # 检查视频是否存在
    if not video_path.exists():
        print(f"GPU {device_id} | Skipping index {index}: Video not found at {video_path}")
        return None

    try:
        # 1. 获取视频首帧尺寸用于坐标缩放
        frame = iio.imread(str(video_path), index=0, plugin="pyav")
        h, w = frame.shape[:2]

        # 2. 获取 BBox - 兼容新旧两种 JSON 格式
        result_data = bbox_data_item.get('result', {})
        
        # 新格式: labels (数组) + bboxes_2d (数组)
        # 旧格式: label (单值) + bbox_2d (单值)
        if 'bboxes_2d' in result_data and 'labels' in result_data:
            # 新格式：多 bbox
            raw_bboxes = result_data['bboxes_2d']
            labels = result_data['labels']
        elif 'bbox_2d' in result_data:
            # 旧格式：单 bbox
            raw_bboxes = [result_data['bbox_2d']]
            labels = [result_data.get('label', 'object')]
        else:
            print(f"GPU {device_id} | Skipping index {index}: No bbox data found")
            return None
        
        if not raw_bboxes or len(raw_bboxes) == 0:
            print(f"GPU {device_id} | Skipping index {index}: Empty bbox list")
            return None
        
        # 转换所有 bbox 为像素坐标（输入是归一化到 0-1000 的格式）
        bboxes_pixel = []
        for bbox in raw_bboxes:
            bbox_pixel = [
                bbox[0] * w / 1000,
                bbox[1] * h / 1000,
                bbox[2] * w / 1000,
                bbox[3] * h / 1000
            ]
            bboxes_pixel.append(bbox_pixel)
        
        # 为每个 bbox 分配一个唯一的 label ID (从 1 开始)
        label_ids = list(range(1, len(bboxes_pixel) + 1))

        # 3. 重置 Predictor 状态 (关键步骤)
        if hasattr(predictor, "inference_state"):
            predictor.inference_state = {}
        predictor.reset_image()

        # 4. 执行推理 - 传入所有 bbox 和对应的 label IDs
        results = predictor(source=str(video_path), bboxes=bboxes_pixel, labels=label_ids)
        
        # 5. 获取当前使用的步长 (stride)
        current_stride = predictor.args.vid_stride
        
        frame_results = []
        for i, res in enumerate(results):
            # 转换数据为 numpy/list 以节省内存
            bbox = res.boxes.xyxy.cpu().numpy() if res.boxes else []
            mask = res.masks.xy if res.masks else []
            
            # 计算真实的视频帧索引
            real_frame_idx = i * current_stride
            
            result_dict = {
                "result_seq_idx": i,
                "frame_idx": real_frame_idx,
                "bbox": bbox,
                "mask": mask,
            }
            frame_results.append(result_dict)

        # 返回包含元数据的完整结果
        return {
            "video_path": str(video_path),
            "episode_name": episode_name,
            "subset_name": subset_name,
            "prompt": bbox_data_item.get('prompt', ''),
            "labels": labels,
            "input_bboxes": raw_bboxes,
            "input_bboxes_pixel": bboxes_pixel,
            "image_size": [w, h],
            "stride": current_stride,
            "frame_results": frame_results
        }
            
    except Exception as e:
        print(f"GPU {device_id} | ERROR processing index {index} - {video_path}: {e}")
        return None


def worker_process(
    device_id: int,
    all_gpu_nums: int,
    bbox_data: list,
    args,
    project_root: Path
):
    """
    单个 GPU 的工作进程
    """
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    
    video_dir = project_root / args.video_dir
    output_dir = project_root / args.output_dir
    vis_output_dir = project_root / args.vis_output_dir if args.visualize else None
    
    output_dir.mkdir(parents=True, exist_ok=True)
    if vis_output_dir:
        vis_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 计算分片索引
    indices_to_process = [
        i for i in range(len(bbox_data)) 
        if i % all_gpu_nums == device_id
    ]
    
    print(f"GPU {device_id}: Responsible for {len(indices_to_process)} items")
    
    if not indices_to_process:
        print(f"GPU {device_id}: No items to process")
        return
    
    # 初始化模型
    overrides = dict(
        conf=0.25,
        task="segment",
        mode="predict",
        imgsz=args.imgsz,
        model=args.model,
        vid_stride=args.vid_stride,
        verbose=False,
        save=False
    )
    
    try:
        print(f"GPU {device_id}: Initializing SAM2VideoPredictor...")
        predictor = SAM2VideoPredictor(overrides=overrides)
        print(f"GPU {device_id}: Model loaded successfully")
    except Exception as e:
        print(f"Error initializing model on device {device_id}: {e}")
        return
    
    # 处理循环
    current_chunk_results = []
    chunk_counter = 0
    visualized_count = 0

    for local_i, global_idx in enumerate(tqdm(
        indices_to_process, 
        desc=f"GPU {device_id}", 
        position=device_id
    )):
        item = bbox_data[global_idx]
        
        result = process_item(predictor, item, device_id, global_idx, video_dir)
        
        if result is not None:
            current_chunk_results.append(result)
            
            # 可视化逻辑
            if args.visualize and visualized_count < args.num_visualize:
                subset_name = result.get('subset_name', 'unknown')
                episode_name = result.get('episode_name', 'unknown')
                vis_filename = vis_output_dir / f"{subset_name}_{episode_name}_tracking.png"
                print(f"\nGPU {device_id}: Visualizing video {visualized_count + 1}/{args.num_visualize}...")
                visualize_tracking_results(
                    result_data=result,
                    output_path=vis_filename,
                    num_frames=8
                )
                visualized_count += 1

        # 分块保存逻辑
        if (local_i + 1) % args.save_interval == 0:
            chunk_filename = output_dir / f"results_gpu_{device_id}_chunk_{chunk_counter}.pkl"
            print(f"GPU {device_id}: Saving chunk {chunk_counter} ({len(current_chunk_results)} videos)...")
            
            with open(chunk_filename, "wb") as f:
                pickle.dump(current_chunk_results, f)
            
            current_chunk_results = []
            chunk_counter += 1
    
    # 保存剩余数据
    if current_chunk_results:
        chunk_filename = output_dir / f"results_gpu_{device_id}_chunk_{chunk_counter}.pkl"
        print(f"GPU {device_id}: Saving final chunk {chunk_counter} ({len(current_chunk_results)} videos)...")
        with open(chunk_filename, "wb") as f:
            pickle.dump(current_chunk_results, f)

    print(f"GPU {device_id}: Finished. All chunks saved to {output_dir}")
    
    if args.visualize:
        print(f"GPU {device_id}: Visualized {visualized_count} videos to {vis_output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Run SAM2VideoPredictor on dataset with multi-GPU support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 使用单个 GPU (GPU 0)
  python sam_video.py --gpus 0
  
  # 使用多个 GPU 并行 (GPU 0, 1, 2, 3)
  python sam_video.py --gpus 0,1,2,3
  
  # 使用所有可用 GPU
  python sam_video.py --gpus all
  
  # 并行处理 + 可视化前 5 个结果
  python sam_video.py --gpus 0,1 --visualize --num_visualize 5
        """
    )
    parser.add_argument(
        "--gpus", 
        type=str, 
        default="0",
        help="GPU IDs to use, comma-separated (e.g., '0,1,2,3') or 'all' for all available GPUs"
    )
    parser.add_argument(
        "--jsonl_path", 
        type=str, 
        default="data_processed/bbox.json",
        help="Path to the input JSONL file containing bbox data"
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="data_processed/videos",
        help="Directory containing video files"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data_processed/annotations/sam_results",
        help="Directory to save the output .pkl files"
    )
    parser.add_argument(
        "--save_interval", 
        type=int, 
        default=100, 
        help="Number of items to process before saving a chunk"
    )
    parser.add_argument(
        "--vid_stride",
        type=int,
        default=5,
        help="Video frame stride (process every N frames)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=512,
        help="Input image size"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sam2.1_l.pt",
        help="SAM2 model name"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize tracking results for first N videos"
    )
    parser.add_argument(
        "--num_visualize",
        type=int,
        default=5,
        help="Number of videos to visualize (default: 5)"
    )
    parser.add_argument(
        "--vis_output_dir",
        type=str,
        default="data_processed/sam_visualization",
        help="Directory to save visualization results"
    )
    args = parser.parse_args()

    # 找到项目根目录
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent.parent
    
    jsonl_path = project_root / args.jsonl_path
    output_dir = project_root / args.output_dir
    
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)

    # 解析 GPU 列表
    if args.gpus.lower() == 'all':
        import torch
        if torch.cuda.is_available():
            gpu_ids = list(range(torch.cuda.device_count()))
        else:
            print("Error: No CUDA GPUs available")
            sys.exit(1)
    else:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
    
    print(f"=" * 60)
    print(f"SAM2 Video Tracking - Multi-GPU")
    print(f"=" * 60)
    print(f"GPUs to use: {gpu_ids}")
    print(f"JSONL path: {jsonl_path}")
    print(f"Video dir: {project_root / args.video_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Model: {args.model}")
    print(f"Video stride: {args.vid_stride}")
    if args.visualize:
        print(f"Visualization: enabled (first {args.num_visualize} videos per GPU)")
        print(f"Vis output dir: {project_root / args.vis_output_dir}")
    print(f"=" * 60)

    # --- 1. 加载数据 ---
    print(f"Loading data from {jsonl_path}...")
    
    if not jsonl_path.exists():
        print(f"Error: JSONL file not found: {jsonl_path}")
        print("Please run vlm_annotation.py first")
        sys.exit(1)
    
    bbox_data = []
    with open(jsonl_path, "r", encoding='utf-8') as f:
        for line in f:
            if line.strip():
                bbox_data.append(json.loads(line))
    
    # 排序以确保多进程分片时的确定性
    bbox_data = sorted(bbox_data, key=lambda x: x.get('video_path', ''))
    print(f"Loaded {len(bbox_data)} items")

    # --- 2. 启动多进程 ---
    if len(gpu_ids) == 1:
        # 单 GPU 模式，直接在主进程运行
        print(f"\nRunning on single GPU: {gpu_ids[0]}")
        worker_process(
            device_id=gpu_ids[0],
            all_gpu_nums=1,
            bbox_data=bbox_data,
            args=args,
            project_root=project_root
        )
    else:
        # 多 GPU 模式，使用 multiprocessing
        import multiprocessing as mp
        mp.set_start_method('spawn', force=True)
        
        print(f"\nStarting {len(gpu_ids)} parallel processes...")
        processes = []
        
        for idx, gpu_id in enumerate(gpu_ids):
            p = mp.Process(
                target=worker_process,
                args=(gpu_id, len(gpu_ids), bbox_data, args, project_root),
                name=f"GPU-{gpu_id}"
            )
            p.start()
            processes.append(p)
            print(f"  Started process for GPU {gpu_id}")
        
        # 等待所有进程完成
        for p in processes:
            p.join()
        
        print(f"\n{'=' * 60}")
        print("All processes completed!")
        print(f"Results saved to: {output_dir}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()


"""
使用示例:

# 单 GPU 处理 (GPU 0)
python sam_video.py --gpus 0

# 使用指定的多个 GPU 并行处理
python sam_video.py --gpus 0,1,2,3

# 使用所有可用 GPU
python sam_video.py --gpus all

# 并行处理 + 可视化前 5 个结果
python sam_video.py --gpus 0,1 --visualize --num_visualize 5

# 指定输入输出路径
python sam_video.py --gpus 0,1,2,3 \\
    --jsonl_path data_processed/bbox.json \\
    --video_dir data_processed/videos \\
    --output_dir data_processed/annotations/sam_results

# 合并结果
python -c "
import pickle
from pathlib import Path

output_dir = Path('data_processed/annotations/sam_results')
all_results = []
for pkl_file in sorted(output_dir.glob('*.pkl')):
    with open(pkl_file, 'rb') as f:
        all_results.extend(pickle.load(f))

print(f'Total videos processed: {len(all_results)}')

# 保存合并结果
with open(output_dir / 'all_results.pkl', 'wb') as f:
    pickle.dump(all_results, f)
"
"""

