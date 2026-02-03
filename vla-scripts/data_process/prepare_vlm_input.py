"""
将 LIBERO 数据集的关键帧图片编码为 base64 并生成 JSONL 文件
供 VLM 标注脚本使用

输入: data_processed/keyframes/ 目录下的图片序列
输出: data_processed/vlm_input.jsonl

JSONL 格式:
{
    "video_path": "data_processed/keyframes/libero_goal_no_noops/episode_00001",
    "prompt": "put the wine bottle on the rack",
    "frames_base64": ["base64_frame_0", "base64_frame_5", ...]
}
"""

import os
import sys
import json
import base64
import argparse
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def encode_image_to_base64(image_path: str) -> str:
    """将图片文件编码为 base64 字符串"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def process_episode(episode_dir: Path, sample_num: int = None) -> dict:
    """
    处理单个 episode 目录，返回 JSONL 条目
    
    Args:
        episode_dir: episode 目录路径
        sample_num: 采样帧数，None 表示使用所有帧
    
    Returns:
        JSONL 条目字典
    """
    # 获取所有帧文件并排序
    frame_files = sorted([
        f for f in episode_dir.iterdir() 
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png'] and f.name.startswith('frame_')
    ])
    
    if not frame_files:
        return None
    
    # 读取 prompt
    prompt_file = episode_dir / "prompt.txt"
    prompt = ""
    if prompt_file.exists():
        prompt = prompt_file.read_text(encoding='utf-8').strip()
    
    # 采样帧
    if sample_num and len(frame_files) > sample_num:
        import numpy as np
        indices = np.linspace(0, len(frame_files) - 1, sample_num, dtype=int)
        frame_files = [frame_files[i] for i in indices]
    
    # 编码帧为 base64
    frames_base64 = []
    for frame_file in frame_files:
        try:
            b64 = encode_image_to_base64(str(frame_file))
            frames_base64.append(b64)
        except Exception as e:
            print(f"Warning: Failed to encode {frame_file}: {e}")
            continue
    
    if not frames_base64:
        return None
    
    return {
        "video_path": str(episode_dir),
        "prompt": prompt,
        "frames_base64": frames_base64
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare VLM input from keyframes")
    parser.add_argument(
        "--input_dir", 
        type=str, 
        default="data_processed/keyframes",
        help="Directory containing keyframes"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="data_processed/vlm_input.jsonl",
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--sample_num", 
        type=int, 
        default=8,
        help="Number of frames to sample per episode (default: 8)"
    )
    parser.add_argument(
        "--max_workers", 
        type=int, 
        default=8,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--subset", 
        type=str, 
        default=None,
        help="Only process specific subset (e.g., libero_goal_no_noops)"
    )
    args = parser.parse_args()
    
    # 找到项目根目录
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    
    input_dir = project_root / args.input_dir
    output_file = project_root / args.output_file
    
    print(f"Input directory: {input_dir}")
    print(f"Output file: {output_file}")
    print(f"Sample frames per episode: {args.sample_num}")
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)
    
    # 收集所有 episode 目录
    episode_dirs = []
    for subset_dir in input_dir.iterdir():
        if not subset_dir.is_dir():
            continue
        if args.subset and subset_dir.name != args.subset:
            continue
        
        for episode_dir in subset_dir.iterdir():
            if episode_dir.is_dir() and episode_dir.name.startswith('episode_'):
                episode_dirs.append(episode_dir)
    
    print(f"Found {len(episode_dirs)} episodes to process")
    
    if not episode_dirs:
        print("No episodes found!")
        sys.exit(1)
    
    # 确保输出目录存在
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 并行处理
    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(process_episode, ep_dir, args.sample_num): ep_dir 
            for ep_dir in episode_dirs
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            ep_dir = futures[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                print(f"Error processing {ep_dir}: {e}")
    
    # 按路径排序
    results.sort(key=lambda x: x['video_path'])
    
    # 写入 JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\nDone! Processed {len(results)} episodes")
    print(f"Output saved to: {output_file}")


if __name__ == "__main__":
    main()












