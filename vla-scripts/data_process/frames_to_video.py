"""
将 LIBERO 数据集的关键帧图片序列转换为视频
供 SAM2 追踪脚本使用

输入: data_processed/keyframes/ 目录下的图片序列
输出: data_processed/videos/ 目录下的 MP4 视频

视频命名规则:
    data_processed/videos/{subset}/{episode_id}.mp4
"""

import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import imageio.v3 as iio
except ImportError:
    print("Please install imageio: pip install imageio[pyav]")
    sys.exit(1)


def frames_to_video(
    frame_dir: Path, 
    output_path: Path, 
    fps: int = 10,
    codec: str = "libx264"
) -> bool:
    """
    将图片序列转换为视频
    
    Args:
        frame_dir: 包含帧图片的目录
        output_path: 输出视频路径
        fps: 帧率
        codec: 视频编码器
    
    Returns:
        是否成功
    """
    # 获取所有帧文件并排序
    frame_files = sorted([
        f for f in frame_dir.iterdir() 
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png'] and f.name.startswith('frame_')
    ])
    
    if not frame_files:
        return False
    
    try:
        # 读取所有帧
        frames = []
        for frame_file in frame_files:
            frame = iio.imread(str(frame_file))
            frames.append(frame)
        
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 写入视频
        iio.imwrite(
            str(output_path),
            frames,
            fps=fps,
            codec=codec,
            plugin="pyav"
        )
        
        return True
        
    except Exception as e:
        print(f"Error converting {frame_dir} to video: {e}")
        return False


def process_episode(episode_dir: Path, output_dir: Path, fps: int) -> dict:
    """
    处理单个 episode
    
    Args:
        episode_dir: episode 目录
        output_dir: 输出视频目录
        fps: 帧率
    
    Returns:
        处理结果信息
    """
    # 获取 subset 名称和 episode ID
    subset = episode_dir.parent.name
    episode_id = episode_dir.name
    
    # 输出路径
    output_path = output_dir / subset / f"{episode_id}.mp4"
    
    # 转换
    success = frames_to_video(episode_dir, output_path, fps=fps)
    
    return {
        "episode_dir": str(episode_dir),
        "output_path": str(output_path),
        "success": success
    }


def main():
    parser = argparse.ArgumentParser(description="Convert keyframes to videos")
    parser.add_argument(
        "--input_dir", 
        type=str, 
        default="data_processed/keyframes",
        help="Directory containing keyframes"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data_processed/videos",
        help="Output directory for videos"
    )
    parser.add_argument(
        "--fps", 
        type=int, 
        default=10,
        help="Video frame rate (default: 10)"
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
    output_dir = project_root / args.output_dir
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"FPS: {args.fps}")
    
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
    
    print(f"Found {len(episode_dirs)} episodes to convert")
    
    if not episode_dirs:
        print("No episodes found!")
        sys.exit(1)
    
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 并行处理
    success_count = 0
    fail_count = 0
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(process_episode, ep_dir, output_dir, args.fps): ep_dir 
            for ep_dir in episode_dirs
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Converting"):
            ep_dir = futures[future]
            try:
                result = future.result()
                if result['success']:
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                print(f"Error processing {ep_dir}: {e}")
                fail_count += 1
    
    print(f"\nDone!")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    main()




