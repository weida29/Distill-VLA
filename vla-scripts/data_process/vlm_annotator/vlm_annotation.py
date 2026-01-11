"""
VLM 自动标注脚本 - 对机器人操作视频进行物体检测和任务描述

基于 internm1_data_annotation.py 适配到 VLA-Adapter 项目

输入: data_processed/vlm_input.jsonl (由 prepare_vlm_input.py 生成)
输出: data_processed/annotations/bbox_results.jsonl

支持两种任务:
1. tracking: 识别被操作物体并返回首帧 BBox
2. task: 生成任务描述

使用示例:
    # 启动 VLM 服务 (vLLM)
    vllm serve Qwen3-VL-30B-A3B-Instruct --port 18000 -tp 8

    # 运行标注
    python vlm_annotation.py \
        --input_file data_processed/vlm_input.jsonl \
        --output_file data_processed/annotations/bbox_results.jsonl \
        --task tracking \
        --frame_num 6 \
        --max_workers 32
"""

import json
import argparse
import os
import sys
import time
import math
import numpy as np
import markdown
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed


def draw_bbox(image: Image.Image, bbox: list) -> Image.Image:
    """在图像上绘制边界框"""
    draw = ImageDraw.Draw(image)
    draw.rectangle(bbox, outline='red', width=4)
    return image


def create_image_grid_pil(pil_images: list, num_columns: int = 8) -> Image.Image:
    """将多张图片拼接成网格"""
    num_rows = math.ceil(len(pil_images) / num_columns)
    img_width, img_height = pil_images[0].size
    grid_width = num_columns * img_width
    grid_height = num_rows * img_height
    grid_image = Image.new('RGB', (grid_width, grid_height))

    for idx, image in enumerate(pil_images):
        row_idx = idx // num_columns
        col_idx = idx % num_columns
        position = (col_idx * img_width, row_idx * img_height)
        grid_image.paste(image, position)

    return grid_image


def parse_json(response: str) -> dict:
    """
    从 VLM 响应中解析 JSON
    支持多种格式: 纯 JSON、Markdown 代码块、混合文本
    """
    try:
        return json.loads(response)
    except:
        try:
            # 尝试从 Markdown 代码块中提取
            html = markdown.markdown(response, extensions=['fenced_code'])
            soup = BeautifulSoup(html, 'html.parser')
            code_block = soup.find('code')
            if code_block:
                json_text = code_block.text
                return json.loads(json_text)
            else:
                # 暴力查找 JSON 边界
                start = response.find('{')
                end = response.rfind('}') + 1
                if start != -1 and end > start:
                    return json.loads(response[start:end])
                raise ValueError("No JSON found in response")
        except Exception as e:
            raise e


# ============== Prompts ==============

# 带任务描述的物体追踪 prompt（推荐使用）
PROMPT_ONE_OBJECT_TRACKING_WITH_TASK = '''\
These images represent frames from a robotic arm manipulation video.

Task Description: {task_description}

Based on the task description above and the visual sequence, identify the **target object being manipulated** (the object that the robot arm picks up or interacts with).

Requirements:
1. Identify the target object's name based on the task description.
2. Detect its bounding box in the **first frame**.

Output Requirement:
Return the result strictly in JSON format as follows:
{{
  "label": "English name of the object",
  "bbox_2d": [xmin, ymin, xmax, ymax]
}}
'''

# 无任务描述的物体追踪 prompt（备用）
PROMPT_ONE_OBJECT_TRACKING = '''\
These images represent frames from a robotic arm manipulation video. Analyze the visual sequence to identify the target object being manipulated.

Task:
1. Identify the object.
2. Detect its bounding box in the **first frame**.

Output Requirement:
Return the result strictly in JSON format as follows:
{
  "label": "English name of the object",
  "bbox_2d": [xmin, ymin, xmax, ymax]
}
'''

PROMPT_ONE_OBJECT_TASK = '''\
These images represent frames from a robotic arm manipulation video. Analyze the visual sequence to identify the target object being manipulated.
Describe the task instruction of the video. Remember to describe the full trajectory of the object.
For example, "pick up the red box on the table and put it into the blue basket".

Output Requirement:
Return the result strictly in JSON format as follows:
{
  "task": "English description of the task"
}
'''


def inference(
    model: str, 
    base_url: str, 
    api_key: str, 
    prompt: str, 
    base64_images: list
) -> str:
    """
    调用 VLM API 进行推理
    
    Args:
        model: 模型名称
        base_url: API 地址
        api_key: API 密钥
        prompt: 提示词
        base64_images: base64 编码的图片列表
    
    Returns:
        VLM 响应内容
    """
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )

    # 构建多图输入
    content = [{
        "type": "image_url",
        "image_url": {
            "url": image if image.startswith("data:image") else f"data:image/jpeg;base64,{image}"
        }
    } for image in base64_images]

    content.append(
        {"type": "text", "text": prompt}
    )

    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]
    
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2
    )
    return completion.choices[0].message.content


def process_line(
    line: str, 
    args: argparse.Namespace, 
    base_url: str, 
    api_key: str, 
    model_name: str
) -> str:
    """
    处理单行 JSONL 数据
    
    Args:
        line: JSONL 行
        args: 命令行参数
        base_url: API 地址
        api_key: API 密钥
        model_name: 模型名称
    
    Returns:
        处理结果 JSON 字符串，失败返回 None
    """
    try:
        data = json.loads(line)
        
        if 'frames_base64' not in data:
            return None

        all_frames = data['frames_base64']
        infer_frames_num = args.frame_num
        
        if len(all_frames) == 0:
            return None
        
        # 均匀采样帧
        if len(all_frames) > infer_frames_num:
            infer_frames_indices = np.linspace(0, len(all_frames) - 1, infer_frames_num, dtype=int)
            infer_frames = [all_frames[int(i)] for i in infer_frames_indices]
        else:
            infer_frames = all_frames

        res_json = None
        task_description = data.get('prompt', '')  # 获取原始任务描述
        
        for retry_time in range(3):
            try:
                # 选择 prompt
                if args.task == 'tracking':
                    # 如果有原始任务描述，使用带任务描述的 prompt
                    if task_description:
                        prompt = PROMPT_ONE_OBJECT_TRACKING_WITH_TASK.format(
                            task_description=task_description
                        )
                    else:
                        prompt = PROMPT_ONE_OBJECT_TRACKING
                elif args.task == 'task':
                    prompt = PROMPT_ONE_OBJECT_TASK
                else:
                    raise ValueError(f"Invalid task: {args.task}")
                
                res = inference(
                    model=model_name,
                    base_url=base_url,
                    api_key=api_key,
                    prompt=prompt,
                    base64_images=infer_frames
                )
                
                res_json = parse_json(res)

                # 验证返回字段
                if args.task == 'tracking':
                    if 'label' in res_json and 'bbox_2d' in res_json:
                        break
                    else:
                        raise ValueError("Missing keys: label or bbox_2d")
                elif args.task == 'task':
                    if 'task' in res_json:
                        break
                    else:
                        raise ValueError("Missing key: task")

            except Exception as e:
                if retry_time == 2:
                    print(f"[ERROR] Failed after 3 retries: {e}")
                time.sleep(0.5)
        
        if res_json:
            save_data = {
                'video_path': data['video_path'],
                'prompt': data.get('prompt', ''),  # 保留原始 prompt
                'result': res_json
            }
            return json.dumps(save_data, ensure_ascii=False)
        else:
            return None

    except Exception as e:
        print(f"[ERROR] Processing failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="VLM annotation for robot manipulation videos")
    parser.add_argument(
        "--input_file", 
        type=str, 
        default="data_processed/vlm_input.jsonl",
        help="Input JSONL file path"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="data_processed/annotations/bbox_results.jsonl",
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--task", 
        type=str, 
        choices=['tracking', 'task'], 
        default='tracking',
        help="Task type: tracking (bbox detection) or task (description)"
    )
    parser.add_argument(
        "--frame_num", 
        type=int, 
        default=6,
        help="Number of frames to use for inference"
    )
    parser.add_argument(
        "--max_workers", 
        type=int, 
        default=8,
        help="Number of parallel workers"
    )
    args = parser.parse_args()

    # 找到项目根目录
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent.parent
    
    input_file = project_root / args.input_file
    output_file = project_root / args.output_file
    
    # 从环境变量获取 API 配置
    base_url = os.getenv("OPENAI_API_BASE_URL", "http://127.0.0.1:18000/v1")
    api_key = os.getenv("OPENAI_API_KEY", "sk-placeholder")
    model_name = os.getenv("OPENAI_MODEL", "Qwen3-VL-30B-A3B-Instruct")

    print(f"=" * 60)
    print(f"VLM Annotation Script")
    print(f"=" * 60)
    print(f"API URL: {base_url}")
    print(f"Model: {model_name}")
    print(f"Task: {args.task}")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Frame num: {args.frame_num}")
    print(f"Workers: {args.max_workers}")
    print(f"=" * 60)

    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        print("Please run prepare_vlm_input.py first")
        sys.exit(1)

    # 读取输入数据
    with open(input_file, "r", encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Loaded {len(lines)} items")

    # 确保输出目录存在
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # 并行处理
    with open(output_file, "w", encoding='utf-8') as f_out:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = []
            for line in lines:
                future = executor.submit(
                    process_line, line, args, base_url, api_key, model_name
                )
                futures.append(future)

            success_count = 0
            fail_count = 0
            
            for future in tqdm(as_completed(futures), total=len(lines), desc="Processing"):
                result = future.result()
                if result:
                    f_out.write(result + '\n')
                    f_out.flush()
                    success_count += 1
                else:
                    fail_count += 1

    print(f"\nDone!")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Results saved to: {output_file}")


if __name__ == "__main__":
    main()


"""
使用示例:

# 1. 启动 VLM 服务 (使用 vLLM)
vllm serve Qwen3-VL-30B-A3B-Instruct \\
    --port 18000 \\
    -tp 8 \\
    --dtype half \\
    --max-model-len 65536

# 2. 运行物体检测标注
OPENAI_API_BASE_URL=http://127.0.0.1:18000/v1 \\
OPENAI_MODEL=Qwen3-VL-30B-A3B-Instruct \\
python vlm_annotation.py \\
    --input_file data_processed/vlm_input.jsonl \\
    --output_file data_processed/annotations/bbox_results.jsonl \\
    --task tracking \\
    --frame_num 6 \\
    --max_workers 32

# 3. 运行任务描述生成
OPENAI_API_BASE_URL=http://127.0.0.1:18000/v1 \\
OPENAI_MODEL=Qwen3-VL-30B-A3B-Instruct \\
python vlm_annotation.py \\
    --input_file data_processed/vlm_input.jsonl \\
    --output_file data_processed/annotations/task_results.jsonl \\
    --task task \\
    --frame_num 8 \\
    --max_workers 32
"""

