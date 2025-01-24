import cv2
import os
import math
import numpy as np
from PIL import Image, ImageDraw

def extract_frames(video_path, output_dir, frames_per_second=10):
    """
    从视频中提取图像帧，每秒提取指定数量的帧。

    Args:
        video_path (str): 视频文件的路径。
        output_dir (str): 图像保存的目录。
        frames_per_second (int): 每秒提取的帧数。
    """

    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return

    # 获取视频帧率和总帧数
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 计算提取帧的间隔
    frame_interval = fps / frames_per_second
    frame_count = 0
    saved_frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 计算当前帧是否需要保存
        if math.isclose(frame_count % frame_interval, 0, abs_tol=1e-5):
            # 生成文件名，格式为 "000000.jpg", "000001.jpg" ...
            filename = os.path.join(output_dir, f"{frame_count:06d}.jpg")

            # 保存帧
            print(f"Saved frame {frame_count} to {filename}")
            cv2.imwrite(filename, frame)
            saved_frame_count += 1


        frame_count += 1

    # 释放视频资源
    cap.release()

    print(f"Extracted {saved_frame_count} frames from {video_path} and saved to {output_dir}")


def crop_circle_from_image(image_path, center, radius, output_path):
    """
    从图像中指定位置裁剪一个圆形图像，并保存为 PNG 格式。

    :param image_path: 输入图像文件路径
    :param center: 圆心坐标 (x, y)
    :param radius: 圆的半径
    :param output_path: 输出的图像文件路径
    """
    # 打开图像
    image = Image.open(image_path).convert("RGBA")
    
    # 创建一个与图像大小相同的透明背景图
    circle_mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(circle_mask)
    
    # 绘制圆形掩膜，填充白色 (255)
    draw.ellipse((center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius), fill=255)
    
    # 应用圆形掩膜
    result = Image.new("RGBA", image.size)
    result.paste(image, mask=circle_mask)
    
    # 裁剪圆形区域
    result = result.crop(result.getbbox())
    
    # 保存结果图像
    result.save(output_path, format="PNG")

if __name__ == '__main__':
    # 设置视频路径和输出目录
    from Violet.Violet_base import list_all_files
    all_video_path = list_all_files(r"D:\Program Files\JiJiDown\Download\gs")

    star = 62

    his = 48

    for id, video_path in enumerate(all_video_path):

        if id + his + 1 < star: continue

        output_dir = "E:/Genshin_Impact/" + os.path.basename(video_path)[:-4] + "/"       # 替换为你想要保存图像的目录
        #设置每秒提取帧数
        frames_per_second = 0.5

        # 执行提取
        extract_frames(video_path, output_dir, frames_per_second)

        # 设置图像路径
        
        
        image_path = list_all_files(output_dir)
        # 设置输出图像路径
        output_path = "E:/Genshin_Impact_circle/" + os.path.basename(video_path)[:-4] + "/"
        # 设置圆形区域的中心点
        center = (430, 95)  # (x, y) 坐标
        # 设置圆形半径
        radius = 36

        # 执行裁剪
        if os.path.exists(output_path) == False:
            os.mkdir(output_path)

        for p in image_path:
            crop_circle_from_image(p, center, radius, output_path + os.path.basename(p))