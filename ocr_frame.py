from CVToolfuben import CVTool
import cv2
import os

# 配置 Tesseract 的路径（如果需要）
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def check_text_in_region(image, region, target_text, cvt:CVTool):
    """
    检查图像指定区域是否包含目标中文文字。

    参数:
    - image_path: 图像路径
    - region: 要检查的区域 (x1, y1, x2, y2)
    - target_text: 目标中文文字

    返回:
    - 是否包含目标文字 (True/False)
    """
    # 加载图像
    #image = cv2.imread(image_path)

    # 提取特定区域
    x1, y1, x2, y2 = region
    cropped = image[y1:y2, x1:x2]
    
    # 转灰度图和二值化（可选）
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 使用 Tesseract OCR 识别文字
    custom_config = r'--oem 3 --psm 6'  # OCR 模式配置
    text = cvt.ocr_all(binary)
    if text != '':
        print(text)
    # 检查文字是否包含目标内容
    for t in target_text:
        if t in text:
            return True
    return False
    #return target_text in text

def process_video(video_path, output_dir, cvt:CVTool, scale=1.0, interval=1.0, region = (50, 50, 200, 200)):
    # 创建保存帧的文件夹
    os.makedirs(output_dir+'/story', exist_ok=True)
    os.makedirs(output_dir+'/other', exist_ok=True)
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件！")
        return

    # 获取视频的帧率和总帧数
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fps = 60
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    print(f"视频时长: {duration:.2f}秒, 帧率: {fps}帧/秒")

    frame_interval = int(fps * interval)  # 计算帧间隔
    frame_index = 0  # 当前帧编号
    save_index = 0   # 保存的图片编号

    
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 判断是否是采样间隔的帧
        
        if frame_index % frame_interval == 0:
            print(f'处理帧: {frame_index}')
            if check_text_in_region(image=frame, target_text=['自','动','菜','单','跳','过'], region=region, cvt=cvt):
                cv2.imwrite(os.path.join(output_dir + '/story', f"{frame_index:06d}.jpg"), frame)
                print('#'*100)
                print(f"保存到{output_dir}'/story/'{frame_index:06d}.jpg")
                #print('#'*100)
            else:
                cv2.imwrite(os.path.join(output_dir + '/other', f"{frame_index:06d}.jpg"), frame)
                #print(f"保存到{output_dir}'/story/'{frame_index:06d}.jpg")
            pass
        frame_index += 1

    cap.release()
    #return true_1/true_2, true_1/all_test, (all_test - true_2)/all_test
    print("处理完成！")

def list_all_files(directory):
        path_list = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                path_list.append(os.path.join(root, file))
        return path_list

def save_video_frame(video_path, frame_time, output_path):
    """
    保存视频中指定描述时间的一帧。

    参数:
    - video_path: 视频文件路径
    - frame_time: 描述时间（以秒为单位，例如 5.5 表示第 5.5 秒）
    - output_path: 保存的图像文件路径
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频文件: {video_path}")
    
    # 获取视频帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int(frame_time * fps)

    # 跳转到指定帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()

    if not ret:
        raise ValueError(f"无法在时间 {frame_time} 秒提取帧")

    # 保存图像
    cv2.imwrite(output_path, frame)
    print(f"帧已保存到: {output_path}")

    # 释放资源
    cap.release()
    

if __name__ == "__main__":
    # 示例

    all_video_path = list_all_files("E:/zzz")
    
    start = 102
    region = (1725, 112, 1800, 150)  # 指定区域 (x1, y1, x2, y2)
    #save_video_frame(all_video_path[56], 606, './frame.jpg')

    #img = cv2.imread("frame.jpg")
    #x1, y1, x2, y2 = region
    #cropped = img[y1:y2, x1:x2]
    #target_text = "自动"  # 目标中文文字
    cvt = CVTool()
    output_dir = "E:/zzz_frames/"
    for index, path in enumerate(all_video_path):
        index = index + 1
        if index < start: continue
        process_video(video_path=path, output_dir=output_dir+str(index).zfill(3), cvt=cvt, region=region, interval=1.015)
    

    #result = check_text_in_region(img, region, target_text, cvt)
    #print("包含目标文字:", result)
