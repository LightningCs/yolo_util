import cv2
from PIL import Image
from PIL.ImageFile import ImageFile as PILImageFile
from typing import List
import tempfile
import os
import io


def extract_frames_from_bytes(video_bytes: bytes, interval: int = 15) -> List[PILImageFile]:
    """
    从视频字节流抽帧
    
    参数:
        video_bytes: 视频文件的字节数据
        interval: 抽帧间隔
        
    返回:
        List[PILImageFile]:             PILImageFile 对象列表
    """
    frames = []
    
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
        tmp_file.write(video_bytes)
        tmp_path = tmp_file.name
    
    try:
        # 使用临时文件路径读取视频
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise ValueError("无法打开视频流")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % interval == 0:
                # 转换为 PILImageFile
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                img_buffer = io.BytesIO()
                pil_image.save(img_buffer, format="JPEG", quality=95)
                img_buffer.seek(0)
                image_file = Image.open(img_buffer)
                image_file._buffer = img_buffer
                frames.append(image_file)
            
            frame_count += 1
        
        cap.release()
    finally:
        # 清理临时文件
        try:
            os.unlink(tmp_path)
        except:
            pass
    
    return frames

