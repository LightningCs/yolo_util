import os

from PIL import Image

def get_great_len(prefix: str) -> int:
    """
    获取图片的宽和高的最大值

    获取图片的宽和高的最大值

    Args:
        prefix (str): 图片所在路径

    Return:
        great_len (int): 图片长度的最大值
    """
    great_len = 0

    # 遍历所有图片路径
    for image_path in os.listdir(prefix):
        image = Image.open(f'{prefix}/{image_path}')
        width, height = image.size

        great_len = max(great_len, width, height)

    return great_len
