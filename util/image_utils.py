import math
import os
import cv2
from tqdm import tqdm

from PIL import Image


def get_great_len(prefix: str) -> int:
    """
    获取图片的宽和高的最大值

    获取图片的宽和高的最大值

    Args:
        prefix (str): 图片所在路径

    Returns:
        great_len (int): 图片长度的最大值
    """
    great_len = 0

    # 遍历所有图片路径
    for image_path in os.listdir(prefix):
        image = Image.open(f'{prefix}/{image_path}')
        width, height = image.size

        great_len = max(great_len, width, height)

    return great_len


def img_tiling_by_labels(img: str,
                         label: str,
                         imgsz=640):
    """
    图片平铺(按标签裁剪)

    图片平铺(按标签裁剪)

    Args:
        img (str): 图片路径
        label (str): 图片标签
        imgsz (int): 裁剪尺寸(imgsz * imgsz)
    """
    # 获取图片的宽和高
    image = cv2.imread(img)
    image_copy = image.copy()

    # 宽和高
    ab_h = image.shape[0]
    ab_w = image.shape[1]

    w = math.ceil(ab_w / imgsz)
    h = math.ceil(ab_h / imgsz)

    # 读取所有行
    lines = open(label, 'r', encoding='utf-8').readlines()

    count = 1

    # 生成标签文件并裁出图片
    for line in lines:
        data = line.strip().split(' ')
        # 归一化下的四个点的位置
        nor_min_x = min(float(data[1]), float(data[3]), float(data[5]), float(data[7]))
        nor_max_x = max(float(data[1]), float(data[3]), float(data[5]), float(data[7]))
        nor_min_y = min(float(data[2]), float(data[4]), float(data[6]), float(data[8]))
        nor_max_y = max(float(data[2]), float(data[4]), float(data[6]), float(data[8]))
        # 归一化下的标签的长宽
        nor_label_w = nor_max_x - nor_min_x
        nor_label_h = nor_max_y - nor_min_y
        # 标签的宽和高
        ab_label_w = math.ceil(nor_label_w * ab_w)
        ab_label_h = math.ceil(nor_label_h * ab_h)
        # 标签的左上角和右下角(绝对)
        ab_min_x = math.ceil(nor_min_x * ab_w)
        ab_min_y = math.ceil(nor_min_y * ab_h)
        ab_max_x = ab_min_x + ab_label_w
        ab_max_y = ab_min_y + ab_label_h

        # 分割图片
        # 遍历行
        for j in range(h):
            ab_img_min_y = j * imgsz
            ab_img_max_y = min((j + 1) * imgsz, ab_h)
            # 遍历列
            for i in range(w):
                # 当前图片的四个点的x,y坐标(绝对)
                ab_img_min_x = i * imgsz
                ab_img_max_x = min((i + 1) * imgsz, ab_w)

                # 预先判断当前图片是否与该标签交叉
                if not (ab_img_min_x > ab_max_x or ab_img_max_y < ab_min_y or ab_img_max_x < ab_min_x or ab_img_min_y > ab_max_y):
                    # 打开标签文件
                    f = open(f'tiles/labels/{img[img.rindex("/") + 1:].split(".")[0]}_{count}.txt', 'a',
                             encoding='utf-8')

                    tmp = [data[0]]
                    tmp.append(str(((max(ab_img_min_x, ab_min_x) - i * imgsz) * 1.0) / imgsz))
                    tmp.append(str(((max(ab_img_min_y, ab_min_y) - j * imgsz) * 1.0) / imgsz))
                    tmp.append(str(((min(ab_img_max_x, ab_max_x) - i * imgsz) * 1.0) / imgsz))
                    tmp.append(str(((max(ab_img_min_y, ab_min_y) - j * imgsz) * 1.0) / imgsz))
                    tmp.append(str(((max(ab_img_min_x, ab_min_x) - i * imgsz) * 1.0) / imgsz))
                    tmp.append(str(((min(ab_img_max_y, ab_max_y) - j * imgsz) * 1.0) / imgsz))
                    tmp.append(str(((min(ab_img_max_x, ab_max_x) - i * imgsz) * 1.0) / imgsz))
                    tmp.append(str(((min(ab_img_max_y, ab_max_y) - j * imgsz) * 1.0) / imgsz))
                    # 写入文件
                    f.write(' '.join(tmp) + '\n')

                    # 裁剪区域
                    tiles = image_copy[ab_img_min_y: ab_img_max_y, ab_img_min_x: ab_img_max_x]
                    cv2.imwrite(f'tiles/images/{img[img.rindex("/") + 1:].split(".")[0]}_{count}.png', tiles)

                    # 数量加1
                    count += 1


def img_tiling_by_area(img: str,
                       label: str,
                       imgsz=640):
    """
    图片平铺(按区域裁剪)

    图片平铺(按区域裁剪)

    Args:
        img (str): 图片路径
        label (str): 图片标签
        imgsz (int): 裁剪尺寸(imgsz * imgsz)
    """
    # 获取图片的宽和高
    image = cv2.imread(img)
    image_copy = image.copy()

    # 宽和高
    ab_h = image.shape[0]
    ab_w = image.shape[1]

    w = math.ceil(ab_w / imgsz)
    h = math.ceil(ab_h / imgsz)

    # 读取所有行
    lines = open(label, 'r', encoding='utf-8').readlines()

    count = 1

    # 分割图片
    # 遍历行
    for j in range(h):
        ab_img_min_y = j * imgsz
        ab_img_max_y = min((j + 1) * imgsz, ab_h)
        # 遍历列
        for i in range(w):
            # 当前图片的四个点的x,y坐标(绝对)
            ab_img_min_x = i * imgsz
            ab_img_max_x = min((i + 1) * imgsz, ab_w)

            # 是否裁剪区域
            flag = False

            # 遍历所有标签
            for line in lines:
                data = line.strip().split(' ')
                # 归一化下的四个点的位置
                nor_min_x = min(float(data[1]), float(data[3]), float(data[5]), float(data[7]))
                nor_max_x = max(float(data[1]), float(data[3]), float(data[5]), float(data[7]))
                nor_min_y = min(float(data[2]), float(data[4]), float(data[6]), float(data[8]))
                nor_max_y = max(float(data[2]), float(data[4]), float(data[6]), float(data[8]))
                # 归一化下的标签的长宽
                nor_label_w = nor_max_x - nor_min_x
                nor_label_h = nor_max_y - nor_min_y
                # 标签的宽和高
                ab_label_w = math.ceil(nor_label_w * ab_w)
                ab_label_h = math.ceil(nor_label_h * ab_h)
                # 标签的左上角和右下角(绝对)
                ab_min_x = math.ceil(nor_min_x * ab_w)
                ab_min_y = math.ceil(nor_min_y * ab_h)
                ab_max_x = ab_min_x + ab_label_w
                ab_max_y = ab_min_y + ab_label_h

                # 判断该区域是否包含标签
                if not (ab_img_min_x > ab_max_x or ab_img_max_y < ab_min_y or ab_img_max_x < ab_min_x or ab_img_min_y > ab_max_y):
                    # 裁剪区域
                    flag = True

                    # 打开标签文件
                    f = open(f'tiles/labels/{img[img.rindex("/") + 1:].split(".")[0]}_{count}.txt', 'a',
                             encoding='utf-8')

                    tmp = [data[0]]
                    tmp.append(str(((max(ab_img_min_x, ab_min_x) - i * imgsz) * 1.0) / imgsz))
                    tmp.append(str(((max(ab_img_min_y, ab_min_y) - j * imgsz) * 1.0) / imgsz))
                    tmp.append(str(((min(ab_img_max_x, ab_max_x) - i * imgsz) * 1.0) / imgsz))
                    tmp.append(str(((max(ab_img_min_y, ab_min_y) - j * imgsz) * 1.0) / imgsz))
                    tmp.append(str(((max(ab_img_min_x, ab_min_x) - i * imgsz) * 1.0) / imgsz))
                    tmp.append(str(((min(ab_img_max_y, ab_max_y) - j * imgsz) * 1.0) / imgsz))
                    tmp.append(str(((min(ab_img_max_x, ab_max_x) - i * imgsz) * 1.0) / imgsz))
                    tmp.append(str(((min(ab_img_max_y, ab_max_y) - j * imgsz) * 1.0) / imgsz))
                    # 写入文件
                    f.write(' '.join(tmp) + '\n')

            if flag:
                # 裁剪区域
                tiles = image_copy[ab_img_min_y: ab_img_max_y, ab_img_min_x: ab_img_max_x]
                cv2.imwrite(f'tiles/images/{img[img.rindex("/") + 1:].split(".")[0]}_{count}.png', tiles)
                # 数量加1
                count += 1


def clear_img(path: str):
    """
    清洗图片

    清洗图片

    Args:
        path (str):             路径
    """

    # 判断路径是否存在
    if not os.path.exists(path):
        raise Exception('路径不存在')

    print('开始清洗图片...')

    # 图片存储路径
    images_path = os.path.join(path, 'images')

    # 标签存储路径
    labels_path = os.path.join(path, 'labels')

    # 去除空标签图片
    for label in tqdm(os.listdir(labels_path)):
        cur_pos = os.path.join(labels_path, label)

        # 不是是文件并且是文本文件
        if not (os.path.isfile(cur_pos) and label.endswith('.txt')):
            continue

        with open(cur_pos, 'r', encoding='utf-8') as f:
            # 若标签长度不为 0，不需要删除
            if len(f.readlines()) != 0:
                continue

        items = label.strip().split('.')

        # 删除图片
        if os.path.exists(os.path.join(images_path, items[0] + '.png')):
            os.remove(os.path.join(images_path, items[0] + '.png'))
        elif os.path.exists(os.path.join(images_path, items[0] + '.jpg')):
            os.remove(os.path.join(images_path, items[0] + '.jpg'))
        else:
            raise Exception(f'不存在图片: {items[0]}')

        # 删除标签
        os.remove(os.path.join(labels_path, label))

    print('图片清洗完成...')

