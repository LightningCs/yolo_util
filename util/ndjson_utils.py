import os
import json
import shutil

def init_file(source_path: str,
              target_path: str):
    """
    初始化目录

    复制图片到目标目录

    Args:
        source_path (str):          图片源路径
        target_path (str):          存储目标路径
    """
    # 清空 train 和 labels
    shutil.rmtree(f'{target_path}/train')
    shutil.rmtree(f'{target_path}/labels')

    shutil.copytree(source_path, f'{target_path}/train')

    # 创建 train 和 labels 目录
    os.makedirs(f'{target_path}/train/train')
    os.makedirs(f'{target_path}/train/val')


class NDJSON:
    """
    NDJSON类
    """
    def __init__(self,
                 type: str,
                 file: str,
                 url: str,
                 width: int,
                 height: int,
                 split: str):
        self.type = type
        self.file = file
        self.url = url
        self.width = width
        self.height = height
        self.split = split
        self.annotations = {'segmentation': [], 'bbox': []}


def gen_file_name(id: int):
    """
    生成序号文件名

    根据传入id生成序号文件名

    Args:
        id (int):                   图片id
    """
    return f'{str(1000000 + id)[1:]}'


def gen_ndjson_dataset(images_source_path: str,
                       json_path: str,
                       target_path: str,
                       train_rate=0.8):
    """
    生成数据集

    根据图片和json文件生成数据集并存储

    Args:
        images_source_path (str):   图片源路径
        json_path (str):            json文件(数据集注释和标签)存放路径
        target_path (str):          数据集存放路径
        train_rate (float):         训练集占比(默认0.8)
    """
    # 加载 json 文件
    result = open(json_path, 'r')
    obj = json.load(result)

    # 数据集数量
    num = len(obj['train'])
    # 训练集数量
    train_num = int(num * train_rate)

    # 初始化文件
    init_file(images_source_path, target_path)

    # 文件名称与 ndjson 对象的映射
    train_map = {}
    val_map = {}

    # 遍历所有图片
    for image in obj['train']:
        width = image['width']
        height = image['height']
        id = image['id']
        file_name = image['file_name'][image['file_name'].rindex('\\') + 1:]

        # 生成新的文件名
        new_file_name = gen_file_name(id)

        # 生成 ndjson 对象
        if train_num > 0:
            # 重命名文件
            os.rename(f'{target_path}/train/{file_name}',
                      f'{target_path}/train/train/{new_file_name}.png')
            train_map[new_file_name] = NDJSON('image', f'{new_file_name}.png',
                                              f'{target_path}/train/train/{new_file_name}.png', width, height,
                                              'train')
            # 数量减一
            train_num -= 1
        else:
            # 重命名文件
            os.rename(f'{target_path}/train/{file_name}',
                      f'{target_path}/train/val/{new_file_name}.png')
            val_map[new_file_name] = NDJSON('image', f'{new_file_name}.png',
                                            f'{target_path}/train/val/{new_file_name}.png', width, height,
                                            'val')

    # 遍历所有标注
    for annotation in obj['annotations']:
        # 图片id
        image_id = annotation['image_id']
        # 类别id
        category_id = annotation['category_id']
        # 分割
        segmentation = annotation['segmentation'][0]
        segmentation.insert(0, category_id)
        # 检测
        bbox = annotation['bbox']
        bbox.insert(0, category_id)

        # 完善对应ndjson对象
        ndjson = train_map[gen_file_name(image_id)] if train_map.get(gen_file_name(image_id)) else val_map[gen_file_name(image_id)]
        ndjson.annotations['segmentation'].append(segmentation)
        ndjson.annotations['bbox'].append(bbox)

    # 将对象保存到文件中
    with open(f'{target_path}/train.ndjson', 'a', encoding='utf-8') as f:
        # 写入文件头
        json.dump({
            "type": "dataset",
            "task": "detect",
            "name": "train",
            "description": "COCO NDJSON train dataset",
            "url": "",
            "class_names": {0: "人井", 1: "光缆说明", 2: "剖面图", 3: "土方挖地字样", 4: "城镇街区建筑", 5: "平面图", 6: "建筑外墙", 7: "新增杆标识",
                            8: "材料表", 9: "水域", 10: "线路路由", 11: "路线路由", 12: "路线路由说明", 13: "道路区域"},
            "bytes": 426342,
            "version": 0.1,
            "created_at": "2025-10-10 15:15",
            "updated_at": "2025-10-10 15:15"
        }, f, default=lambda x: x.__dict__)
        for item in train_map.values():
            json.dump(item, f, default=lambda x: x.__dict__)
            f.write('\n')

    with open(f'{target_path}/val.ndjson', 'a', encoding='utf-8') as f:
        # 写入文件头
        json.dump({
            "type": "dataset",
            "task": "detect",
            "name": "val",
            "description": "COCO NDJSON val dataset",
            "url": "",
            "class_names": {0: "人井", 1: "光缆说明", 2: "剖面图", 3: "土方挖地字样", 4: "城镇街区建筑", 5: "平面图", 6: "建筑外墙", 7: "新增杆标识",
                            8: "材料表", 9: "水域", 10: "线路路由", 11: "路线路由", 12: "路线路由说明", 13: "道路区域"},
            "bytes": 426342,
            "version": 0.1,
            "created_at": "2025-10-10 15:15",
            "updated_at": "2025-10-10 15:15"
        }, f, default=lambda x: x.__dict__)
        for item in val_map.values():
            json.dump(item, f, default=lambda x: x.__dict__)
            f.write('\n')
