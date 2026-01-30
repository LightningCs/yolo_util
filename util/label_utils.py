import os
import numpy as np

from tqdm import tqdm


def statistic_labels_num(classes_path: str,
                         labels_path: str) -> dict:
    """
    统计各类标签数量

    统计各类标签数量

    Args:
        classes_path (str):     分类文件路径
        labels_path (str):      标签文件路径

    Returns:
        res (dict):             各类标签统计结果
    """

    # 判断分类路径是否存在
    if not os.path.exists(classes_path):
        raise Exception('分类路径不存在')

    # 判断标签路径是否存在
    if not os.path.exists(labels_path):
        raise Exception('标签路径不存在')

    # 分类映射
    classes_map = {}
    with open(classes_path, 'r', encoding='utf-8') as f:
        categories = f.readlines()
        for i, category in enumerate(categories):
            classes_map[i] = category.strip()

    res = {}

    for file in tqdm(os.listdir(labels_path)):
        cur_pos = os.path.join(labels_path, file)
        # 是文件且以txt为结尾
        if os.path.isfile(cur_pos) and file.endswith('.txt'):
            with open(cur_pos, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip()
                    # 空行
                    if line == '':
                        continue

                    line = line.split(' ')
                    label = classes_map[int(line[0])]
                    res[label] = res.get(label, 0) + 1

    return res


def statistic_labels_num_by_images(classes_path: str,
                                   labels_path: str) -> dict and set:
    """
    统计各张图片的标签数量及各种标签的个数
    
    统计各张图片的标签数量及各种标签的个数

    Args:
        classes_path (str):     分类文件路径
        labels_path (str):      标签文件路径

    Returns:
        res (dict):             各张图片的各类表签统计结果及各种标签的总数
        classes_set (set):      所有标签
    """

    # 判断分类路径是否存在
    if not os.path.exists(classes_path):
        raise Exception('分类路径不存在')

    # 判断标签路径是否存在
    if not os.path.exists(labels_path):
        raise Exception('标签路径不存在')

    # 分类映射
    classes_map = {}
    with open(classes_path, 'r', encoding='utf-8') as f:
        categories = f.readlines()
        for i, category in enumerate(categories):
            classes_map[i] = category.strip()

    res = {"result": []}

    for file in tqdm(os.listdir(labels_path)):
        file_name = file.split('.')[0]

        cur_pos = os.path.join(labels_path, file)
        # 是文件且以txt为结尾
        if os.path.isfile(cur_pos) and file.endswith('.txt'):
            entity = {"image": file_name}

            with open(cur_pos, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip()
                    # 空行
                    if line == '':
                        continue

                    line = line.split(' ')
                    label = classes_map[int(line[0])]
                    # 统计各张图片各种标签数量
                    entity[label] = entity.get(label, 0) + 1
                    # 统计标签总数
                    res[label] = res.get(label, 0) + 1
                    res["total"] = res.get("total", 0) + 1

            res["result"].append(entity)
            
    classes_set = set(classes_map.values())

    return res, classes_set


def balance_dataset_labels(classes_path: str,
                                 dataset_path: str,
                                 type: str):
    """
    
    均衡数据集标签
    
    均衡数据集标签
    
    Todo:
        - 待完善

    Args:
        classes_path (str):     分类文件路径
        dataset_path (str):     数据集路径
        type (str):             数据集类型
    """
    if not os.path.exists(classes_path):
        raise Exception('分类路径不存在')
    if not os.path.exists(os.path.join(dataset_path)):
        raise Exception('数据集路径不存在')

    # 统计标签数量 & 图片级信息
    statistic_res, classes_set = statistic_labels_num_by_images(classes_path, os.path.join(dataset_path, 'labels', type))

    image_info_list = statistic_res["result"]
    classes_set = list(classes_set)

    # === 1. 计算中位数阈值 ===
    labels_num = np.array([statistic_res[c] for c in classes_set])
    median = np.median(labels_num)
    threshold = median * 1.5

    # === 2. 定义保护类 ===
    __PROTECT_CLASSES = {}

    # === 3. 只对「数量超阈值」的大类进行处理 ===
    large_classes = sorted(
        [c for c in classes_set if statistic_res[c] > threshold],
        key=lambda c: statistic_res[c],
        reverse=True  # 先削最大的
    )

    for label in large_classes:
        # 可删除图片候选：
        # 1️⃣ 含该 label
        # 2️⃣ 不含任何保护类
        candidates = [
            img for img in image_info_list
            if img.get(label, 0) > 0
            and all(img.get(p, 0) == 0 for p in __PROTECT_CLASSES)
        ]

        # 优先删除「该 label 数量最多」的图片
        candidates.sort(key=lambda x: x.get(label, 0), reverse=True)

        for img in candidates:
            if statistic_res[label] <= threshold:
                break

            # 更新统计
            for c in classes_set:
                statistic_res[c] -= img.get(c, 0)

            # 删除图片信息、图片以及标签文件
            image_info_list.remove(img)
            if os.path.exists(os.path.join(dataset_path, 'images', type, img["image"] + '.jpg')):
                os.remove(os.path.join(dataset_path, 'images', type, img["image"] + '.jpg'))
            else:
                os.remove(os.path.join(dataset_path, 'images', type, img["image"] + '.png'))
            os.remove(os.path.join(dataset_path, 'labels', type, img["image"] + '.txt'))
    
    