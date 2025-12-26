import os

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
    统计各张图片的标签数量

    Args:
        classes_path (str):     分类文件路径
        labels_path (str):      标签文件路径

    Returns:
        res (dict):             各张图片的各类表签统计结果及各种标签的总数
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

    return res, set(classes_map.values())

