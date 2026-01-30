import os
import math
import random
import shutil

import PIL.Image
import ultralytics

import yaml
from fastapi import UploadFile
from ultralytics.utils.metrics import DetMetrics
# 开启进度条
from tqdm import tqdm
from .label_utils import statistic_labels_num_by_images


class YoloConfig:
    """
    配置类
    """
    DEFAULT_BATCH_SIZE = 16
    DEFAULT_EPOCHS = 100
    DEFAULT_IMGSZ = 640
    DEFAULT_WORKERS = 0
    DEFAULT_TRAINT_RATE = 0.7 
    DEFAULT_VAL_RATE = 0.3
    DEFAULT_TEST_RATE = 0.2
    DEFAULT_CONF = 0.25
    DEFAULT_HSV_V = 0.4
    DEFAULT_TRANSLATE = 0.1
    DEFAULT_DEVICE = 'cpu'
    DEFAULT_WORKERS = None


def load_categories(classes_path: str) -> dict:
    """
    加载分类编号映射

    加载分类编号映射

    Args:
        classes_path (str):         分类文件所在路径

    Returns:
        categories_map (dict):      分类编号映射
    """
    categories_map = {}
    with open(classes_path, 'r', encoding='utf-8') as f:
        categories = f.readlines()
        for i, category in enumerate(categories):
            categories_map[i] = category.strip()
    return categories_map


def gen_datasets_conf(classes_path: str,
                      target_path: str,
                      file_name: str,
                      datasets_path: str,
                      train_path: str,
                      val_path: str,
                      test_path: str) -> str:
    """
    生成数据集配置文件

    生成数据集配置文件

    Args:
        classes_path (str):         分类集文件路径
        target_path (str):          数据集配置文件存放路径
        file_name (str):            文件名称(无需指定尾缀)
        datasets_path (str):        数据集根路径
        train_path (str):           训练图片集合存放路径
        val_path (str):             验证图片集合存放路径
        test_path (str):            测试图片集合存放路径

    Returns:
        datasets_conf_path (str):   数据集配置文件存放路径
    """
    # 分类字典
    categories = load_categories(classes_path)

    # 数据集配置文件
    datasets_conf_obj = {
        "path": datasets_path,
        "train": train_path,
        "val": val_path,
        "test": test_path,
        "names": categories
    }

    # 保存数据集配置文件
    yaml.dump(datasets_conf_obj, open(os.path.join(target_path, file_name + '.yaml'), 'w', encoding='utf-8'),
              allow_unicode=True,
              encoding='utf-8')

    return os.path.join(target_path, file_name + '.yaml')


def init_file(source_path: str,
              target_path: str) -> None:
    """
    初始化目录

    复制图片和标签到目标目录

    Args:
        source_path (str):          图片源路径
        target_path (str):          存储目标路径
    """
    # 初始化目录
    if os.path.exists(os.path.join(target_path, 'images')):
        shutil.rmtree(os.path.join(target_path, 'images'))

    if os.path.exists(os.path.join(target_path, 'labels')):
        shutil.rmtree(os.path.join(target_path, 'labels'))

    # 训练集目录
    shutil.copytree(os.path.join(source_path, 'images'), os.path.join(target_path, 'images', 'train'))
    shutil.copytree(os.path.join(source_path, 'images'), os.path.join(target_path, 'labels', 'train'))

    # 测试集目录
    os.mkdir(os.path.join(target_path, 'images', 'val'))
    os.mkdir(os.path.join(target_path, 'labels', 'val'))


def split_datasets(classes_path: str,
                   datasets_path: str,
                   target_path: str,
                   train_rate=YoloConfig.DEFAULT_TRAINT_RATE,
                   val_rate=YoloConfig.DEFAULT_VAL_RATE):
    """
    数据集分割

    数据集分割

    Args:
        classes_path (str):         分类文件路径
        datasets_path (str):        数据集根路径
        target_path (str):          存储路径
        train_rate (float):         训练集占比(默认0.7)
        val_rate (float):           验证集占比(默认0.3)
    """
    
    if not os.path.exists(target_path):
        print('创建根目录')
        os.mkdir(target_path)

    if not os.path.exists(os.path.join(target_path)):
        print('创建根目录')
        os.mkdir(target_path)

    if not os.path.exists(os.path.join(target_path, 'images')):
        print('创建图片目录')
        os.mkdir(os.path.join(target_path, 'images'))

    if not os.path.exists(os.path.join(target_path, 'labels')):
        print('创建标签目录')
        os.mkdir(os.path.join(target_path, 'labels'))

    if not os.path.exists(os.path.join(target_path, 'images', 'train')):
        # 创建图片训练集目录
        print('创建图片训练集目录')
        os.mkdir(os.path.join(target_path, 'images', 'train'))
    if not os.path.exists(os.path.join(target_path, 'labels', 'train')):
        # 创建标签训练集目录
        print('创建标签训练集目录')
        os.mkdir(os.path.join(target_path, 'labels', 'train'))
    if not os.path.exists(os.path.join(target_path, 'images', 'val')):
        # 创建图片验证集目录
        print('创建图片验证集目录')
        os.mkdir(os.path.join(target_path, 'images', 'val'))
    if not os.path.exists(os.path.join(target_path, 'labels', 'val')):
        # 创建标签验证集目录
        print('创建标签验证集目录')
        os.mkdir(os.path.join(target_path, 'labels', 'val'))
    # 若训练集占比加验证集占比不为 1.0，则创建测试集
    if train_rate + val_rate != 1.0:
        if not os.path.exists(os.path.join(target_path, 'images', 'test')):
            print('创建图片测试集目录')
            os.mkdir(os.path.join(target_path, 'images', 'test'))
        if not os.path.exists(os.path.join(target_path, 'labels', 'test')):
            print('创建标签测试集目录')
            os.mkdir(os.path.join(target_path, 'labels', 'test'))

    statistic_res, classes_set = statistic_labels_num_by_images(classes_path=classes_path,
                                                                labels_path=os.path.join(datasets_path, 'labels'))

    image_entity_list = statistic_res["result"]
    # 随机排序
    random.shuffle(image_entity_list)

    # 数据集数组——其中索引 0 为训练集、1 为验证集、2 为测试集
    datasets_arr = init_datasets_arr(image_entity_list, statistic_res, classes_set, train_rate, val_rate)

    # 图片划分
    split_images(image_entity_list, datasets_arr, statistic_res, classes_set, datasets_path, target_path, train_rate, val_rate)


def init_datasets_arr(
        image_entity_list: list,
        statistic_res: dict,
        classes_set: set,
        train_rate: float,
        val_rate: float) -> list:
    """
    初始化数据集数组

    初始化数据集数组

    Args:
        image_entity_list (list):   图片信息列表
        statistic_res (dict):       统计结果
        classes_set (set):          标签集合
        train_rate (float):         训练集占比
        val_rate (float):           验证集占比

    Returns:
        list: 数据集数组
    """
    datasets_arr = [{}, {}, {}]

    # 更新训练集图片总数
    datasets_arr[0]["total"] = math.ceil(len(image_entity_list) * train_rate)
    # 验证集图片总数
    if train_rate + val_rate != 1.0:
        datasets_arr[1]["total"] = math.ceil(len(image_entity_list) * val_rate)
    else:
        datasets_arr[1]["total"] = len(image_entity_list) - datasets_arr[0]["total"]
    # 测试集图片总数
    datasets_arr[2]["total"] = len(image_entity_list) - datasets_arr[0]["total"] - datasets_arr[1]["total"]

    # 遍历各种标签
    for classes in classes_set:
        # 更新训练集中的标签数量
        datasets_arr[0][classes] = math.ceil(statistic_res.get(classes, 0) * train_rate)
        # 若训练集占比加验证集占比不为 1.0，则创建测试集
        if train_rate + val_rate != 1.0:
            datasets_arr[1][classes] = math.ceil(statistic_res.get(classes, 0) * val_rate)
        else:
            datasets_arr[1][classes] = statistic_res.get(classes, 0) - datasets_arr[0].get(classes, 0)
        # 更新测试集中的标签数量
        datasets_arr[2][classes] = statistic_res.get(classes, 0) - datasets_arr[0].get(classes, 0) - datasets_arr[
            1].get(classes, 0)

    return datasets_arr


def split_images(image_entity_list: list,
                 datasets_arr: list,
                 statistic_res: dict,
                 classes_set: set,
                 datasets_path: str,
                 target_path: str,
                 train_rate:  float,
                 val_rate: float):
    """
    图片划分算法

    图片划分算法

    Args:
        image_entity_list (list):   图片信息列表
        datasets_arr (list):        训练集、验证集、测试集
        statistic_res (dict):       统计结果
        classes_set (set):          标签集合
        datasets_path (str):        数据集根路径
        target_path (str):          存储路径
        train_rate (float):         训练集占比
        val_rate (float):           验证集占比
    """
    for image_entity in tqdm(image_entity_list):
        # 最大分数及索引
        max_i = 0
        max_score = 0.0

        # 初始化索引
        for i in range(len(datasets_arr)):
            if datasets_arr[i]["total"] > 0:
                max_i = i
                break

        # 遍历训练集、验证集、测试集
        for i in range(len(datasets_arr)):
            # 若图片已分配完成，则跳过
            if datasets_arr[i]["total"] <= 0:
                continue

            score = 0

            for classes in classes_set:
                if datasets_arr[i].get(classes, 0) > 0:
                    # 当前数据集该标签剩余数量 / 该标签总数量
                    demand_ratio = datasets_arr[i][classes] / statistic_res[classes]
                    # 图片中该标签数量 * 需求紧迫程度
                    score += image_entity.get(classes, 0) * demand_ratio

            score *= train_rate if i == 0 else (val_rate if i == 1 else 1 - train_rate - val_rate)

            # 更新最大分数及索引
            if score > max_score:
                max_score = score
                max_i = i

        # 更新数量
        for classes in classes_set:
            # 减去图片中该类别的图片数量，不能小于 0
            datasets_arr[max_i][classes] = max(datasets_arr[max_i].get(classes, 0) - image_entity.get(classes, 0), 0)
            statistic_res["total"] -= image_entity.get(classes, 0)

        # 减去图片数量，不能小于 0
        datasets_arr[max_i]["total"] -= 1
        type = "train" if max_i == 0 else ("val" if max_i == 1 else "test")

        if os.path.exists(os.path.join(datasets_path, 'images', f'{image_entity["image"]}.png')):
            shutil.copy(os.path.join(datasets_path, 'images', f'{image_entity["image"]}.png'), os.path.join(target_path, 'images', type, f'{image_entity["image"]}.png'))
        elif os.path.exists(os.path.join(datasets_path, 'images', f'{image_entity["image"]}.jpg')):
            shutil.copy(os.path.join(datasets_path, 'images', f'{image_entity["image"]}.jpg'), os.path.join(target_path, 'images', type, f'{image_entity["image"]}.jpg'))
        else:
            raise FileNotFoundError(f'{datasets_path}\\images\\{image_entity["image"]}.png or .jpg not found')

        shutil.copy(os.path.join(datasets_path, 'labels', f'{image_entity["image"]}.txt'), os.path.join(target_path, 'labels', type, f'{image_entity["image"]}.txt'))


def gen_yolo_datasets(source_path: str,
                      target_path: str,
                      test_rate=YoloConfig.DEFAULT_TEST_RATE) -> None:
    """
    生成yolo数据集

    生成yolo数据集

    Args:
        source_path (str):          图片源路径
        target_path (str):          存储目标路径
        test_rate (float):          验证集占比(默认0.2)
    """
    init_file(source_path, target_path)

    file_list = os.listdir(f'{target_path}/images/train')

    # 训练集数量
    test_num = math.ceil(len(file_list) * test_rate)

    while test_num > 0:
        # 随机索引
        random_index = random.randint(0, len(file_list) - 1)

        # 移动文件
        shutil.move(os.path.join(target_path, 'images', 'train', file_list[random_index]),
                    os.path.join(target_path, 'images', 'val', file_list[random_index]))
        shutil.move(os.path.join(target_path, 'labels', 'train', file_list[random_index][:-4] + '.txt'),
                    os.path.join(target_path, 'labels', 'val', file_list[random_index][:-4] + '.txt'))

        file_list.pop(random_index)

        # 数据集和训练集数量减一
        test_num -= 1


def train_model(model: ultralytics.YOLO,
                datasets_conf_path: str,
                alias: str,
                project: str,
                lr0: float,
                lrf: float,
                workers=YoloConfig.DEFAULT_WORKERS,
                device=None,
                batch=YoloConfig.DEFAULT_BATCH_SIZE,
                epochs=YoloConfig.DEFAULT_EPOCHS,
                imgsz=YoloConfig.DEFAULT_IMGSZ,
                cos_lr=False,
                cache=False,
                rect=False,
                resume=False,
                hsv_v=YoloConfig.DEFAULT_HSV_V,
                translate=YoloConfig.DEFAULT_TRANSLATE) -> tuple[dict, str]:
    """
    训练和验证模型

    训练和验证模型

    Args:
        model (ultralytics.YOLO):   模型
        datasets_conf_path (str):   数据集配置文件存放路径
        alias (str):                模型别名
        project (str):              项目目录名称
        lr0 (float):                初始学习率
        lrf (float):                最终学习率
        workers (int):              用于数据加载的工作线程数（每个 RANK ，如果是多 GPU 训练）。影响数据预处理和输入模型的速度，在多 GPU 设置中尤其有用
        device (int | list):        指定用于训练的计算设备：单个 GPU（device=0），多个 GPU（device=[0,1]），CPU（device=cpu），适用于 Apple 芯片的 MPS（device=mps），或自动选择最空闲的 GPU（device=-1）或多个空闲 GPU （device=[-1,-1])
        batch (int):                每个批次的图片数量(默认16)
        epochs (int):               训练总轮数(默认100)
        imgsz (int):                图片大小(若rect为False，则为imgsz*imgsz, 默认640)
        cos_lr (bool):              使用余弦学习率调度器，在 epochs 上按照余弦曲线调整学习率。有助于管理学习率，从而实现更好的收敛
        cache (bool):               启用在内存中缓存数据集图像（True/ram），在磁盘上缓存（disk），或禁用缓存（False）。通过减少磁盘 I/O 来提高训练速度，但会增加内存使用量
        rect (bool):                是否启用最小填充策略(默认不启动)
        resume (bool):              是否从上次保存的检查点恢复训练(若以训练完毕则设置为false，否则会报错)
        hsv_v (float):              通过一小部分修改图像的明度（亮度），帮助模型在各种光照条件下表现良好
        translate (float):          将图像按图像尺寸的一小部分进行水平和垂直平移，有助于学习检测部分可见的物体

    Returns:
        results (dict):             模型训练结果
        model_name (str):           模型名称
    """
    model_name = f'{alias}-{batch}-{epochs}-{lr0}-{lrf}'

    # 训练
    results = model.train(data=datasets_conf_path,
                          imgsz=imgsz,
                          rect=rect,
                          batch=batch,
                          epochs=epochs,
                          project=project,
                          name=f'{model_name}/train',
                          lr0=lr0,
                          lrf=lrf,
                          workers=workers,
                          device=device,
                          exist_ok=True,
                          resume=resume,
                          cos_lr=cos_lr,
                          cache=cache,
                          hsv_v=hsv_v,
                          translate=translate)

    return results, model_name


def val_model(model: ultralytics.YOLO,
              project: str,
              name: str,
              imgsz=YoloConfig.DEFAULT_IMGSZ,
              rect=False) -> DetMetrics:
    """
    验证模型

    验证模型

    Args:
        model (ultralytics.YOLO):   模型
        project (str):              项目目录名称
        name (str):                 模型名称
        imgsz (int):                图片大小(若rect为False，则为imgsz*imgsz, 默认640)
        rect (boolean):             是否启用最小填充策略

    Returns:
        metrics (dict):             模型评估指标
    """
    # 验证
    metrics = model.val(project=project,
                        name=f'{name}/val',
                        imgsz=imgsz,
                        rect=rect,
                        plots=True,
                        save_txt=True,
                        save_conf=True)

    return metrics


def predict_img(model: ultralytics.YOLO,
                project: str,
                source: str | list[UploadFile],
                classes_path: str,
                conf=YoloConfig.DEFAULT_CONF) -> list[set]:
    """
    图片预测

    对图片进行预测

    Args:
        model (ultralytics.YOLO):       模型
        project (str):                  项目路径
        source (str | list[UploadFile]):待预测图片路径
        classes_path (str):             类型文件路径
        conf (float):                   检测的最小置信度阈值(默认0.25)

    Returns:
        results (list[set]):            预测结果集合
    """
    # 分类
    categories_map = load_categories(classes_path)
    results = []
    n = len(categories_map)

    if isinstance(source, str):
        for img in source:
            tmp = set()
            result = predict(model=model,
                             project=project,
                             name=img[:img.rindex('.')],
                             source=os.path.join(source, img),
                             num=n,
                             conf=conf)

            for i in result[0].boxes.cls:
                tmp.add(categories_map[int(i)])

            results.append(tmp)
    elif isinstance(source, list):
        for f in source:
            tmp = set()
            result = predict(model=model,
                             project=project,
                             name=f.filename[:f.filename.rindex('.')],
                             source=PIL.Image.open(f.file),
                             num=n,
                             conf=conf)

            for i in result[0].boxes.cls:
                tmp.add(categories_map[int(i)])

            results.append(tmp)

    return results


def predict(model: ultralytics.YOLO,
            project: str,
            name: str,
            source: str | PIL.Image.Image,
            num: int,
            conf=YoloConfig.DEFAULT_CONF) -> list:
    """
    图片预测

    对图片进行预测

    Args:
        model (ultralytics.YOLO):       模型
        project (str):                  项目路径
        name (str):                     项目名称
        source (str | PIL.Image.Image): 待预测图片
        num (int):                      分类数量
        conf (float):                   检测的最小置信度阈值(默认0.25)

    Returns:
        results (list):                 预测结果集合
    """

    return model.predict(source,
                         project=f'{project}/predict_images-{conf}',
                         name=name,
                         conf=conf,
                         classes=[i for i in range(num)],
                         save=True,
                         save_txt=True,
                         save_conf=True,
                         save_crop=True)


def export_model(model: ultralytics.YOLO,
                 format: str,
                 imgsz=YoloConfig.DEFAULT_IMGSZ,
                 dynamic=False,
                 device=YoloConfig.DEFAULT_DEVICE) -> None:
    """
    导出模型

    导出模型

    Args:
        model (ultralytics.YOLO):   模型
        format (str):               导出格式
        imgsz (int):                图片大小(默认640)
        dynamic (bool):             动态输入大小
        device (str):               指定导出设备(默认cpu)
    """
    model.export(format=format,
                 imgsz=imgsz,
                 dynamic=dynamic,
                 device=device)
