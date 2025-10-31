import os
import math
import random
import shutil
import json
import re

import PIL.Image as Image
import ultralytics

import yaml
from ultralytics.utils.metrics import DetMetrics


class YoloConfig:
    """
    配置类
    """
    DEFAULT_BATCH_SIZE = 16
    DEFAULT_EPOCHS = 100
    DEFAULT_IMGSZ = 640
    DEFAULT_TEST_RATE = 0.2
    DEFAULT_CONF = 0.25
    DEFAULT_DEVICE = 'cpu'


def yolo_to_Json(score: float,
                 version: str,
                 predict_result_path: str,
                 target_path: str,
                 classes_path: str) -> None:
    """
    将yolo格式数据转化为json格式

    将yolo格式数据转化为json格式

    Args:
        score (float):              置信度
        version (str):              模型版本
        predict_result_path (str):  预测结果存放路径，即项目路径(project)
        target_path (str):          json文件存放路径
        classes_path (str):         分类文件存放路径
    """
    # 所有上传文件映射
    upload_file_map = {
        file[((file.rindex('-') if re.match('^.*-.*.[a-z]{3}$', file) else -1) + 1): file.rindex('.')]: file
        for file in
        os.listdir('C:\\Users\\86151\\AppData\\Local\\label-studio\\label-studio\\media\\upload\\3')}
    # 分类
    categories = open(classes_path, 'r', encoding='utf-8').readlines()
    # 分类映射表
    categories_map = {i: categories[i].strip() for i in range(len(categories))}
    #  获取预测结果存放目录的所有子目录
    dirs = os.listdir(predict_result_path)
    data_list = []

    for dir in dirs:
        cur_path = os.path.join(predict_result_path, dir, 'labels')
        # cur_path = os.path.join(predict_result_path, dir)
        results = []
        # 遍历所有标注结果文件
        for file in os.listdir(cur_path):
            id = 1
            image = Image.open(os.path.join(predict_result_path, dir, file[:file.rindex('.')] + '.jpg'))
            with open(os.path.join(cur_path, file), 'r', encoding='utf-8') as f:
                content = f.readlines()
                # 遍历所有标注结果
                for line in content:
                    val = line.strip().split(' ')
                    width, height = image.size
                    category = categories_map[int(val[0])]
                    x = float(val[1])
                    y = float(val[2])
                    w = float(val[3])
                    h = float(val[4])

                    result = {
                        "id": str(id),
                        "type": "polygonlabels",
                        "from_name": "label",
                        "to_name": "image",
                        "original_width": width,
                        "original_height": height,
                        "image_rotation": 0,
                        "value": {
                            "rotation": 0,
                            "x": (x - w / 2) * 100,
                            "y": (y - h / 2) * 100,
                            "width": w * 100,
                            "height": h * 100,
                            "polygonlabels": [category]
                        }
                    }

                    id += 1
                    results.append(result)

        data_list.append({
            "data": {
                "image": f"/data/upload/3/{upload_file_map[dir[((dir.rindex('-') if re.match('^.*-.*.[a-z]{3}$', dir) else -1) + 1):]]}"},
            "predictions": [
                {
                    "model_version": version,
                    "score": score,
                    "result": results
                }
            ]
        })

    # 保存至json文件
    json.dump(data_list, open(target_path, 'w', encoding='utf-8'), ensure_ascii=False)


def gen_img_json(project: int,
                 predict_result_path: str,
                 target_path: str):
    """
    生成将图片导入label-studio的json文件

    生成将图片导入label-studio的json文件

    Args:
        project (int):              项目编号
        predict_result_path (str):  预测结果存放路径
        target_path (str):          json文件存放路径
    """
    data_list = []

    for dir in os.listdir(predict_result_path):
        # 遍历所有标注结果文件
        for file in os.listdir(os.path.join(predict_result_path, dir)):
            if os.path.isfile(os.path.join(predict_result_path, dir, file)):
                data_list.append({
                    "data": {"image": f"/data/upload/{project}/{file}"},
                    "annotations": [],
                    "predictions": []
                })

    json.dump(data_list, open(target_path, 'w', encoding='utf-8'), ensure_ascii=False)


def load_categories(classes_path: str) -> dict:
    """
    加载分类编号映射

    加载分类编号映射

    Args:
        classes_path (str):         分类文件所在路径

    Return:
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
        file_name (str):            文件名称
        datasets_path (str):        数据集根路径
        train_path (str):           训练图片集合存放路径
        val_path (str):             验证图片集合存放路径
        test_path (str):            测试图片集合存放路径

    Return:
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
                batch=YoloConfig.DEFAULT_BATCH_SIZE,
                epochs=YoloConfig.DEFAULT_EPOCHS,
                imgsz=YoloConfig.DEFAULT_IMGSZ,
                rect=False,
                resume=False) -> tuple[dict, str]:
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
        batch (int):                每个批次的图片数量(默认16)
        epochs (int):               训练总轮数(默认100)
        imgsz (int):                图片大小(若rect为False，则为imgsz*imgsz, 默认640)
        rect (boolean):             是否启用最小填充策略(默认不启动)
        resume (boolean):           是否从上次保存的检查点恢复训练(若以训练完毕则设置为false，否则会报错)

    Return:
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
                          exist_ok=True,
                          resume=resume)

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

    Return:
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


def predict(model: ultralytics.YOLO,
            project: str,
            path: str,
            predict_images: list,
            classes_path: str,
            conf=YoloConfig.DEFAULT_CONF) -> list:
    """
    图片预测

    对图片进行预测

    Args:
        model (ultralytics.YOLO):   模型
        project (str):              项目路径
        path (str):                 带预测图片路径
        predict_images (list):      待预测图片集合
        classes_path (str):         类型文件路径
        conf (float):               检测的最小置信度阈值(默认0.25)

    Return:
        results (list):             预测结果集合
    """
    # 分类
    categories_map = load_categories(classes_path)

    results = []

    # 预测
    for image in predict_images:
        tmp = []
        # 预测结果
        result = model.predict(os.path.join(path, image),
                               project=f'{project}/predict_images-{conf}',
                               name=image[:image.rindex('.')],
                               conf=conf,
                               classes=[i for i in range(len(categories_map))],
                               save=True,
                               save_txt=True,
                               save_conf=True,
                               save_crop=True)

        # 将预测结果编号映射为名称
        for i in result[0].boxes.cls:
            tmp.append(categories_map[int(i)])

        results.append(tmp)

    return results


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
