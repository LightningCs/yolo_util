import os

import json
import re

import PIL.Image as Image
import requests

from tqdm import tqdm


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


def import_img(import_url: str,
               reimport_url: str,
               path: str,
               token: str,
               auth: str):
    """
    将图片导入label-studio中，用于上传预测文件中图片映射

    将图片导入label-studio中，用于上传预测文件中图片映射

    Args:
        import_url (str):           导入接口url
        reimport_url (str):         再导入接口url
        path (str):                 图片所在路径
        token (str):                token
        auth (str):                 用户api_key
    """
    for file in tqdm(os.listdir(path)):
        print(f'正在处理{file}...')
        # 导入文件
        response = requests.post(import_url,
                                 headers={
                                     'token': token,
                                     'Authorization': auth
                                 },
                                 files=[('filename', (
                                     file, open(os.path.join(path, file), 'rb'), 'image/png'))])

        print(f'{response.json()}')
        file_upload_ids = response.json()['file_upload_ids']

        # 文件再导入
        response = requests.post(reimport_url,
                                 headers={
                                     'token': token,
                                     'Authorization': auth
                                 },
                                 data={'file_upload_ids': file_upload_ids,
                                       'files_as_tasks_list': False})

        print('处理完毕...')
