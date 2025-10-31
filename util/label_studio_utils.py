import os

import requests


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
    for file in path:
        print(f'正在处理{file}...')
        # 导入文件
        response = requests.post(import_url,
                                 headers={
                                     'token': token,
                                     'Authorization': auth},
                                 files=[('filename', (
                                     file, open(os.path.join(path, file), 'rb'), 'image/png'))])

        print(f'{response.json()}')
        file_upload_ids = response.json()['file_upload_ids']

        # 文件再导入
        response = requests.post(reimport_url,
                                 headers={
                                     'token': token,
                                     'Authorization': auth},
                                 data={'file_upload_ids': file_upload_ids,
                                       'files_as_tasks_list': False})

        print('处理完毕...')
