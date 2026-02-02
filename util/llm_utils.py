import base64
import json
import re

from openai import OpenAI
from io import BytesIO
from PIL.ImageFile import ImageFile as PILImageFile


__client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)


def image_chat(source: list[PILImageFile | str],
               stream=False) -> None:
    """
    图片对话

    Args:
        source (list[PILImageFile | str]):      图片资源
        stream (bool):                          是否开启流式对话(默认 False)

    Returns:
        lambda_obj (LLMResult | None):          返回 LLMResult 对象
    """

    response = __client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": __content(source)
            }
        ],
        stream=stream
    )

    text = __clean_text(response.choices[0].message.content)

    if text is None:
        return text

    return None


def __content(source: list[PILImageFile | str]) -> list[dict]:
    """
    准备多模态输入数据

    准备多模态输入数据

    Args:
        source (list[PILImageFile | str]):      图片资源

    Returns:
        input_data (list[dict]):                输入数据
    """

    input_data = [
        {"type": "text", "text": PROMPT}
    ]

    # 图片资源的 base64 字符串集合
    b64_images = __base64_image(source) if isinstance(source, str) else [__base64_image(s) for s in source]

    for s, b64_image in zip(source, b64_images):
        input_data.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{s[s.rindex('.') + 1:] if isinstance(s, str) else s.filename[s.filename.rindex('.') + 1:]};base64,{b64_image}"
            }
        })

    return input_data


def __base64_image(source: str | PILImageFile) -> str:
    """
    将图片编码为 base64 字符串

    将图片编码为 base64 字符串

    Args:
        source (str | PILImageFile):            图片资源

    Returns:
        b64_image                               base64 字符串
    """
    # 将图片编码为 base64 字符串
    if isinstance(source, str):
        with open(source, 'rb') as f:
            data = f.read()
    elif isinstance(source, PILImageFile):
        buffer = BytesIO()
        source.save(buffer, format=source.format)
        data = buffer.getvalue()
    else:
        raise TypeError(f"Unsupported source type: {type(source)}")

    return base64.b64encode(data).decode('utf-8')


def __clean_text(text: str) -> str | None:
    """
    清理文本

    清理文本

    Args:
        text (str):                             文本

    Returns:
        text (str | None):                      清理后的文本
    """

    # 删除 ```json、``` 和空白
    text = text.replace("```json", "").replace("```", "").strip()

    # 尝试解析为 dict
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        print("json: " + text)
        s_fixed = re.sub(r"'", '"', text)
        data = json.loads(s_fixed)

    return json.dumps(data, ensure_ascii=False, indent=4)

