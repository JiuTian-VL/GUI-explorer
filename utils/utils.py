from androguard.util import set_log

try:
    set_log("ERROR")  # 关闭琐碎的DEBUG输出
except:
    pass

import subprocess
import time
import re
from androguard.core.apk import APK
import os

# from dotenv import load_dotenv
import io
import json
from PIL import Image
import uuid
import base64
import hashlib
import cv2


# load_dotenv(verbose=True, override=True)

import requests
import urllib3

urllib3.disable_warnings()

import pickle
import zstd


def save_object_to_disk(obj: object, file_path: str, compress_level: int = 3):
    """将对象序列化为pickle格式并使用Zstandard压缩保存到本地文件
    Args:
        obj (object): 要保存的对象
        file_path (str): 保存文件的路径
        compress_level (int): compression level, ultra-fast levels from -100 (ultra) to -1 (fast) available since zstd-1.3.4, and from 1 (fast) to 22 (slowest), 0 or unset - means default (3). Default 3.
    """
    pickled_data = pickle.dumps(obj)
    compressed_data = zstd.compress(pickled_data, compress_level)
    with open(file_path, "wb") as file:
        file.write(compressed_data)


def load_object_from_disk(file_path: str) -> object:
    """从本地文件读取Zstandard压缩的pickle数据并反序列化为对象"""
    with open(file_path, "rb") as file:
        compressed_data = file.read()
    pickled_data = zstd.decompress(compressed_data)
    return pickle.loads(pickled_data)


from PIL import Image
import numpy as np


def resize_pil_image(image: Image.Image, target_max_size: int = 1000) -> Image.Image:
    """
    Resize a PIL image to fit within a square of target_max_size x target_max_size pixels,
    maintaining the aspect ratio.
    """
    width, height = image.size
    if width > height:
        new_width = target_max_size
        new_height = int((height / width) * target_max_size)
    else:
        new_height = target_max_size
        new_width = int((width / height) * target_max_size)
    return image.resize((new_width, new_height), Image.LANCZOS)


def resize_ndarray_image(image: np.ndarray, target_max_size: int = 1000) -> np.ndarray:
    """
    Resize a numpy ndarray image to fit within a square of target_max_size x target_max_size pixels, maintaining the aspect ratio.
    """
    return np.array(resize_pil_image(Image.fromarray(image), target_max_size))


def openai_request(
    messages: list,
    model: str = "env",
    max_retry: int = 5,
    timeout: int = 60,
    temperature: float = 0.0,
    max_tokens: int = 300,
    usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0},
) -> str:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f'Bearer {os.getenv("OPENAI_API_KEY")}',
    }
    data = {
        "model": os.getenv("OPENAI_API_MODEL", model) if model == "env" else model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    url = (
        f"{os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")}/chat/completions"
    )
    HTTP_PROXY = os.getenv("HTTP_PROXY", None)
    proxies = None
    # if HTTP_PROXY:
    #     proxies = {
    #         "http": HTTP_PROXY,
    #         "https": HTTP_PROXY,
    #     }
    r = None
    for i in range(max_retry + 1):
        try:
            r = requests.post(
                url,
                headers=headers,
                json=data,
                timeout=timeout,
                verify=False,  # 禁用证书验证
                proxies=proxies,
            )  # .json()
            d = r.json()
            content = d.get("choices", [{}])[0].get("message", {})["content"]
            usage["prompt_tokens"] += d.get("usage", {}).get("prompt_tokens", 0)
            usage["completion_tokens"] += d.get("usage", {}).get("completion_tokens", 0)
            return content
        except Exception as e:
            print(
                f"Request failed: {e} , retrying {i+1} of {max_retry} after {(i + 1) ** 3} seconds"
            )
            if r is not None:
                print(r.text)
            time.sleep((i + 1) ** 3)
    raise Exception(f"Request failed after retrying {max_retry} times")


def str_to_md5(input_str: str) -> str:
    return hashlib.md5(input_str.encode()).hexdigest().upper()


def pil_to_webp_base64(img: Image.Image) -> str:
    buffered = io.BytesIO()
    img.convert("RGB").save(buffered, format="WEBP", quality=95)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def ndarray_to_webp_base64(img: np.ndarray) -> str:
    """
    Convert a numpy ndarray image to a base64 encoded string.
    """
    return pil_to_webp_base64(Image.fromarray(img))


def base64_to_pil(base64_str: str) -> Image.Image:
    """
    Convert a base64 encoded string to a PIL Image.

    Args:
        base64_str (str): The base64 string representing the image.

    Returns:
        Image.Image: A PIL Image object.
    """
    return Image.open(io.BytesIO(base64.b64decode(base64_str))).convert("RGB")


def cv2_to_pil(cv2_img):
    # 将 cv2 图像转换为 RGB 格式（OpenCV 使用 BGR）
    cv2_img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    # 将 NumPy 数组转换为 PIL 图像
    pil_img = Image.fromarray(cv2_img_rgb)
    return pil_img


def safe_decode(byte_data, encoding_list=["utf-8", "gbk"]):
    for encoding in encoding_list:
        try:
            return byte_data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"Unable to decode with encodings: {encoding_list}")


import ast
import re
import json
from typing import Any, Optional


def extract_json(s: str) -> Optional[dict[str, Any]]:
    """Extracts the first JSON object found in a string.

    Handles multi-line JSON and JSON embedded within other text.

    Args:
      s: A string potentially containing a JSON object.
         E.g., "{'hello': 'world'}" (Python-like) or '"key": "value", "boolean": true, "nothing": null' (Standard JSON) or CoT: "let's think step-by-step, ..., { ... json ... } ... more text"

    Returns:
      The parsed JSON object as a Python dictionary, or None if no valid
      JSON object is found or parsing fails.
    """
    pattern = r"\{.*\}"
    match = re.search(pattern, s, re.DOTALL)
    if match:
        potential_json_string = match.group()
        try:
            return json.loads(potential_json_string)
        except json.JSONDecodeError as json_error:
            # print(
            #     f"JSON parsing failed ({json_error}), attempting Python literal eval."
            # )
            try:
                return ast.literal_eval(potential_json_string)
            except (SyntaxError, ValueError) as eval_error:
                print(
                    f"Python literal eval also failed ({eval_error}), cannot extract dictionary."
                )
                return None
    else:
        return None


def get_apk(package_name: str, local_apk_path: str, device_serial: str = None) -> str:
    command = "adb "
    if device_serial:
        command += f" -s {device_serial} "
    command += f" shell pm path {package_name}"
    apk_path = execute_cmd(command)
    if apk_path == "ERROR":
        return "ERROR"
    apk_path = apk_path.split("package:")[1].strip()
    command = "adb "
    if device_serial:
        command += f" -s {device_serial} "
    command += f" pull {apk_path} {local_apk_path}"
    return execute_cmd(command)


def execute_cmd(command: str, verbose=True) -> str:
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    if verbose:
        print(f"Command execution failed: {command}")
        print(result.stderr)
    return "ERROR"


def get_all_devices() -> list:
    command = "adb devices"
    device_list = []
    result = execute_cmd(command)
    if result != "ERROR":
        devices = result.split("\n")[1:]
        for d in devices:
            device_list.append(d.split()[0])

    return device_list
