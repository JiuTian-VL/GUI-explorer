"""
启动 python -m utils.demo_web_backend
"""

import os
import requests


def send_message(message: dict = None, text: str = None, images: list[str] = None):
    url = (
        os.getenv("MESSAGE_SERVER_ENDPOINT", "http://127.0.0.1:5768")
        + "/sent_a_massage"
    )
    rsp, ret = None, None
    try:
        try:
            _message = message if message else {}
            if text:
                _message["text"] = text
            if images:
                _message["images"] = images
            if len(_message.keys()) == 0:
                return None
            rsp = requests.post(url, json=_message)
            ret = rsp.json()
        except:
            ret = rsp.text
    except:
        return ret


def send_message2(message: dict = None, text: str = None, images: list[str] = None):
    url = (
        os.getenv("MESSAGE_SERVER_ENDPOINT", "http://127.0.0.1:5768")
        + "/sent_a_massage2"
    )
    rsp, ret = None, None
    try:
        try:
            _message = message if message else {}
            if text:
                _message["text"] = text
            if images:
                _message["images"] = images
            if len(_message.keys()) == 0:
                return None
            rsp = requests.post(url, json=_message)
            ret = rsp.json()
        except:
            ret = rsp.text
    except:
        return ret


def get_a_message3():
    url = (
        os.getenv("MESSAGE_SERVER_ENDPOINT", "http://127.0.0.1:5768")
        + "/get_a_massage3"
    )
    rsp = requests.get(url)
    try:
        try:
            return rsp.json()
        except:
            return rsp.text
    except:
        return None


def is_need_stop() -> bool:
    msg = str(get_a_message3()).lower()
    return "stop" in msg


from multiprocessing import Queue
from queue import Empty as QueueEmpty
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
from fastapi.middleware.gzip import GZipMiddleware

app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=6)

massage_queue = None
massage_queue2 = None
massage_queue3 = None
from utils.device import Device

d = Device()

from PIL import Image
import io
import base64


@app.get("/get_a_massage")
async def get_massage():
    """
    ```js
    fetch('http://127.0.0.1:5768/get_a_massage', {
    method: 'GET',
    headers: {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }})
    .then(response => response.text())
    .then(data => console.log(data))
    .catch(error => console.error(error));
    ```
    """
    try:
        msg = massage_queue.get_nowait()
        # print(f"Sent message: {str(msg)[:30]} ...")
        return msg
    except QueueEmpty:
        # print(f"Sent message: <None> ")
        return "<None>"


@app.post("/sent_a_massage")
async def sent_massage(request: Request):
    """
    ```js
    fetch('http://127.0.0.1:5768/sent_a_massage', {
    method: 'POST',
    headers: {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({
        "data": "测试任意的json body",
        "image":"123"
    })
    })
    .then(response => response.text())
    .then(data => console.log(data))
    .catch(error => console.error(error));
    ```
    """
    # 从请求中解析原始 JSON
    massage = await request.json()
    massage_queue.put(massage)
    print(f"Received message: {str(massage)[:30]} ...")
    return "success"


@app.get("/get_a_massage2")
async def get_massage2():
    try:
        msg = massage_queue2.get_nowait()
        return msg
    except QueueEmpty:
        return "<None>"


@app.post("/sent_a_massage2")
async def sent_massage2(request: Request):
    massage = await request.json()
    massage_queue2.put(massage)
    print(f"Received message: {str(massage)[:30]} ...")
    return "success"


@app.get("/get_a_massage3")
async def get_massage3():
    try:
        msg = massage_queue3.get_nowait()
        return msg
    except QueueEmpty:
        return "<None>"


@app.post("/sent_a_massage3")
async def sent_massage3(request: Request):
    massage = await request.json()
    massage_queue3.put(massage)
    print(f"Received message: {str(massage)[:30]} ...")
    return "success"


@app.post("/reset")
async def reset(request: Request):
    massage = await request.json()
    print(f"Received message: {str(massage)[:30]} ...")
    d.stop_all_apps()
    # d.home()
    return "success"


from fastapi.responses import StreamingResponse


@app.get("/get_screenshot")
async def get_screenshot():
    sc = d.get_screenshot()
    scale = 2.4
    sc = sc.resize((int(sc.size[0] / scale), int(sc.size[1] / scale)), Image.LANCZOS)
    # sc = sc.convert("RGB")
    buffered = io.BytesIO()
    sc.save(buffered, format="WEBP", quality=75)
    buffered.seek(0)
    return StreamingResponse(buffered, media_type="image/webp")


if __name__ == "__main__":
    print("Fast API is starting")
    massage_queue = Queue()
    massage_queue2 = Queue()
    massage_queue3 = Queue()
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5768, timeout_graceful_shutdown=3)
