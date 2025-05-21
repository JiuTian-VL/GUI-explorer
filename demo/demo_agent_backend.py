"""
启动 python -m utils.demo_agent_backend
"""

import os
from MLLM_Agent.GUI_explorer import GUI_explorer

os.environ["no_proxy"] = "localhost, 127.0.0.1/8, ::1"
print("Agent Service")
print("Loading Agent...")
assert os.getenv("TURN_ON_DEMO_MODE", "False").lower() == "true"
agent = GUI_explorer()

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


@app.post("/run_task")
async def sent_massage(request: Request):
    """
    ```js
    fetch('http://127.0.0.1:8767/run_task', {
    method: 'POST',
    headers: {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({
        "task_goal": "打开chrome浏览器",
    })
    })
    .then(response => response.text())
    .then(data => console.log(data))
    .catch(error => console.error(error));
    ```
    """
    # 从请求中解析原始 JSON
    massage = await request.json()
    print(f"Received message: {str(massage)[:30]} ...")
    agent.early_stop = False
    agent.run(massage["task_goal"])
    agent.early_stop = False
    return "success"


@app.post("/stop")
async def sent_massage2(request: Request):
    massage = await request.json()
    print(f"Received message: {str(massage)[:30]} ...")
    agent.early_stop = True
    return "success"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8767, timeout_graceful_shutdown=3)
