# -*- coding: utf-8 -*-
"""
websocket_client_policy.py —— 机器人端 websocket 客户端
====================================================================
【作用】跑在机器人(或仿真器)这边。通过网络连到推理服务端,
       把机器人当前观测(图像+语言+状态)发过去,收回模型生成的动作。

【核心流程(infer)】
  pack(观测) → ws.send → ws.recv → unpack(动作)
  (和服务端 _handler 的 recv→infer→send 正好反过来配对)

【观测长什么样】(见文件末尾 main 的示例)
  {
    "image": {3 个相机 RGB},   ← 图像
    "state": 关节状态,         ← 当前状态
    "prompt": ["任务指令"],    ← 语言
  }
  正是之前确认的"推理输入三件套"。

【在项目中的位置】
  跑在机器人端;连到 websocket_policy_server.py(GPU 服务器端)。
  机器人控制循环里反复调 client.infer(观测) → 得动作 → 执行。
====================================================================
"""
import logging
import os
import time
from typing import Dict, Optional, Tuple

from typing_extensions import override
import websockets.sync.client
from .msgpack_numpy import Packer, unpackb     # msgpack 序列化(numpy 友好)


class WebsocketClientPolicy:
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    机器人端客户端:连服务端,发观测、收动作。"""

    def __init__(self, host: str = "0.0.0.0", port: Optional[int] = None, api_key: Optional[str] = None) -> None:
        self._uri = f"ws://{host}"                # 拼接服务端地址
        if port is not None:
            self._uri += f":{port}"
        self._packer = Packer()                    # msgpack 打包器
        self._api_key = api_key                    # 可选鉴权密钥
        self._ws, self._server_metadata = self._wait_for_server()   # ★ 启动时就连服务端

    def get_server_metadata(self) -> Dict:
        """返回服务端的元信息(连接时收到的)。"""
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        """★ 连接服务端(带重试):服务端还没起就一直等,连上后收 metadata。"""
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                ping_interval = _optional_float_env("WEBSOCKET_PING_INTERVAL")
                ping_timeout = _optional_float_env("WEBSOCKET_PING_TIMEOUT")
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    ping_interval=ping_interval,
                    ping_timeout=ping_timeout,
                )
                metadata = unpackb(conn.recv())    # 连上后服务端先发 metadata
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")   # 服务端没起,5秒后重试
                time.sleep(5)

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        """★ 核心:发观测 → 收动作。
        pack(观测) → send → recv → unpack(动作)。
        和服务端 _handler 的 recv→policy.infer→send 配对。"""
        data = self._packer.pack(obs)             # ① 打包观测
        self._ws.send(data)                       # ② 发给服务端
        response = self._ws.recv()                # ③ 收响应
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            # 收到字符串说明服务端出错了(发了 traceback)
            raise RuntimeError(f"Error in inference server:\n{response}")
        return unpackb(response)                  # ④ 解包成动作

    @override
    def reset(self, robo_name: str) -> None:
        """切换机器人:发一个 {reset=True, robo_name=...} 信号,让服务端重新装载配置。"""
        self.infer(dict(reset=True, robo_name=robo_name))

def _optional_float_env(name: str) -> Optional[float]:
    """从环境变量读 float(未设/None 返回 None)。配置心跳用。"""
    value = os.environ.get(name)
    if value is None or value.lower() == "none":
        return None
    return float(value)

if __name__ == "__main__":
    # ===== 演示:怎么用客户端发观测、收动作 =====
    policy_on_device = WebsocketClientPolicy(port=8000)   # 连本地 8000 端口的服务端
    import torch
    import numpy as np
    from PIL import Image
    from .image_tools import convert_to_uint8
    device = torch.device("cuda")

    # 造假观测(实际用真实相机/状态)
    base_0_rgb = np.random.randint(0, 256, size=(1, 3, 224, 224), dtype=np.uint8)
    left_wrist_0_rgb = np.random.randint(0, 256, size=(1, 3, 224, 224), dtype=np.uint8)
    state = np.random.rand(1,8).astype(np.float32)
    prompt = ["do something"]

    # observation = {
    #     "image": {
    #         "base_0_rgb": torch.from_numpy(base_0_rgb).to(device)[None],
    #         "left_wrist_0_rgb": torch.from_numpy(left_wrist_0_rgb).to(device)[None],
    #     },
    #     "state": torch.from_numpy(state).to(device)[None],
    #     "prompt": prompt,
    # }

    # ★ 观测三件套:图像(3相机)+ 状态 + 语言指令
    observation = {
        "image": {
            "base_0_rgb": convert_to_uint8(base_0_rgb),         # 顶部相机
            "left_wrist_0_rgb": convert_to_uint8(left_wrist_0_rgb),   # 左腕相机
            "right_wrist_0_rgb": convert_to_uint8(left_wrist_0_rgb),  # 右腕相机
        },
        "state": state,                                          # 当前关节状态
        "prompt": prompt,                                        # 任务指令
    }

    policy_on_device.infer(observation)     # ★ 发观测,收动作
    from IPython import embed;embed()
