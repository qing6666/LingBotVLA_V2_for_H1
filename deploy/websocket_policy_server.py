# -*- coding: utf-8 -*-
"""
websocket_policy_server.py —— 把推理策略包装成 websocket 网络服务
====================================================================
【作用】机器人(或仿真器)作为客户端,通过网络把观测发给本服务;本服务调
       policy.infer(观测) 算出动作,再通过网络把动作发回去。

【通信协议】websocket + msgpack 序列化(msgpack 对 numpy 数组友好,比 JSON 高效)。

【核心流程(_handler)】
  连接建立 → 发 metadata
  循环:
    recv 观测(msgpack 解包)→ policy.infer(推理)→ send 动作(msgpack 打包)
  直到连接关闭。

【在项目中的位置】
  lingbot_vla_v2_policy.py 的 main() 建 WebsocketPolicyServer(model, port).serve_forever()
  机器人端用 websocket_client_policy.py 连上来。
====================================================================
"""
import asyncio
import http
import logging
import os
import time
import traceback

from .msgpack_numpy import Packer, unpackb     # msgpack 序列化(numpy 友好)
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    把推理策略(policy)包装成 websocket 服务,供远程客户端(机器人)调用。"""

    def __init__(
        self,
        policy,                         # 推理策略(LingbotVLAv2Server,含 infer 方法)
        host: str = "0.0.0.0",           # 监听地址(0.0.0.0 = 所有网卡)
        port: int | None = None,        # 监听端口
        metadata: dict | None = None,   # 连接建立时发给客户端的元信息
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        """启动服务(阻塞)。用 asyncio 跑事件循环。"""
        asyncio.run(self.run())

    async def run(self):
        """用 websockets 库启动服务,每个客户端连接交给 _handler 处理。"""
        async with _server.serve(
            self._handler,               # 连接处理函数
            self._host,
            self._port,
            compression=None,            # 关闭压缩(图像数据已紧凑,压缩反而耗 CPU)
            max_size=None,               # 不限消息大小(图像观测可能较大)
            ping_interval=_optional_float_env("WEBSOCKET_PING_INTERVAL"),  # 心跳(保活)
            ping_timeout=_optional_float_env("WEBSOCKET_PING_TIMEOUT"),
            process_request=_health_check,   # /healthz 健康检查
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        """★ 核心:每个客户端连接的处理循环。
        收观测 → policy.infer → 发动作,循环到连接关闭。"""
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = Packer()                # msgpack 打包器

        # 连接建立:先发 metadata(告诉客户端服务端信息)
        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:                      # 主循环:一发一收
            try:
                start_time = time.monotonic()
                obs = unpackb(await websocket.recv())   # ★ 1) 收观测(msgpack 解包)

                infer_time = time.monotonic()
                action = self._policy.infer(obs)        # ★ 2) 调推理策略算动作
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,      # 记录推理耗时(监控用)
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))   # ★ 3) 发动作(msgpack 打包)
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:        # 客户端断开:正常退出
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:                          # 出错:把 traceback 发给客户端,再关闭
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    """健康检查:访问 /healthz 返回 OK(供监控/负载均衡探活)。"""
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None


def _optional_float_env(name: str) -> float | None:
    """从环境变量读 float(未设/None 则返回 None)。用于配置心跳参数。"""
    value = os.environ.get(name)
    if value is None or value.lower() == "none":
        return None
    return float(value)
