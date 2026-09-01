"""启动朱顶红无菌实训评估台。

用法（在 yolov26 环境中）:
    python lab_web.py
    python lab_web.py --port 7861
    浏览器打开提示的地址
"""
from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def port_free(host: str, port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def pick_port(host: str, preferred: int) -> int:
    for port in range(preferred, preferred + 20):
        if port_free(host, port):
            return port
    raise SystemExit(f"从 {preferred} 起连续 20 个端口都被占用，请先关掉旧的评估台进程。")


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="朱顶红无菌实训评估台")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    port = pick_port(args.host, args.port)
    if port != args.port:
        print(f"端口 {args.port} 已被占用，改用 {port}")
    print(f"朱顶红无菌实训评估台  http://{args.host}:{port}")
    uvicorn.run("webui.server:app", host=args.host, port=port, reload=False)
