import asyncio
import subprocess
from mcp.server.fastmcp import FastMCP

# 创建一个名为 GPU Monitor 的 MCP Server
mcp = FastMCP("GPU Monitor")

@mcp.tool()
def get_gpu_status() -> str:
    """获取当前系统的 nvidia-smi 状态，包括 2 张 GTX 1060 的显存余量"""
    try:
        result = subprocess.run(
            ["nvidia-smi"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        return result.stdout
    except Exception as e:
        return f"获取 GPU 状态失败: {str(e)}"

if __name__ == "__main__":
    mcp.run_stdio_async()
