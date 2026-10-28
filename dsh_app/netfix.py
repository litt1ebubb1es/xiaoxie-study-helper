"""第三方 HTTP 栈的兼容补丁（只补一处，见下）。

**为什么需要这个文件**

这个应用是「base 解释器 + 项目 venv」的混合环境：双击 `小蟹学习助手.exe` 跑的是
base 的 `python.exe`，`run_app.py` 再把 `.venv\\Lib\\site-packages` 补到 `sys.path`
**末尾**（见该文件的注释）。于是同一个模块可能来自两个地方——venv 里有、base 里也有时，
**base 的先被 import**。

踩到的组合：

- venv 里的 `httpx2` 2.x 解 `Content-Encoding: br` 时调
  `decompressor.process(data, output_buffer_limit=...)`
- base 里的 `brotli` 1.1.0 是 C 实现，`process()` **不认这个关键字**（1.2 才有）

结果：只要网关用 br 压缩响应，请求就抛
`TypeError: process() takes no keyword arguments`。用户看到的是
「模型列表失败：TypeError: process() takes no keyword arguments」——
一句和"模型"毫无关系的报错，而且只在某些网关上出现，非常难查。

**修法**：不靠猜版本号，直接**试一次**那个关键字（能力探测）。不支持就把 `br` 从
httpx2 的解码器表和 `Accept-Encoding` 里摘掉，服务器改发 gzip/deflate——功能完全不受
影响，只是少了 br 那点压缩率。用户不用装任何东西。
"""

from __future__ import annotations


def _brotli_supports_output_limit() -> bool:
    """brotli 的 process() 认不认 output_buffer_limit。真试一次，不看版本号。"""
    try:
        import brotli
    except Exception:                       # noqa: BLE001
        return True                         # 没装 brotli：不涉及这个问题
    try:
        brotli.Decompressor().process(b"", output_buffer_limit=1024)
        return True
    except TypeError:
        return False                        # 就是它：C 签名不认关键字
    except Exception:                       # noqa: BLE001
        return True                         # 别的报错（状态/数据）不算签名不兼容


def patch_http_encodings() -> str:
    """摘掉当前环境里不可用的 `br` 编码。返回一句人话（什么都没做就返回空串）。"""
    try:
        from httpx2 import _client as hx_client
        from httpx2 import _decoders as hx_decoders
    except Exception:                       # noqa: BLE001
        return ""                           # 没装 httpx2：不归这里管
    table = getattr(hx_decoders, "SUPPORTED_DECODERS", None)
    if not isinstance(table, dict) or "br" not in table:
        return ""
    if _brotli_supports_output_limit():
        return ""
    table.pop("br", None)
    # Accept-Encoding 是模块级字符串，请求时才拼进头里，改这里就生效
    keys = [k for k in table if k != "identity"]
    try:
        hx_client.ACCEPT_ENCODING = ", ".join(keys)
    except Exception:                       # noqa: BLE001
        pass
    return (f"当前 brotli 不支持 output_buffer_limit（httpx2 需要 >=1.2），"
            f"已改用 {'/'.join(keys)} 压缩；否则 br 响应会报 "
            f"TypeError: process() takes no keyword arguments")