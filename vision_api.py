from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

from PIL import Image


_DEFAULT_PROMPT = "详细描述这张图片的内容"


def _load_image(image_input: Any) -> Image.Image:
    if isinstance(image_input, Image.Image):
        img = image_input
    elif isinstance(image_input, (str, Path)):
        img = Image.open(image_input)
    else:
        raise TypeError("image_input must be str/Path or PIL.Image.Image")
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    return img


def _downscale(img: Image.Image, max_pixels: int) -> Image.Image:
    if max_pixels <= 0:
        raise ValueError("max_pixels must be > 0")
    pixels = img.width * img.height
    if pixels <= max_pixels:
        return img
    scale = (max_pixels / float(pixels)) ** 0.5
    new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
    return img.resize(new_size, Image.LANCZOS)


def _image_to_png_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _pick_cfg():
    import mykey as mk  # 只运行时引用，不主动读取密钥文件内容

    last_err = None
    for name in ("claude_config141", "native_claude_config2", "native_claude_config84", "native_claude_config5535", "oai_config"):
        try:
            cfg = getattr(mk, name)
            if isinstance(cfg, dict):
                return dict(cfg)
        except Exception as e:
            last_err = e
    if last_err:
        raise RuntimeError("no usable config found in mykey (tried claude_config141/native_claude_config2/84/5535/oai_config)") from last_err
    raise RuntimeError("no usable config found in mykey")


def _make_session(timeout: int):
    from llmcore import LLMSession, NativeClaudeSession

    cfg = _pick_cfg()
    cfg["timeout"] = timeout
    cfg["read_timeout"] = max(timeout, int(cfg.get("read_timeout", timeout)))
    api_base = str(cfg.get("apibase", "")).lower()
    model = str(cfg.get("model", "")).lower()
    if "anthropic" in api_base or "claude" in model or cfg.get("api_format") == "claude-native":
        sess = NativeClaudeSession(cfg)
        sess.stream = False
        return sess, "native_claude"
    return LLMSession(cfg), "oai"


def _collect_text_from_gen(gen):
    parts = []
    content_blocks = []
    try:
        while True:
            parts.append(next(gen))
    except StopIteration as e:
        content_blocks = e.value or []
    text = "".join(parts).strip()
    if not text and content_blocks:
        text = "\n".join(
            b.get("text", "") for b in content_blocks
            if isinstance(b, dict) and b.get("type") == "text"
        ).strip()
    return text


def ask_vision(image_input, prompt: str | None = None, timeout: int = 60, max_pixels: int = 1_440_000) -> str:
    """
    调用 vision 能力分析图片。
    :param image_input: 文件路径(str/Path) 或 PIL Image 对象
    :param prompt: 提示词，默认“详细描述这张图片的内容”
    :param timeout: 超时秒数
    :param max_pixels: 最大像素数，超出自动缩放
    :return: str，成功为模型回复，失败为 Error: ...
    """
    try:
        img = _downscale(_load_image(image_input), max_pixels=max_pixels)
        b64 = _image_to_png_base64(img)
        prompt = prompt or _DEFAULT_PROMPT
        session, mode = _make_session(timeout)

        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
            ],
        }

        if mode == "native_claude":
            resp = session.ask(msg)
            return (resp.content or "").strip() or "Error: empty response"

        from llmcore import _msgs_claude2oai

        messages = _msgs_claude2oai([msg])
        text = _collect_text_from_gen(session.raw_ask(messages))
        return text or "Error: empty response"
    except Exception as e:
        return f"Error: {e}"