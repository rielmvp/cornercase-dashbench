"""OpenAI-compatible chat-completions client used for every model in DashBench.

It works with the OpenAI API (e.g. GPT-5), OpenRouter (e.g. Gemini-3-Flash) and any local server that speaks the
OpenAI protocol (e.g. `vllm serve`). Frames are sent either as one image_url per frame or as a single video_url.
"""
import base64
import os
import threading
import time

from openai import OpenAI

thread_local = threading.local()

# Models that only accept the default sampling temperature
NO_TEMPERATURE_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_client(api_key, base_url):
    """One client per thread."""
    if not hasattr(thread_local, "client"):
        thread_local.client = OpenAI(api_key=api_key, base_url=base_url, timeout=None)
    return thread_local.client


def supports_temperature(model):
    name = model.split("/")[-1].lower()
    return not name.startswith(NO_TEMPERATURE_PREFIXES)


def build_request(model, prompt, image_paths, temperature=0.0, video_path=None, guided_json=None,
                  system_prompt=None, system_in_user=False, disable_thinking=False, max_tokens=None):
    """Build the chat-completions parameters (no network access)."""
    if video_path:
        # One video_url entry. InternVL needs an explicit <video> placeholder in the text.
        text = f"<video>\n{prompt}" if "internvl" in model.lower() else prompt
        content = [{"type": "text", "text": text},
                   {"type": "video_url", "video_url": {"url": f"file://{os.path.abspath(video_path)}"}}]
    else:
        content = [{"type": "text", "text": prompt}]
        for path in image_paths:
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encode_image_to_base64(path)}"}})

    messages = []
    if system_prompt and system_in_user:
        # For chat templates that cannot combine a system role with list-typed user content (e.g. InternVL3)
        content = [{"type": "text", "text": f"{system_prompt}\n\n{content[0]['text']}"}] + content[1:]
    elif system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})

    params = {"model": model, "messages": messages}
    if supports_temperature(model):
        params["temperature"] = temperature
    extra_body = {}
    if guided_json:
        extra_body["guided_json"] = guided_json
    if disable_thinking:
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}
    if extra_body:
        params["extra_body"] = extra_body
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    return params


def make_api_call(api_key, base_url, **request_kwargs):
    """Send one request, retrying with exponential backoff (30 s, 60 s)."""
    params = build_request(**request_kwargs)
    client = get_client(api_key, base_url)
    for attempt in range(3):
        try:
            return client.chat.completions.create(**params).model_dump()
        except Exception as e:
            print(f"API attempt {attempt + 1} failed: {e}")
            if attempt == 2:
                raise
            time.sleep((2 ** attempt) * 30)


def model_folder_name(model):
    """Safe folder name for a model id such as 'Qwen/Qwen3-VL-8B-Instruct'."""
    return model.replace("/", "_")
