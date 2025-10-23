from pathlib import Path
import json
import asyncio
import aiohttp
from fastapi import FastAPI, HTTPException, Response  # 增加 Response
import uuid

# ----------------- 配置 -----------------
COMFY_HOST = "127.0.0.1"
COMFY_PORT = 8188
BASE = f"http://{COMFY_HOST}:{COMFY_PORT}"
WORKFLOW_PATH = Path(__file__).with_name("workflow.json")
# ---------------------------------------

app = FastAPI(title="ComfyUI 文生图 API", version="1.0")

with WORKFLOW_PATH.open(encoding="utf-8") as f:
    WORKFLOW_TEMPLATE = json.load(f)

def build_workflow(prompt: str) -> dict:
    """
    Builds a ComfyUI workflow from the template with the given prompt.

    Args:
        prompt: The text prompt to be used in the workflow.

    Returns:
        A dictionary representing the ComfyUI workflow.
    """
    wf = json.loads(json.dumps(WORKFLOW_TEMPLATE))
    wf["6"]["inputs"]["text"] = prompt
    return wf

async def wait_for_image_meta(prompt_id: str, timeout: int = 120) -> dict:
    """
    Polls the /history/{prompt_id} endpoint until the first output image metadata is available.

    Args:
        prompt_id: The ID of the prompt to wait for.
        timeout: The maximum time to wait in seconds.

    Returns:
        A dictionary containing the metadata of the first output image,
        e.g., {"filename": "...", "subfolder": "...", "type": "output"}.

    Raises:
        HTTPException: If the ComfyUI server times out.
    """
    url = f"{BASE}/history/{prompt_id}"
    async with aiohttp.ClientSession() as session:
        for _ in range(timeout):
            async with session.get(url) as resp:
                if resp.status != 200:
                    await asyncio.sleep(1)
                    continue
                h = await resp.json()
                if not h or prompt_id not in h:
                    await asyncio.sleep(1)
                    continue
                outputs = h[prompt_id].get("outputs", {})
                for node in outputs.values():
                    for img in node.get("images", []):
                        return {
                            "filename": img["filename"],
                            "subfolder": img.get("subfolder", ""),
                            "type": img.get("type", "output"),
                        }
            await asyncio.sleep(1)
    raise HTTPException(502, "ComfyUI 超时，未拿到输出图片")

async def download_image_bytes(meta: dict) -> bytes:
    """
    Downloads the image bytes from the /view endpoint.

    Args:
        meta: A dictionary containing the image metadata.

    Returns:
        The image data as bytes.

    Raises:
        HTTPException: If the image download fails.
    """
    params = {
        "filename": meta["filename"],
        "subfolder": meta.get("subfolder", ""),
        "type": meta.get("type", "output"),
    }
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{BASE}/view", params=params) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise HTTPException(502, f"下载图片失败: {text}")
            return await resp.read()

@app.post("/txt2img", summary="Text to Image")
async def txt2img(prompt: str):
    """
    Generates an image from a text prompt.

    Args:
        prompt: The text prompt to generate the image from.

    Returns:
        A Response object containing the generated image in PNG format.

    Raises:
        HTTPException: If the ComfyUI server fails to process the request.
    """
    client_id = str(uuid.uuid4())
    workflow = build_workflow(prompt)

    # 1) 提交任务
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{BASE}/prompt", json={"prompt": workflow, "client_id": client_id}) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise HTTPException(502, f"ComfyUI 提交失败: {text}")
            data = await resp.json()
            prompt_id = data["prompt_id"]

    # 2) 等待完成（注意改成按 prompt_id 轮询）
    meta = await wait_for_image_meta(prompt_id)

    # 3) 通过 /view 拉取图片并直接返回
    img_bytes = await download_image_bytes(meta)
    return Response(
        content=img_bytes,
        media_type="image/png",
        headers={"Content-Disposition": f'inline; filename="{meta["filename"]}"'},
    )

@app.get("/")
def root():
    """
    Root endpoint that returns a welcome message.

    Returns:
        A dictionary with a welcome message.
    """
    return {"message": "ComfyUI FastAPI wrapper is running"}