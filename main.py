from pathlib import Path
import json
import asyncio
import aiohttp
from fastapi import FastAPI, HTTPException, Response, UploadFile, File
import uuid

# ----------------- 配置 -----------------
COMFY_HOST = "127.0.0.1"
COMFY_PORT = 8188
BASE = f"http://{COMFY_HOST}:{COMFY_PORT}"
WORKFLOW_PATH = Path(__file__).with_name("flux_canny_model_v1.json")
DEFAULT_SYSTEM_PROMPT = "You are JADE-CAD v6, an expert AI jade-design engine trained on 2.3M annotated carving blueprints, 480k historical rubbings, 52k gemmological reports, and 8k CNC tool-path datasets. You output only manufacturable, culture-accurate, cost-aware jade carving designs that honor traditional Chinese jade art while meeting modern production standards. Specialize in classical motifs, auspicious symbols, and technical precision. Based on your expertise, please create a design that precisely matches the following user requirements:"# ---------------------------------------

app = FastAPI(title="ComfyUI Flux Canny API", version="1.0")

with WORKFLOW_PATH.open(encoding="utf-8") as f:
    WORKFLOW_TEMPLATE = json.load(f)

def build_workflow(prompt: str, image_filename: str = None, system_prompt: str = None) -> dict:
    """
    构建 Flux Canny 工作流
    
    Args:
        prompt: 用户文本提示词
        image_filename: 输入图像文件名（可选）
        system_prompt: 系统提示词（可选）
    
    Returns:
        工作流字典
    """
    wf = json.loads(json.dumps(WORKFLOW_TEMPLATE))
    
    # 组合系统提示词和用户提示词
    if system_prompt:
        combined_prompt = f"{system_prompt}\n\n{prompt}"
    else:
        combined_prompt = f"{DEFAULT_SYSTEM_PROMPT}\n\n{prompt}"
    
    # 更新节点23的文本提示词
    wf["23"]["inputs"]["text"] = combined_prompt
    
    # 如果提供了图像文件名，更新节点17的输入图像
    if image_filename:
        wf["17"]["inputs"]["image"] = image_filename
    
    return wf

async def upload_image_to_comfyui(image_bytes: bytes, filename: str) -> str:
    """
    上传图像到 ComfyUI 服务器
    """
    # 验证文件类型
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    file_ext = Path(filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(400, f"不支持的文件类型: {file_ext}")
    
    data = aiohttp.FormData()
    data.add_field('image', image_bytes, filename=filename, content_type='image/jpeg')
    
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{BASE}/upload/image", data=data) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise HTTPException(502, f"上传图像失败: {text}")
            result = await resp.json()
            return result.get('name', filename)

async def wait_for_image_meta(prompt_id: str, timeout: int = 120) -> dict:
    """
    等待图像生成完成并获取元数据
    优先返回 SaveImage 节点（节点9）的输出
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
                
                # 优先查找 SaveImage 节点（节点9）的输出
                if "9" in outputs:
                    node_output = outputs["9"]
                    for img in node_output.get("images", []):
                        return {
                            "filename": img["filename"],
                            "subfolder": img.get("subfolder", ""),
                            "type": img.get("type", "output"),
                        }
                
                # 如果没有找到 SaveImage 节点，则查找其他图像输出节点
                for node_id, node_output in outputs.items():
                    for img in node_output.get("images", []):
                        return {
                            "filename": img["filename"],
                            "subfolder": img.get("subfolder", ""),
                            "type": img.get("type", "output"),
                        }
            await asyncio.sleep(1)
    raise HTTPException(502, "ComfyUI 超时，未拿到输出图片")

async def download_image_bytes(meta: dict) -> bytes:
    """
    下载生成的图像
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

@app.post("/txt2img", summary="Text to Image with Flux")
async def txt2img(prompt: str, system_prompt: str = None):
    """
    使用 Flux 模型生成图像
    
    Args:
        prompt: 用户文本提示词
        system_prompt: 系统提示词（可选）
    
    Returns:
        生成的图像
    """
    client_id = str(uuid.uuid4())
    workflow = build_workflow(prompt, system_prompt=system_prompt)

    async with aiohttp.ClientSession() as session:
        async with session.post(f"{BASE}/prompt", json={"prompt": workflow, "client_id": client_id}) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise HTTPException(502, f"ComfyUI 提交失败: {text}")
            data = await resp.json()
            prompt_id = data["prompt_id"]

    meta = await wait_for_image_meta(prompt_id)
    img_bytes = await download_image_bytes(meta)
    
    return Response(
        content=img_bytes,
        media_type="image/png",
        headers={"Content-Disposition": f'inline; filename="{meta["filename"]}"'},
    )

@app.post("/img2img", summary="Image to Image with Flux Canny")
async def img2img(prompt: str, image: UploadFile = File(...), system_prompt: str = None):
    # 验证文件类型
    if not image.content_type.startswith('image/'):
        raise HTTPException(400, "文件必须是图像类型")
    
    # 读取上传的图像
    image_bytes = await image.read()
    
    # 确保文件名有正确的扩展名
    if not image.filename or '.' not in image.filename:
        raise HTTPException(400, "文件名必须包含扩展名")
    
    # 上传到 ComfyUI
    uploaded_filename = await upload_image_to_comfyui(image_bytes, image.filename)
    
    client_id = str(uuid.uuid4())
    workflow = build_workflow(prompt, uploaded_filename, system_prompt)

    async with aiohttp.ClientSession() as session:
        async with session.post(f"{BASE}/prompt", json={"prompt": workflow, "client_id": client_id}) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise HTTPException(502, f"ComfyUI 提交失败: {text}")
            data = await resp.json()
            prompt_id = data["prompt_id"]

    meta = await wait_for_image_meta(prompt_id)
    img_bytes = await download_image_bytes(meta)
    
    return Response(
        content=img_bytes,
        media_type="image/png",
        headers={"Content-Disposition": f'inline; filename="{meta["filename"]}"'},
    )

@app.get("/")
def root():
    """
    根端点
    """
    return {"message": "ComfyUI Flux Canny API is running"}