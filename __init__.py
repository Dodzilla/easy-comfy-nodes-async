import requests
import base64
import io
import os
import re
import threading
import tempfile
import urllib.parse
import numpy as np
from PIL import Image, ImageOps
import torch
import boto3
import comfy
import pillow_avif

from pillow_heif import register_heif_opener
register_heif_opener()

class HttpPostNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"url": ("STRING", {"default": ""}), "body": ("DICT",)}}
    RETURN_TYPES = ("INT", )
    RETURN_NAMES=("status_code",)
    FUNCTION = "execute"
    CATEGORY = "HTTP"
    OUTPUT_NODE=True

    def _perform_post(self, url, body):
        try:
            response = requests.post(url, json=body)
            print(f"Background POST to {url} completed. Status: {response.status_code}, Response: {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"Background POST to {url} failed: {e}")

    def execute(self, url, body):
        thread = threading.Thread(target=self._perform_post, args=(url, body))
        thread.daemon = True
        thread.start()
        print(f"HTTP POST request to {url} initiated in background.")
        return (202,)

class EmptyDictNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}
    RETURN_TYPES = ("DICT", )
    RETURN_NAMES=("dict",)
    FUNCTION = "execute"
    CATEGORY = "DICT"

    def execute(self):
        return ({},)

class AssocStrNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"dict": ("DICT",), "key": ("STRING", {"default": ""}), "value": ("STRING", {"default": ""})}}
    RETURN_TYPES = ("DICT", )
    RETURN_NAMES=("dict",)
    FUNCTION = "execute"
    CATEGORY = "DICT"

    def execute(self, dict, key, value):
        return ({**dict, key: value},)

class AssocDictNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"dict": ("DICT",), "key": ("STRING", {"default": ""}), "value": ("DICT", {"default": {}})}}
    RETURN_TYPES = ("DICT", )
    RETURN_NAMES=("dict",)
    FUNCTION = "execute"
    CATEGORY = "DICT"

    def execute(self, dict, key, value):
        return ({**dict, key: value},)

class AssocImgNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
                   "required": {
                       "dict": ("DICT",),
                       "key": ("STRING", {"default": ""}),
                       "value": ("IMAGE", {"default": ""}),
                   },
                   "optional": {
                       "format": ("STRING", {"default": "webp"}),
                       "quality": ("INT", {"default": 92})
                   }
               }
    RETURN_TYPES = ("DICT", )
    RETURN_NAMES=("dict",)
    FUNCTION = "execute"
    CATEGORY = "DICT"

    def execute(self, dict, key, value, format="webp", quality=92):
        image = Image.fromarray(np.clip(255. * value[0].cpu().numpy(), 0, 255).astype(np.uint8))
        buffered = io.BytesIO()
        image.save(buffered, format=format, quality=quality)
        img_bytestr =  base64.b64encode(buffered.getvalue())
        return ({**dict, key: (bytes(f'data:image/{format};base64,', encoding='utf-8') + img_bytestr).decode() },)

def loadImageFromUrl(url):
    # Lifted mostly from https://github.com/sipherxyz/comfyui-art-venture/blob/main/modules/nodes.py#L43
    if url.startswith("data:image/"):
        i = Image.open(io.BytesIO(base64.b64decode(url.split(",")[1])))
    elif url.startswith("s3://"):
        s3 = boto3.client('s3')
        bucket, key = url.split("s3://")[1].split("/", 1)
        obj = s3.get_object(Bucket=bucket, Key=key)
        i = Image.open(io.BytesIO(obj['Body'].read()))
    else:
        response = requests.get(url, timeout=5)
        if response.status_code != 200:
            raise Exception(response.text)

        i = Image.open(io.BytesIO(response.content))

    i = ImageOps.exif_transpose(i)

    if i.mode != "RGBA":
        i = i.convert("RGBA")

    # recreate image to fix weird RGB image
    alpha = i.split()[-1]
    image = Image.new("RGB", i.size, (0, 0, 0))
    image.paste(i, mask=alpha)

    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]
    if "A" in i.getbands():
        mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
        mask = 1.0 - torch.from_numpy(mask)
    else:
        mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

    return (image, mask)

class LoadImageFromUrlNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"url": ("STRING", {"default": ""})}}
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES=("image", "mask")
    FUNCTION = "execute"
    CATEGORY = "HTTP"

    def execute(self, url):
        return {"result": loadImageFromUrl(url)}

class LoadImagesFromUrlsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"urls": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False})}}
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES=("images",)
    FUNCTION = "execute"
    CATEGORY = "HTTP"

    def execute(self, urls):
        print(urls.split("\n"))
        images = [loadImageFromUrl(u)[0] for u in urls.split("\n")]
        firstImage = images[0]
        restImages = images[1:]
        if len(restImages) == 0:
            return (firstImage,)
        else:
            image1 = firstImage
            for image2 in restImages:
                if image1.shape[1:] != image2.shape[1:]:
                    image2 = comfy.utils.common_upscale(image2.movedim(-1, 1), image1.shape[2], image1.shape[1], "bilinear", "center").movedim(1, -1)
                image1 = torch.cat((image1, image2), dim=0)
            return (image1,)


def _get_comfy_input_directory():
    try:
        import folder_paths  # type: ignore

        return folder_paths.get_input_directory()
    except Exception:
        # Fallback for non-ComfyUI environments:
        # this repo is typically installed at ComfyUI/custom_nodes/easy-comfy-nodes-async
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "input"))


def _safe_relative_subdirectory(subdirectory):
    subdirectory = (subdirectory or "").strip()
    if subdirectory == "":
        return ""

    subdirectory = subdirectory.replace("\\", "/")
    drive, _ = os.path.splitdrive(subdirectory)
    if drive or subdirectory.startswith("/"):
        raise ValueError("subdirectory must be relative to the ComfyUI input directory")

    subdirectory = os.path.normpath(subdirectory)
    if subdirectory in (".", ""):
        return ""
    if subdirectory == ".." or subdirectory.startswith(f"..{os.sep}") or subdirectory.startswith("../"):
        raise ValueError("subdirectory must not traverse outside the ComfyUI input directory")

    return subdirectory


def _sanitize_filename(name):
    name = (name or "").strip().strip("\x00")
    if name == "":
        return None

    if (name.startswith('"') and name.endswith('"')) or (name.startswith("'") and name.endswith("'")):
        name = name[1:-1].strip()

    name = name.replace("\\", "/")
    name = os.path.basename(name).strip()
    if name in ("", ".", ".."):
        return None

    return name


def _filename_from_content_disposition(header_value):
    if not header_value:
        return None

    match = re.search(r"filename\\*=([^']*)''([^;]+)", header_value, flags=re.IGNORECASE)
    if match:
        encoding = match.group(1) or "utf-8"
        encoded_name = match.group(2)
        try:
            return urllib.parse.unquote(encoded_name, encoding=encoding, errors="replace")
        except Exception:
            return urllib.parse.unquote(encoded_name)

    match = re.search(r'filename="([^"]+)"', header_value, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()

    match = re.search(r"filename=([^;]+)", header_value, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip().strip('"')

    return None


class DownloadFilesToInputNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "urls": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False}),
            },
            "optional": {
                "filenames": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False}),
                "subdirectory": ("STRING", {"default": ""}),
                "overwrite": ("BOOLEAN", {"default": False}),
                "timeout_seconds": ("INT", {"default": 30, "min": 1, "max": 600}),
                "fail_on_error": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("saved_paths", "downloaded_count")
    FUNCTION = "execute"
    CATEGORY = "HTTP"
    OUTPUT_NODE = True

    def execute(
        self,
        urls,
        filenames="",
        subdirectory="",
        overwrite=False,
        timeout_seconds=30,
        fail_on_error=True,
    ):
        input_dir = _get_comfy_input_directory()
        subdir = _safe_relative_subdirectory(subdirectory)

        dest_dir = os.path.abspath(os.path.join(input_dir, subdir))
        input_dir_abs = os.path.abspath(input_dir)
        if os.path.commonpath([input_dir_abs, dest_dir]) != input_dir_abs:
            raise ValueError("Invalid subdirectory; path escapes the ComfyUI input directory")

        os.makedirs(dest_dir, exist_ok=True)

        url_lines = urls.splitlines()
        filename_lines = (filenames or "").splitlines()

        saved_paths = []
        downloaded_count = 0

        for i, raw_url in enumerate(url_lines):
            url = (raw_url or "").strip()
            if url == "" or url.startswith("#"):
                continue

            requested_name = ""
            if i < len(filename_lines):
                requested_name = (filename_lines[i] or "").strip()
            name = _sanitize_filename(requested_name)

            if name is None:
                if url.startswith("s3://"):
                    try:
                        _, rest = url.split("s3://", 1)
                        _, key = rest.split("/", 1)
                        name = _sanitize_filename(os.path.basename(key))
                    except ValueError:
                        name = None
                else:
                    parsed = urllib.parse.urlparse(url)
                    name = _sanitize_filename(urllib.parse.unquote(os.path.basename(parsed.path)))

            dest_name = name
            dest_path = os.path.join(dest_dir, dest_name) if dest_name else None

            try:
                if url.startswith("s3://"):
                    if dest_name is None:
                        dest_name = f"download_{i + 1}"
                        dest_path = os.path.join(dest_dir, dest_name)

                    if os.path.exists(dest_path) and not overwrite:
                        rel = os.path.relpath(dest_path, input_dir_abs).replace(os.sep, "/")
                        saved_paths.append(rel)
                        continue

                    _, rest = url.split("s3://", 1)
                    bucket, key = rest.split("/", 1)
                    s3 = boto3.client("s3")
                    with tempfile.NamedTemporaryFile(delete=False, dir=dest_dir) as tmp:
                        tmp_path = tmp.name
                    try:
                        s3.download_file(bucket, key, tmp_path)
                        os.replace(tmp_path, dest_path)
                    finally:
                        if os.path.exists(tmp_path):
                            try:
                                os.remove(tmp_path)
                            except OSError:
                                pass

                    downloaded_count += 1
                    rel = os.path.relpath(dest_path, input_dir_abs).replace(os.sep, "/")
                    saved_paths.append(rel)
                    continue

                if dest_name is not None and dest_path is not None and os.path.exists(dest_path) and not overwrite:
                    rel = os.path.relpath(dest_path, input_dir_abs).replace(os.sep, "/")
                    saved_paths.append(rel)
                    continue

                response = requests.get(url, stream=True, timeout=timeout_seconds)
                if response.status_code != 200:
                    raise Exception(f"Download failed ({response.status_code}): {response.text}")

                if dest_name is None:
                    cd_name = _sanitize_filename(_filename_from_content_disposition(response.headers.get("Content-Disposition")))
                    if cd_name:
                        dest_name = cd_name
                    else:
                        dest_name = f"download_{i + 1}"
                    dest_path = os.path.join(dest_dir, dest_name)

                if os.path.exists(dest_path) and not overwrite:
                    rel = os.path.relpath(dest_path, input_dir_abs).replace(os.sep, "/")
                    saved_paths.append(rel)
                    continue

                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, dir=dest_dir) as tmp:
                        tmp_path = tmp.name
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if not chunk:
                                continue
                            tmp.write(chunk)

                    os.replace(tmp_path, dest_path)
                finally:
                    response.close()
                    if tmp_path and os.path.exists(tmp_path):
                        try:
                            os.remove(tmp_path)
                        except OSError:
                            pass

                downloaded_count += 1
                rel = os.path.relpath(dest_path, input_dir_abs).replace(os.sep, "/")
                saved_paths.append(rel)
            except Exception as e:
                if fail_on_error:
                    raise
                print(f"Failed downloading {url}: {e}")

        return ("\n".join(saved_paths), downloaded_count)


class LoadLatentFromPathNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"latent_path": ("STRING", {"default": ""})}}

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "execute"
    CATEGORY = "Latent"

    def execute(self, latent_path):
        try:
            import nodes as comfy_nodes  # type: ignore
        except Exception as e:
            raise Exception("ComfyUI core module `nodes` was not found; this node must run inside ComfyUI.") from e

        if not hasattr(comfy_nodes, "LoadLatent"):
            raise Exception("ComfyUI core node `LoadLatent` was not found (nodes.LoadLatent).")

        loader = comfy_nodes.LoadLatent()
        preferred_fn = getattr(loader, "FUNCTION", None) or getattr(comfy_nodes.LoadLatent, "FUNCTION", None)
        candidate_fns = []
        if isinstance(preferred_fn, str):
            candidate_fns.append(preferred_fn)
        candidate_fns.extend(["load", "load_latent", "execute"])

        for fn_name in candidate_fns:
            fn = getattr(loader, fn_name, None)
            if callable(fn):
                return fn(latent_path)

        raise Exception("Unable to call ComfyUI core LoadLatent implementation.")

class S3Upload:
    """
    Uploads first file from VHS_FILENAMES from ComfyUI-VideoHelperSuite to S3.

    See also: https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "filenames": ("VHS_FILENAMES",),
                "s3_bucket": ("STRING", {"default": ""}),
                "s3_object_name": ("STRING", {"default": "default/result.webp"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("s3_url",)
    OUTPUT_NODE = True
    CATEGORY = "Video"
    FUNCTION = "execute"

    def execute(
        self,
        filenames=(),
        s3_bucket="",
        s3_object_name="",
    ):
        s3 = boto3.resource('s3')
        s3.Bucket(s3_bucket).upload_file(filenames[1][1], s3_object_name)
        s3url = f's3://{s3_bucket}/{s3_object_name}'
        print(f'Uploading file to {s3url}')
        return (s3url,)

NODE_CLASS_MAPPINGS = {
    "EZHttpPostNode": HttpPostNode,
    "EZEmptyDictNode": EmptyDictNode,
    "EZAssocStrNode": AssocStrNode,
    "EZAssocDictNode": AssocDictNode,
    "EZAssocImgNode": AssocImgNode,
    "EZLoadImgFromUrlNode": LoadImageFromUrlNode,
    "EZLoadImgBatchFromUrlsNode": LoadImagesFromUrlsNode,
    "EZDownloadFilesToInputNode": DownloadFilesToInputNode,
    "EZLoadLatentFromPathNode": LoadLatentFromPathNode,
    "EZS3Uploader": S3Upload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EZHttpPostNode": "HTTP POST",
    "EZEmptyDictNode": "Empty Dict",
    "EZAssocStrNode": "Assoc Str",
    "EZAssocDictNode": "Assoc Dict",
    "EZAssocImgNode": "Assoc Img",
    "EZLoadImgFromUrlNode": "Load Img From URL (EZ)",
    "EZLoadImgBatchFromUrlsNode": "Load Img Batch From URLs (EZ)",
    "EZDownloadFilesToInputNode": "Download Files To Input (EZ)",
    "EZLoadLatentFromPathNode": "Load Latent From Path (EZ)",
    "EZS3Uploader": "S3 Upload (EZ)",
}
