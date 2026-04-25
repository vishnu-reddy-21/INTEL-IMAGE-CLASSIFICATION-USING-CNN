from PIL import Image
import io

def read_image_from_bytes(data: bytes):
    return Image.open(io.BytesIO(data)).convert("RGB")