import logging
import os
from transformers import VFusion3DModel, VFusion3DTokenizer

os.environ["TRANSFORMERS_CACHE"] = "/mnt/d/transformers_cache"

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Load the tokenizer and the model
tokenizer = VFusion3DTokenizer.from_pretrained("facebook/vfusion3d")
model = VFusion3DModel.from_pretrained("facebook/vfusion3d")

from PIL import Image
import requests
from io import BytesIO

# Load an image from a URL
image_path = "/mnt/e/game_sprites/1c311cb8-72b2-494b-8727-ce6786c5bb24.jpg"
input_image = Image.open(image_path)
# Prepare the image for the model
inputs = tokenizer(images=input_image, return_tensors="pt")
