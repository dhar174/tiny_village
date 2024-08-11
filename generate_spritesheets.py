import logging
import os
import numpy as np
import torch
from PIL import Image, ImageChops, ImageTk
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    AutoPipelineForText2Image,
    AutoPipelineForImage2Image,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
)
import torchvision.transforms as transforms
from torchvision.models import vgg19
import torch.nn as nn
import torch.optim as optim
import tkinter as tk
from tkinter import filedialog, messagebox
from threading import Lock

model_lock = Lock()

import concurrent.futures

os.environ["TRANSFORMERS_CACHE"] = "/mnt/d/transformers_cache"

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


# Load the VGG19 model for style transfer
def load_vgg():
    vgg = vgg19(pretrained=True).features
    for param in vgg.parameters():
        param.requires_grad = False
    return vgg


# Helper functions for style transfer
def get_features(image, model):
    layers = {
        "0": "conv1_1",
        "5": "conv2_1",
        "10": "conv3_1",
        "19": "conv4_1",
        "28": "conv5_1",
    }
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features


def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram


# Style transfer function
def transfer_style(
    content_image,
    style_image,
    model,
    num_steps=300,
    style_weight=1000000,
    content_weight=1,
):
    preprocess = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    content_img = preprocess(content_image).unsqueeze(0)
    style_img = preprocess(style_image).unsqueeze(0)

    content_img = content_img.to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    style_img = style_img.to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    target = content_img.clone().requires_grad_(True)

    optimizer = optim.Adam([target], lr=0.003)
    mse_loss = nn.MSELoss()

    style_features = get_features(style_img, model)
    content_features = get_features(content_img, model)
    style_grams = {
        layer: gram_matrix(style_features[layer]) for layer in style_features
    }

    for step in range(num_steps):
        target_features = get_features(target, model)
        content_loss = mse_loss(target_features["conv4_2"], content_features["conv4_2"])

        style_loss = 0
        for layer in style_features:
            target_gram = gram_matrix(target_features[layer])
            style_gram = style_grams[layer]
            layer_style_loss = mse_loss(target_gram, style_gram)
            style_loss += layer_style_loss / target_features[layer].nelement()

        total_loss = content_weight * content_loss + style_weight * style_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    target = target.cpu().clone().detach().squeeze(0)
    target = transforms.ToPILImage()(target)
    return target


# GUI creation
def create_gui(config):
    root = tk.Tk()
    root.title("Sprite Sheet Generator")
    preview_label = tk.Label(root)
    preview_label.pack()
    style_label = tk.Label(root, text="Style Image Path:")
    style_label.pack()
    style_entry = tk.Entry(root, width=50)
    style_entry.pack()

    def update_preview(image):
        img = ImageTk.PhotoImage(image)
        logging.info(f"Updating preview with image: {image.size}")
        preview_label.config(image=img)
        logging.info(f"Updated preview with image: {img.width()}x{img.height()}")
        preview_label.image = img
        logging.info(
            f"Updated preview with image: {preview_label.image.width()}x{preview_label.image.height()}"
        )
        preview_label.image_data = image  # Store the original image
        # preview_label.pack()

    def on_generate_button_click():
        logging.info("Generate button clicked.")
        prompt = prompt_entry.get()
        style_image_path = style_entry.get()
        # messagebox.showinfo(
        #     "Generation Started",
        #     "Sprite sheet generation has started in the background.",
        # )
        # async_generate_sprites(config, prompt, style_image_path, update_preview)
        generate_sprites(
            config, prompt, style_image_path, update_preview, modifications_entry.get()
        )
        logging.info("Generation started.")

    def on_save_button_click():
        logging.info("Saving image...")
        if hasattr(preview_label, "image_data"):
            logging.info("Image data found.")
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png", filetypes=[("PNG files", "*.png")]
            )
            if file_path:
                preview_label.image_data.save(file_path)
                messagebox.showinfo(
                    "Image Saved", f"Image has been saved to {file_path}"
                )
        else:
            logging.warning(f"No image data found. Preview label: {preview_label}")
            messagebox.showwarning("No Image", "There is no image to save.")

    # Input fields and buttons
    prompt_label = tk.Label(root, text="Base Prompt:")
    prompt_label.pack()
    prompt_entry = tk.Entry(root, width=50)
    prompt_entry.pack()

    modifications_label = tk.Label(root, text="Modifications:")
    modifications_label.pack()
    modifications_entry = tk.Entry(root, width=50)
    modifications_entry.pack()

    browse_button = tk.Button(
        root,
        text="Browse",
        command=lambda: style_entry.insert(0, filedialog.askopenfilename()),
    )
    browse_button.pack()

    generate_button = tk.Button(root, text="Generate", command=on_generate_button_click)
    generate_button.pack()
    save_button = tk.Button(root, text="Save", command=on_save_button_click)
    save_button.pack()

    root.mainloop()


# Advanced settings for GUI
def create_advanced_settings(root):
    settings_window = tk.Toplevel(root)
    settings_window.title("Advanced Settings")

    steps_label = tk.Label(settings_window, text="Number of Inference Steps:")
    steps_label.pack()
    steps_entry = tk.Entry(settings_window, width=10)
    steps_entry.pack()

    style_weight_label = tk.Label(settings_window, text="Style Weight:")
    style_weight_label.pack()
    style_weight_entry = tk.Entry(settings_window, width=10)
    style_weight_entry.pack()

    apply_button = tk.Button(
        settings_window,
        text="Apply",
        command=lambda: apply_settings(steps_entry.get(), style_weight_entry.get()),
    )
    apply_button.pack()


def apply_settings(steps, style_weight):
    global inference_steps, style_transfer_weight
    inference_steps = int(steps)
    style_transfer_weight = float(style_weight)


# Frame generation and sprite sheet creation
def generate_frames_from_embedding(
    pipeline, base_embedding, prompts, modifications, style_image=None, num_steps=50
):
    frames = []
    logging.info(
        f"Generating frames for {prompts} with modifications: {modifications} with base embedding of type {type(base_embedding)}"
    )
    i = 0
    logging.info(f"Scheduler used by pipeline: {pipeline.scheduler}")
    for mod in modifications:
        i += 1
        with torch.no_grad():
            prompt = f"{prompts} {mod}"
            logging.info(f"Generating frame for prompt: {prompt}")
            if type(base_embedding) == torch.Tensor:
                image = pipeline(
                    prompt=prompt,
                    num_inference_steps=num_steps,
                    latents=base_embedding,
                )[0]
            elif type(base_embedding) == Image.Image:
                image = pipeline(
                    prompt=prompt,
                    image=base_embedding,
                    num_inference_steps=num_steps,
                )[0]
            elif type(base_embedding) == np.ndarray:
                image = (
                    pipeline(
                        prompt=prompt,
                        image=Image.fromarray(base_embedding),
                        num_inference_steps=num_steps,
                    )[0]
                    if pipeline.__class__.__name__ == "StableDiffusionImg2ImgPipeline"
                    else pipeline(
                        prompt=prompt,
                        latents=base_embedding,
                        num_inference_steps=num_steps,
                    )[0]
                )
            if style_image:
                image = transfer_style(
                    content_image=image, style_image=style_image, model=load_vgg()
                )
            image[0].show()
            frames.append(image[0])
    logging.info(f"Frames generated: {len(frames)}")
    logging.info(f"Frames: {frames}")
    return frames


def interpolate_frames(frame1, frame2, steps=5):
    frames = [frame1]
    for i in range(1, steps):
        alpha = i / steps
        interpolated_frame = ImageChops.blend(frame1, frame2, alpha)
        frames.append(interpolated_frame)
    frames.append(frame2)
    return frames


# Configuration class to manage dependencies
class Config:
    def __init__(self, model=None, inference_steps=100, style_weight=1.0):
        self.model = model or self.load_model()
        self.inference_steps = inference_steps
        self.style_weight = style_weight
        self.img2img = lambda: self.load_img2img()
        # self.model.scheduler = DDIMScheduler.from_config(
        #     self.model.scheduler.config,
        #     rescale_betas_zero_snr=True,
        #     timestep_spacing="trailing",
        # )

    @staticmethod
    def load_model():
        pipeline = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            cache_dir="/mnt/d/transformers_cache",
            offload_folder="/mnt/d/transformers_cache",
            device_map="balanced",
        )
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config,
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
        )
        assert torch.cuda.is_available(), "CUDA must be available for model loading"
        # pipeline = pipeline.to("cuda" if torch.cuda.is_available() else "cpu")
        return pipeline

    def load_img2img(self):
        pipeline = StableDiffusionImg2ImgPipeline(**self.model.components)
        assert torch.cuda.is_available(), "CUDA must be available for model loading"
        return pipeline


# Frame generation and sprite sheet creation
def create_sprite_sheet(frames, rows, cols, padding=0):
    frame_width, frame_height = frames[0].size
    sprite_sheet_width = (frame_width + padding) * cols - padding
    sprite_sheet_height = (frame_height + padding) * rows - padding
    sprite_sheet = Image.new("RGBA", (sprite_sheet_width, sprite_sheet_height))

    for index, frame in enumerate(frames):
        x = (index % cols) * (frame_width + padding)
        y = (index // cols) * (frame_height + padding)
        sprite_sheet.paste(frame, (x, y))
    logging.info(f"Sprite sheet created: {sprite_sheet.size}")
    return sprite_sheet


def generate_sprites(config, prompt, style_image_path, update_callback, modifications):
    with model_lock:
        if not config.model:
            logging.info("Loading model...")
            config.model = config.load_model()
        if style_image_path:
            style_image = Image.open(style_image_path)

        model = config.model
        base_embedding = generate_base_embedding(
            model, prompt, style_image if style_image_path else None
        )

        # Example logic for sprite generation
        logging.info("Generating frames...")
        if type(base_embedding) == Image.Image:
            model = config.img2img()
        elif type(base_embedding) == torch.Tensor:
            model = config.model

        frames = generate_frames_from_embedding(
            model,
            base_embedding,
            prompt,
            modifications.split(","),
            style_image if style_image_path else None,
            num_steps=config.inference_steps,
        )
        # frames = interpolate_frames(frames[0], frames[1], steps=5) + frames[1:]
        logging.info(f"Frames generated: {len(frames)}")
        logging.info("Creating sprite sheet...")
        sprite_sheet = create_sprite_sheet(frames, 1, len(frames))
        logging.info("Sprite sheet created. Updating callback...")
        update_callback(sprite_sheet)
    return sprite_sheet


def async_generate_sprites(config, prompt, style_image_path, update_callback):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(
            generate_sprites, config, prompt, style_image_path, update_callback
        )
        return future


# Generate a base embedding from the initial prompt
def generate_base_embedding(pipeline, base_prompt, style_image=None):
    # Tokenize and encode the prompt
    # text_inputs = pipeline.tokenizer(
    #     base_prompt,
    #     return_tensors="pt",
    #     padding="max_length",
    #     max_length=pipeline.tokenizer.model_max_length,
    #     truncation=True,
    # )
    # text_embeddings = pipeline.text_encoder(text_inputs.input_ids.to(pipeline.device))[
    #     0
    # ]

    # # Set up latents
    # height = 512  # default height of the image
    # width = 512  # default width of the image
    # latents = torch.randn(
    #     (1, pipeline.unet.config.in_channels, height // 8, width // 8),
    #     generator=torch.manual_seed(0),
    # ).to(pipeline.device)
    # # Denoising loop (this is where you can extract latents)
    # pipeline.scheduler.set_timesteps(num_inference_steps=200)

    # for t in pipeline.scheduler.timesteps:
    #     # Predict the noise residual
    #     with torch.no_grad():
    #         noise_pred = pipeline.unet(
    #             latents, t, encoder_hidden_states=text_embeddings
    #         )["sample"]

    #     # Compute the previous noisy sample x_t -> x_t-1
    #     latents = pipeline.scheduler.step(noise_pred, t, latents)["prev_sample"]

    # # Latents now contains the final latent representation
    # final_latents = latents

    # logging.info(f"Base embedding generated. Latents: {final_latents.size()}")

    image = pipeline(prompt=base_prompt, num_inference_steps=200)[0][0]
    if style_image:
        image = transfer_style(image, style_image, load_vgg())

    # Briefly show the initial image
    image.show()
    return image


# Entry point
if __name__ == "__main__":
    config = Config()
    create_gui(config)
