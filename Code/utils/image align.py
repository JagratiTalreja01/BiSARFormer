from PIL import Image, ImageDraw, ImageFont

def align_and_concatenate(images, target_size=(256, 256), labels=None, output_path="aligned_output.png"):
    """
    Resizes all images to the same size and places them side by side with optional labels.
    
    Args:
        images (list[str]): List of file paths to the input images.
        target_size (tuple): Desired (width, height) for resizing each image.
        labels (list[str]): Labels to display under each image. Default is None.
        output_path (str): File path to save the output image.
    """
    resized_imgs = []
    for path in images:
        img = Image.open(path)
        resized_imgs.append(img.resize(target_size, Image.Resampling.LANCZOS))

    # Canvas size (width = n * target_width, height = target_height + label space if labels are given)
    n = len(resized_imgs)
    canvas_height = target_size[1] + (40 if labels else 0)
    canvas = Image.new("RGB", (n * target_size[0], canvas_height), "white")

    # Paste images side by side
    for i, img in enumerate(resized_imgs):
        canvas.paste(img, (i * target_size[0], 0))

    # Add labels if provided
    if labels:
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("arial.ttf", 20)  # Adjust font size
        except:
            font = ImageFont.load_default()
        for i, label in enumerate(labels):
            text_x = i * target_size[0] + target_size[0] // 2
            text_y = target_size[1] + 5
            draw.text((text_x, text_y), label, fill="black", font=font, anchor="mm")

    canvas.save(output_path)
    print(f"Saved aligned comparison to {output_path}")


# Example usage
images = [
    "SAR_VV.png", "SAR_VH.png", "Target_Optical.png",
    "MT_GAN.png", "BicycleGAN.png", "Pix2Pix.png",
    "CycleGAN.png", "BiSARFormerGAN.png"
]

labels = ["SAR_VV", "SAR_VH", "Target Optical", "MT_GAN",
          "BicycleGAN", "Pix2Pix", "CycleGAN", "BiSARFormer GAN (Ours)"]

align_and_concatenate(images, target_size=(256, 256), labels=labels, output_path="comparison.png")
