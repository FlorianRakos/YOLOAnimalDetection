import os
from PIL import Image
import PIL as pil
from PIL import ImageFont
from PIL import ImageDraw 

cwd = os.getcwd()

nas_path = os.path.join(cwd, 'nas')
v8_path = os.path.join(cwd, 'v8')
truth_path = os.path.join(cwd, 'truth')
combined_path = os.path.join(cwd, 'combined')

v8_files = os.listdir(v8_path)
nas_files = os.listdir(nas_path)
truth_files = os.listdir(truth_path)

total_width = pil.Image.open(os.path.join(truth_path, truth_files[0])).size[0] * 3
third_width = int(total_width/3.0)
padding = 20
border = 5
print(total_width)


for i, filename in enumerate(truth_files):
    truth_image = pil.Image.open(os.path.join(truth_path, filename))
    nas_image = pil.Image.open(os.path.join(nas_path, nas_files[i]))
    v8_image = pil.Image.open(os.path.join(v8_path, v8_files[i]))

    combined_image = pil.Image.new('RGB', (total_width + border*2, truth_image.size[1]))
    combined_image.paste(truth_image, (0, 0))
    combined_image.paste(nas_image, (third_width + border, 0))
    combined_image.paste(v8_image, (third_width*2 + border*2, 0))

    # label images
    draw = ImageDraw.Draw(combined_image)
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
    #font = ImageFont.load_default()
    
    draw.text((0 + padding, 0 + padding), "Ground Truth", (255,255,255), font=font)
    draw.text((third_width + border + padding, 0 + padding), "NAS", (255,255,255), font=font)
    draw.text((third_width*2 + border*2 + padding, 0 + padding), "V8", (255,255,255), font=font)

    combined_image.save(os.path.join(combined_path, filename))
    