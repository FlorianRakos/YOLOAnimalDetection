import os
from PIL import Image
import PIL as pil
from PIL import ImageFont
from PIL import ImageDraw 
import glob
cwd = os.getcwd()

nas_path = os.path.join(cwd, 'nas/m')
v8_path = os.path.join(cwd, 'v8/m/predict')
truth_path = os.path.join(cwd, 'truth')
combined_path = os.path.join(cwd, 'combined')

v8_files = glob.glob(os.path.join(v8_path, '*.jpg'))
nas_files = glob.glob(os.path.join(nas_path, '*.jpg'))
truth_files = glob.glob(os.path.join(truth_path, '*.jpg'))

v8_files.sort()
nas_files.sort()
truth_files.sort()

total_width = 1920
third_width = 640
padding = 20
border = 5
print("Total width: ", total_width)
print("Saving to: ", combined_path)
counter = 0

for i, file in enumerate(truth_files):
    print("Stitching: ", file)
    truth_image = pil.Image.open(file)
    nas_image = pil.Image.open(nas_files[i])
    v8_image = pil.Image.open(v8_files[i])

    combined_image = pil.Image.new('RGB', (total_width + border*2, truth_image.size[1]))
    combined_image.paste(truth_image, (0, 0))
    combined_image.paste(nas_image, (third_width + border, 0))
    combined_image.paste(v8_image, (third_width*2 + border*2, 0))

    draw = ImageDraw.Draw(combined_image, "RGB")
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
    #font = ImageFont.load_default()

    draw.rectangle([padding-5, padding-5, 200, 50], fill=(0,0,0,100))
    draw.rectangle([third_width + padding - 5, padding-5, third_width + 200, 50], fill=(0,0,0,100))
    draw.rectangle([third_width*2 + border + padding-5, padding-5, third_width*2 + 200, 50], fill=(0,0,0,100))

    draw.text((0 + padding, 0 + padding), "Ground Truth", (255,255,255), font=font)
    draw.text((third_width + border + padding, 0 + padding), "NAS", (255,255,255), font=font)
    draw.text((third_width*2 + border*2 + padding, 0 + padding), "V8", (255,255,255), font=font)

    combined_image.save(os.path.join(combined_path, os.path.basename(file)))

    counter += 1


    
print("Stitched ", counter, " images.")