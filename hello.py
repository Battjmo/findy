# import numpy as np

# msg = "Roll a dice"
# print(msg)

# print(np.random.randint(1,9))

import requests
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
from ultralytics import YOLO
from openai import OpenAI
import base64

# def encode_image(image_path):
#   with open(image_path, "rb") as image_file:
#     return base64.b64encode(image_file.read()).decode('utf-8')
  
# image_path = 'crops/person/im.jpg.jpg'
# image_base64 = encode_image(image_path)


# response = client.chat.completions.create(
#   model="gpt-4-vision-preview",
#   messages=[
#     {
#       "role": "user",
#       "content": [
#         {"type": "text", "text": "What’s in this image?"},
#         {
#           "type": "image_url",
#           "image_url": {
#             "url": "crops/person/im.jpg.jpg",
#           },
#         },
#       ],
#     }
#   ],
#   max_tokens=300,
# )

# print(response.choices[0])



# url = 'https://media.newyorker.com/cartoons/63dc6847be24a6a76d90eb99/master/w_1160,c_limit/230213_a26611_838.jpg'
# image = Image.open(requests.get(url, stream=True).raw).convert('RGB')  
# image.resize((596, 437))

# processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
# model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)

mode1 = YOLO('yolov8n.pt')
# generate boxy image
mode1.predict('IMG_2961.jpg', save=True, imgsz=640, conf=0.5)

# save result of parsing for manipulation
# result = mode1('IMG_2961.jpg')

# save out the crops as separate images
# result[0].save_crop('crops')
# result[0].save_txt('text_output')
# inputs = processor(image, return_tensors="pt").to(device, torch.float16)


# generated_ids = model.generate(**inputs, max_new_tokens=20)
# generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
# print(generated_text)
