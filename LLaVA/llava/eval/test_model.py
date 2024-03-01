import argparse
import logging

import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math

disable_torch_init()
model_path = os.path.expanduser(f"/home/jianshengli/gpt-4v-distribution-shift/llava_model/llava-v1.5-13b")
model_name = get_model_name_from_path(model_path)
model_base = model_path
images=[]
conv = conv_templates['vicuna_v1'].copy()
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name)
qs = f"""Given the image,answer the following question using the specified format.
Question: What is in this image? Choice list:[car,person].
Please choose a choice from the list and respond with the following format:
---BEGIN FORMAT TEMPLATE---
Answer Choice: [Your Answer Choice Here]
---END FORMAT TEMPLATE---
Do not deviate from the above format. Repeat the format template for the answer."""
qs = qs.replace('Given the image', 'Given the following image')
qs = qs.replace('in this image', 'in the following image')
# print('qs', qs)
# ans = 'Answer choice: '
qs_query=qs.split('\n')
cur_qs=qs_query[0]+'\n'
cur_answer='\n'.join(i for i in qs_query[1:-1])

print('cur_qs',cur_qs)
print('cur_answer',cur_answer)
ans = '''Answer Choice: [Your Answer Choice Here]'''
ice_text = cur_qs+DEFAULT_IMAGE_TOKEN + '\n'+cur_answer.replace('[Your Answer Choice Here]', 'car')+'\n\n'+cur_qs+DEFAULT_IMAGE_TOKEN + '\n'+cur_answer.replace('[Your Answer Choice Here]', 'car')+'\n\n'+cur_qs+DEFAULT_IMAGE_TOKEN + '\n'+cur_answer.replace('[Your Answer Choice Here]', 'car')+'\n\n'+cur_qs+DEFAULT_IMAGE_TOKEN + '\n'+cur_answer.replace('[Your Answer Choice Here]', 'person')+'\n\n'+cur_qs+DEFAULT_IMAGE_TOKEN + '\n'+cur_answer
# ice_text = cur_qs+DEFAULT_IMAGE_TOKEN + '\n'+ans.replace('[Your Answer Choice Here]', 'car')+'\n'+cur_qs+DEFAULT_IMAGE_TOKEN + '\n'+ans.replace('[Your Answer Choice Here]', 'bird')+'\n'+cur_qs+DEFAULT_IMAGE_TOKEN + '\n'+cur_answer
print('ice_text',ice_text)
# qs=DEFAULT_IMAGE_TOKEN + '\n' + DEFAULT_IMAGE_TOKEN + '\n'+DEFAULT_IMAGE_TOKEN + '\n'+DEFAULT_IMAGE_TOKEN + '\n'+DEFAULT_IMAGE_TOKEN + '\n'+'what is in the first image?'+ 'what is in the second image?'+'what is in the third image?'+ 'what is in the fourth image?'+'what is in the fifth image?'

image = Image.open('/home/jianshengli/gpt-4v-distribution-shift/data/VLCS/LabelMe/car/62d97fc5fa1b47beb338ba368c1d29c7.jpg')
images.append(image)
image = Image.open('/home/jianshengli/gpt-4v-distribution-shift/data/VLCS/Caltech101/bird/096f53557fc048ad80ffe969840ae952.jpg')
images.append(image)
image = Image.open('/home/jianshengli/gpt-4v-distribution-shift/data/VLCS/LabelMe/car/0016e3211cee4a8bb8e660bf26737d97.jpg')
images.append(image)
image = Image.open('/home/jianshengli/gpt-4v-distribution-shift/data/VLCS/LabelMe/car/0172ce85efa049a09cdf142bc400d3d0.jpg')
images.append(image)

image = Image.open('/home/jianshengli/gpt-4v-distribution-shift/data/VLCS/LabelMe/person/003f7ffbd3104adc99b5771416abb7fb.jpg')
images.append(image)
# image = Image.open('/home/jianshengli/gpt-4v-distribution-shift/data/VLCS/LabelMe/chair/12a4d4d5bad04ba99fc58ddff5539840.jpg')
# images.append(image)
# image = Image.open('/home/jianshengli/gpt-4v-distribution-shift/data/VLCS/LabelMe/chair/12a4d4d5bad04ba99fc58ddff5539840.jpg')
# images.append(image)
# image = Image.open('/home/jianshengli/gpt-4v-distribution-shift/data/VLCS/LabelMe/chair/12a4d4d5bad04ba99fc58ddff5539840.jpg')
# images.append(image)
conv.append_message(conv.roles[0], ice_text)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()
image_tensor = image_processor.preprocess(images, return_tensors='pt')['pixel_values']
input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
keywords = [stop_str]
stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        images=image_tensor.half().cuda(),
        do_sample=False,
        temperature=0,
        top_p=0,
        num_beams=1,
        # no_repeat_ngram_size=3,
        max_new_tokens=2048,
        use_cache=True)

input_token_len = input_ids.shape[1]
n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
if n_diff_input_output > 0:
    print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
outputs = outputs.strip()
print('outputs',outputs)
if outputs.endswith(stop_str):
    outputs = outputs[:-len(stop_str)]
outputs = outputs.strip()
print('outputs',outputs)
