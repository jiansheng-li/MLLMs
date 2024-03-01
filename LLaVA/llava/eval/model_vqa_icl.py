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

XCOVFour_ice_data_paths = [
    '/home/jianshengli/data_icl/XCOVFour_ICE/normal/IM-0046-0001.jpeg',
    '/home/jianshengli/data_icl/XCOVFour_ICE/normal/IM-0016-0001.jpeg',
    '/home/jianshengli/data_icl/XCOVFour_ICE/normal/IM-0011-0001-0001.jpeg',
    '/home/jianshengli/data_icl/XCOVFour_ICE/normal/IM-0033-0001.jpeg',
    '/home/jianshengli/data_icl/XCOVFour_ICE/normal/IM-0033-0001-0001.jpeg',
    '/home/jianshengli/data_icl/XCOVFour_ICE/normal/IM-0013-0001.jpeg',
    '/home/jianshengli/data_icl/XCOVFour_ICE/normal/IM-0007-0001.jpeg',
    '/home/jianshengli/data_icl/XCOVFour_ICE/normal/IM-0011-0001-0002.jpeg',
    '/home/jianshengli/data_icl/XCOVFour_ICE/normal/IM-0039-0001.jpeg',
    '/home/jianshengli/data_icl/XCOVFour_ICE/normal/IM-0043-0001.jpeg',
    '/home/jianshengli/data_icl/XCOVFour_ICE/PNEUMONIA_COVID/10.jpeg',
    '/home/jianshengli/data_icl/XCOVFour_ICE/PNEUMONIA_COVID/8.jpg',
    '/home/jianshengli/data_icl/XCOVFour_ICE/PNEUMONIA_COVID/1.jpeg',
    '/home/jianshengli/data_icl/XCOVFour_ICE/PNEUMONIA_COVID/4.jpeg',
    '/home/jianshengli/data_icl/XCOVFour_ICE/PNEUMONIA_COVID/9.jpeg',
    '/home/jianshengli/data_icl/XCOVFour_ICE/PNEUMONIA_COVID/2.jpeg',
    '/home/jianshengli/data_icl/XCOVFour_ICE/PNEUMONIA_COVID/5.jpg',
    '/home/jianshengli/data_icl/XCOVFour_ICE/PNEUMONIA_COVID/7.jpg',
    '/home/jianshengli/data_icl/XCOVFour_ICE/PNEUMONIA_COVID/6.jpg',
    '/home/jianshengli/data_icl/XCOVFour_ICE/PNEUMONIA_COVID/3.jpeg', ]

iwild_ice_data_paths = [
    '/home/jianshengli/data_icl/iwildcam_v2.0_processed_ICE/location_23/aepyceros melampus/8a0ff010-21bc-11ea-a13a-137349068a90.jpg',
    '/home/jianshengli/data_icl/iwildcam_v2.0_processed_ICE/location_23/aepyceros melampus/8a5eb8a8-21bc-11ea-a13a-137349068a90.jpg',
    '/home/jianshengli/data_icl/iwildcam_v2.0_processed_ICE/location_23/aepyceros melampus/8a2b0eea-21bc-11ea-a13a-137349068a90.jpg',
    '/home/jianshengli/data_icl/iwildcam_v2.0_processed_ICE/location_23/aepyceros melampus/8a4e4ca2-21bc-11ea-a13a-137349068a90.jpg',
    '/home/jianshengli/data_icl/iwildcam_v2.0_processed_ICE/location_23/aepyceros melampus/8a74df16-21bc-11ea-a13a-137349068a90.jpg',
    '/home/jianshengli/data_icl/iwildcam_v2.0_processed_ICE/location_23/aepyceros melampus/8a3fc6b4-21bc-11ea-a13a-137349068a90.jpg',
    '/home/jianshengli/data_icl/iwildcam_v2.0_processed_ICE/location_23/aepyceros melampus/8a23ba96-21bc-11ea-a13a-137349068a90.jpg',
    '/home/jianshengli/data_icl/iwildcam_v2.0_processed_ICE/location_23/aepyceros melampus/8a5a80c6-21bc-11ea-a13a-137349068a90.jpg',
    '/home/jianshengli/data_icl/iwildcam_v2.0_processed_ICE/location_23/aepyceros melampus/8a4d4d52-21bc-11ea-a13a-137349068a90.jpg',
    '/home/jianshengli/data_icl/iwildcam_v2.0_processed_ICE/location_23/aepyceros melampus/8a4e1e76-21bc-11ea-a13a-137349068a90.jpg',
    '/home/jianshengli/data_icl/iwildcam_v2.0_processed_ICE/location_23/equus grevyi/8ad9df74-21bc-11ea-a13a-137349068a90.jpg',
    '/home/jianshengli/data_icl/iwildcam_v2.0_processed_ICE/location_23/equus grevyi/8a37520e-21bc-11ea-a13a-137349068a90.jpg',
    '/home/jianshengli/data_icl/iwildcam_v2.0_processed_ICE/location_23/equus grevyi/8a7c2cf8-21bc-11ea-a13a-137349068a90.jpg',
    '/home/jianshengli/data_icl/iwildcam_v2.0_processed_ICE/location_23/equus grevyi/8ae6072c-21bc-11ea-a13a-137349068a90.jpg',
    '/home/jianshengli/data_icl/iwildcam_v2.0_processed_ICE/location_23/equus grevyi/8a364a3a-21bc-11ea-a13a-137349068a90.jpg',
    '/home/jianshengli/data_icl/iwildcam_v2.0_processed_ICE/location_23/equus grevyi/8a6721c8-21bc-11ea-a13a-137349068a90.jpg',
    '/home/jianshengli/data_icl/iwildcam_v2.0_processed_ICE/location_23/equus grevyi/8a6581f6-21bc-11ea-a13a-137349068a90.jpg',
    '/home/jianshengli/data_icl/iwildcam_v2.0_processed_ICE/location_23/equus grevyi/8a0c54d2-21bc-11ea-a13a-137349068a90.jpg',
    '/home/jianshengli/data_icl/iwildcam_v2.0_processed_ICE/location_23/equus grevyi/8aa45dc2-21bc-11ea-a13a-137349068a90.jpg',
    '/home/jianshengli/data_icl/iwildcam_v2.0_processed_ICE/location_23/equus grevyi/8a709262-21bc-11ea-a13a-137349068a90.jpg', ]

nih_ice_data_paths = [
    '/home/jianshengli/data_icl/NIH-Chest_ICE/Chest/No Finding/00025024_000.png',
    '/home/jianshengli/data_icl/NIH-Chest_ICE/Chest/No Finding/00030246_001.png',
    '/home/jianshengli/data_icl/NIH-Chest_ICE/Chest/No Finding/00025109_001.png',
    '/home/jianshengli/data_icl/NIH-Chest_ICE/Chest/No Finding/00025124_002.png',
    '/home/jianshengli/data_icl/NIH-Chest_ICE/Chest/No Finding/00025097_002.png',
    '/home/jianshengli/data_icl/NIH-Chest_ICE/Chest/No Finding/00024792_000.png',
    '/home/jianshengli/data_icl/NIH-Chest_ICE/Chest/No Finding/00024986_000.png',
    '/home/jianshengli/data_icl/NIH-Chest_ICE/Chest/No Finding/00024868_000.png',
    '/home/jianshengli/data_icl/NIH-Chest_ICE/Chest/No Finding/00025006_000.png',
    '/home/jianshengli/data_icl/NIH-Chest_ICE/Chest/No Finding/00025081_016.png',
    '/home/jianshengli/data_icl/NIH-Chest_ICE/Chest/Effusion/00025252_003.png',
    '/home/jianshengli/data_icl/NIH-Chest_ICE/Chest/Effusion/00025489_000.png',
    '/home/jianshengli/data_icl/NIH-Chest_ICE/Chest/Effusion/00025082_004.png',
    '/home/jianshengli/data_icl/NIH-Chest_ICE/Chest/Effusion/00029821_014.png',
    '/home/jianshengli/data_icl/NIH-Chest_ICE/Chest/Effusion/00030137_004.png',
    '/home/jianshengli/data_icl/NIH-Chest_ICE/Chest/Effusion/00030389_013.png',
    '/home/jianshengli/data_icl/NIH-Chest_ICE/Chest/Effusion/00025228_008.png',
    '/home/jianshengli/data_icl/NIH-Chest_ICE/Chest/Effusion/00029795_004.png',
    '/home/jianshengli/data_icl/NIH-Chest_ICE/Chest/Effusion/00025227_006.png',
    '/home/jianshengli/data_icl/NIH-Chest_ICE/Chest/Effusion/00025262_003.png',
]

fmow_ice_data_path = [
    '/home/jianshengli/data_icl/fmow_v1.1_processed_ICE/region_1/park/rgb_img_169348.png',
    '/home/jianshengli/data_icl/fmow_v1.1_processed_ICE/region_1/park/rgb_img_169412.png',
    '/home/jianshengli/data_icl/fmow_v1.1_processed_ICE/region_1/park/rgb_img_169606.png',
    '/home/jianshengli/data_icl/fmow_v1.1_processed_ICE/region_1/park/rgb_img_170480.png',
    '/home/jianshengli/data_icl/fmow_v1.1_processed_ICE/region_1/park/rgb_img_169301.png',
    '/home/jianshengli/data_icl/fmow_v1.1_processed_ICE/region_1/park/rgb_img_169518.png',
    '/home/jianshengli/data_icl/fmow_v1.1_processed_ICE/region_1/park/rgb_img_169390.png',
    '/home/jianshengli/data_icl/fmow_v1.1_processed_ICE/region_1/park/rgb_img_169347.png',
    '/home/jianshengli/data_icl/fmow_v1.1_processed_ICE/region_1/park/rgb_img_169583.png',
    '/home/jianshengli/data_icl/fmow_v1.1_processed_ICE/region_1/park/rgb_img_169392.png',
    '/home/jianshengli/data_icl/fmow_v1.1_processed_ICE/region_1/airport/rgb_img_338580.png',
    '/home/jianshengli/data_icl/fmow_v1.1_processed_ICE/region_1/airport/rgb_img_338578.png',
    '/home/jianshengli/data_icl/fmow_v1.1_processed_ICE/region_1/airport/rgb_img_338579.png',
    '/home/jianshengli/data_icl/fmow_v1.1_processed_ICE/region_1/airport/rgb_img_338425.png',
    '/home/jianshengli/data_icl/fmow_v1.1_processed_ICE/region_1/airport/rgb_img_338426.png',
    '/home/jianshengli/data_icl/fmow_v1.1_processed_ICE/region_1/airport/rgb_img_338475.png',
    '/home/jianshengli/data_icl/fmow_v1.1_processed_ICE/region_1/airport/rgb_img_338577.png',
    '/home/jianshengli/data_icl/fmow_v1.1_processed_ICE/region_1/airport/rgb_img_338576.png',
    '/home/jianshengli/data_icl/fmow_v1.1_processed_ICE/region_1/airport/rgb_img_338574.png',
    '/home/jianshengli/data_icl/fmow_v1.1_processed_ICE/region_1/airport/rgb_img_338575.png',
]

ordinals = [
    "first", "second", "third", "fourth", "fifth",
    "sixth", "seventh", "eighth", "ninth", "tenth",
    "eleventh", "twelfth", "thirteenth", "fourteenth",
    "fifteenth", "sixteenth", "seventeenth", "eighteenth",
    "nineteenth", "twentieth", "twenty-first"]


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def icl_process(ice_num, qs, ice_data_path, conv, model, tokenizer, image_processor):
    qs = qs.replace('Given the image', 'Given the image above')
    qs = qs.replace('in this image', 'in the image above')
    # print('qs', qs)
    # ans = 'Answer choice: '
    ans = '''        ---BEGIN FORMAT TEMPLATE---
            Answer Choice: [Your Answer Choice Here]
            ---END FORMAT TEMPLATE---'''
    outputs = 0
    for i in range(ice_num):
        logging.info('begin icl')
        j = int(i / 2) if i % 2 == 0 else int(len(ice_data_path) - 1 - i // 2)
        # print('j', j)
        image_path = ice_data_path[j]
        label = image_path.split('/')[-2]
        print(image_path)
        # ice_text = ans.replace('[Your Answer Choice Here]', label) + '\n\n'
        ice_text = qs + ans.replace('[Your Answer Choice Here]', label) + '\n\n'
        conv.append_message(conv.roles[0], ice_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        image = Image.open(image_path)
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        print(image_tensor)
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        logging.info('icl finish')
    return outputs


def icl_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    model_base = model_path
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name)
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        images = []
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        qs = qs.replace('Given the image', 'Given the following image')
        qs = qs.replace('in this image', 'in the following image')
        # print('qs', qs)
        # ans = 'Answer choice: '
        qs_query = qs.split('\n')
        cur_qs = qs_query[0] + '\n'
        cur_answer = '\n'.join(i for i in qs_query[1:-1])
        if 'NIH' in args.dataset:
            ice_data_path = nih_ice_data_paths
        if 'fmow' in args.dataset:
            ice_data_path = fmow_ice_data_path
        if 'iwild' in args.dataset:
            ice_data_path = iwild_ice_data_paths
        if 'XCOV' in args.dataset:
            ice_data_path = XCOVFour_ice_data_paths
        ice_num = args.ice_num
        conv = conv_templates['vicuna_v1'].copy()
        ice_text = ''
        for i in range(ice_num):
            j = int(i / 2) if i % 2 == 0 else int(len(ice_data_path) - 1 - i // 2)
            # print('j', j)
            image_path = ice_data_path[j]
            label = image_path.split('/')[-2]
            # ice_text = ans.replace('[Your Answer Choice Here]', label) + '\n\n'
            ice_text = ice_text + cur_qs + DEFAULT_IMAGE_TOKEN + '\n' + cur_answer.replace('[Your Answer Choice Here]',
                                                                                    label) + '\n\n'
            image = Image.open(image_path)
            images.append(image)
        ice_text = ice_text + cur_qs + DEFAULT_IMAGE_TOKEN + '\n' + cur_answer
        image = Image.open(os.path.join(args.image_folder, image_file))
        images.append(image)
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
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        ans_id = shortuuid.uuid()
        if 'Answer' not in outputs:
            outputs = 'Answer choice:' + outputs
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": ice_text,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()


def icl_model2(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    model_base = model_path
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name)
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        images = []
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        qs = qs.replace('Given the image,', 'Given the image and the following examples,')
        tmp = qs.partition('\n')
        # print('tmp', tmp)
        ice_message = ''
        # print('qs', qs)
        # ans = 'Answer choice: '

        if 'NIH' in args.dataset:
            ice_data_path = nih_ice_data_paths
        if 'fmow' in args.dataset:
            ice_data_path = fmow_ice_data_path
        if 'iwild' in args.dataset:
            ice_data_path = iwild_ice_data_paths
        if 'XCOV' in args.dataset:
            ice_data_path = XCOVFour_ice_data_paths
        ice_num = args.ice_num
        conv = conv_templates[args.conv_mode].copy()
        image_qs = DEFAULT_IMAGE_TOKEN + '\n'
        for i in range(ice_num):
            j = int(i / 2) if i % 2 == 0 else int(len(ice_data_path) - 1 - i // 2)
            # print('j', j)
            image_path = ice_data_path[j]
            label = image_path.split('/')[-2]
            # ice_text = ans.replace('[Your Answer Choice Here]', label) + '\n\n'
            seq = ordinals[i]
            ice_message += 'The ' + seq + ' image is {}.\n'.format(label)
            image_qs += DEFAULT_IMAGE_TOKEN + '\n'
            image = Image.open(image_path)
            images.append(image)
        qs = '\n'.join([tmp[0], ice_message, tmp[-1]])
        qs = qs.replace('this image?', 'the {} image?'.format(ordinals[ice_num]))
        qs = image_qs + qs
        logging.info(qs)
        print(qs)
        image = Image.open(os.path.join(args.image_folder, image_file))
        images.append(image)
        conv.append_message(conv.roles[0], qs)
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
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        ans_id = shortuuid.uuid()
        if 'Answer' not in outputs:
            outputs = 'Answer choice:' + outputs
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": qs,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    model_base = model_path
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    print(answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]

        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        image = Image.open(os.path.join(args.image_folder, image_file))

        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": qs,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--ice_num", type=int, default=0)
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args()

    icl_model(args)
