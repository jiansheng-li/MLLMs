import sys
import argparse
import torch
import os
import json
from PIL import Image
import math
from tqdm import tqdm
import shortuuid
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_pretrained_emu_model():
    tokenizer = AutoTokenizer.from_pretrained('BAAI/Emu2-Chat')  # "BAAI/Emu2-Chat"

    model = AutoModelForCausalLM.from_pretrained(
        'BAAI/Emu2-Chat',
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True)


    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, context_len


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    tokenizer, model, context_len = load_pretrained_emu_model()
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        cur_prompt = qs
        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        inputs = model.build_input_ids(
            text=[cur_prompt],
            tokenizer=tokenizer,
            image=[image]
        )
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image=inputs["image"].to(torch.bfloat16),
                max_new_tokens=2048,
                length_penalty=-1)
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "metadata": {}}) + "\n")
        print({"question_id": idx,
               "prompt": cur_prompt,
               "text": outputs,
               "answer_id": ans_id,
               "metadata": {}})
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    args = parser.parse_args()

    eval_model(args)
