
## 🐙 Requirements

1. Clone this repository and navigate to the project directory

```bash
git clone git@github.com:jiansheng-li/MLLMs.git
cd MLLMs
```

2. Create a Conda environment and activate it

```bash
conda create -n MLLMs python==3.10 -y
conda activate MLLMs
```

3. Navigate to the LLaVA directory and install its dependencies
   (if you want to evalution LLaVA)
```bash
cd LLaVA
pip install -e .
```

4. Return to the project root directory and install its dependencies

```bash
cd MLLMs
pip install -e .
```

#evaluation
1.zero-shot evaluation

model list:['LLaVA', 'Qwen', 'mPlug', 'intern', 'cogvlm', 'minigpt', 'LLaMA', 'blip2',
                                 'instructblip', 'emu']

All models are evaluated based on LLaVA format

To evaluate LLaMA, please download model first, and change your model path in module/LLaMA.py
```bash
python evaluation/eval_zeroshot.py --model_name model_to_choose
```

We recommend you to set num_sample to 500 which is the maximum number of all samples

To evaluate GPT-4
```bash
python evaluation/eval_gpt.py --model_name gpt --openai_api_key your openai key
```

To evaluate gemini
```bash
python evaluation/eval_ICL.py --model_name gemini --gemini_api_key your gemini key
```

2.CLIP evaluation

2.1 scaling law
You can evaluate different model of CLIP.

```bash
python evaluation/eval_CLIP --model_name available clip model
```

![Static Badge](https://img.shields.io/badge/Model-CLIP-green) 

The available clip model can refer to one provided by openai on https://github.com/openai/CLIP

2.2 linear_probe

check your 'train data path' and 'test data path' before your evaluation in evaluation/eval_linear_probe.py
and
```bash
python evaluation/eval_linear_probe.py
```

3.In-context-learing

You can set ice_num as 0,2,4,8

3.1 ICL of GPT-4
```bash
python evaluation/eval_gpt_ICL.py --model_name gpt --openai_api_key your openai key --ice_num 0
```


3.2 ICL of gemini
```bash
python evaluation/eval_gemini_ICL.py --model_name gemini --gemini_api_key your gemini key --ice_num 0
```
