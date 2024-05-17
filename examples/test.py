
import os, sys
sys.path.append(os.getcwd())
# os.environ["TORCH_USE_CUDA_DSA"] = "1"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from utils import load, download_url, load_jsonl, enable_streaming_llm

import re
import time
import json
import argparse
from typing import Tuple, List

from tqdm import tqdm
# from modelscope import snapshot_download
from rouge import Rouge
import numpy as np

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaForCausalLM,
    TextStreamer,
)
from datasets import load_dataset, load_from_disk
from models.modify_llama import MyLlamaForCausalLM, hijack_llama
from models.tp import maybe_init_dist, apply_tp

# python -m debugpy --listen 56781 --wait-for-client examples/test.py

device = "cuda"

def load_model(model_id, sparsity_method) -> Tuple[LlamaForCausalLM, LlamaTokenizer]:
    # model: MyLlamaForCausalLM
    config = AutoConfig.from_pretrained(model_id)
    if sparsity_method == "sink":
        config.use_sink = True
    elif sparsity_method == "h2o":
        config.use_h2o = True
    elif sparsity_method == "snapkv":
        config.use_snapkv = True

    model = LlamaForCausalLM.from_pretrained(
        model_id,
        config=config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_id
    )
    return model, tokenizer


def longbench_one(task:str, model: LlamaForCausalLM, tokenizer: LlamaTokenizer, datapath:str):
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r", encoding="utf-8"))
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r", encoding="utf-8"))
    prompt_template = dataset2prompt[task]
    max_gen = dataset2maxlen[task]
    
    data = load_from_disk(f'{datapath}/{task}')

    context_lengths = []
    pbar = tqdm(data)
    torch.cuda.reset_peak_memory_stats()
    for json_obj in pbar:
        prompt = prompt_template.format(**json_obj)
        inputs = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = inputs.input_ids.shape[-1]
        context_lengths.append(context_length)
        use_cache = False
        pbar.set_description(f"len {context_length}")
        if context_length > 8000:
            continue
        output = model.generate(
            **inputs,
            max_new_tokens=max_gen,
            do_sample=True,
            temperature=1.0,
            use_cache=use_cache,
        )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        
        with open(f"results/h2o/{task}.jsonl", "a+", encoding="utf-8") as f:
            json.dump(
                {"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"], "token_len": context_length}, 
                f, ensure_ascii=False
            )
            f.write('\n')
    
    print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")


def data_range(task="qasper", datapath=""):
    from collections import Counter
    data = load_from_disk(f'{datapath}/{task}')
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r", encoding="utf-8"))
    prompt_template = dataset2prompt[task]
    cnt = Counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    for i, row in enumerate(data):
        prompt = prompt_template.format(**row)
        inputs = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        l = inputs.input_ids.shape[1]
        cnt[l // 1000 * 1000] += 1
        print(f"{i}: {l}")
    
    cnt = {k: v for k,v in sorted(list(cnt.items()))}
    print(cnt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id", type=str, default="lmsys/longchat-7b-v1.5-32k"
    )
    parser.add_argument("--datapath", type=str, default="~/LongBench")
    parser.add_argument("--sparsity_method", type=str, choices=["full", "sink", "h2o", "snapkv"])
    # parser.add_argument("--enable_streaming", action="store_true")
    # parser.add_argument("--start_size", type=int, default=4)
    # parser.add_argument("--recent_size", type=int, default=2000)
    args = parser.parse_args()
    model_id = args.model_id
    hijack_llama()
    model, tokenizer = load_model(model_id, args.sparsity_method)
    for task in ["qasper", "qmsum", "triviaqa", "passage_count", "lcc"]:
        with torch.no_grad():
            longbench_one(task, model, tokenizer, args.datapath)
    # data_range()
