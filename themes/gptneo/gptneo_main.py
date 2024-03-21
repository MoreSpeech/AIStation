# -*- coding: utf-8 -*-
"""
@License :   Copyright (c) 2024 AITEK All rights reserved
@Author  :   人工智能技术派
@Time    :   2024/3/10 12:08:35
@Desc    :   展示gptneo运行过程，代码参考:https://github.com/feifeibear/LLMSpeculativeSampling
"""

import torch
from colorama import Fore, Style
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from loguru import logger

def sample(probs : torch.Tensor, num_samples: int = 1):
    idx_next = torch.multinomial(probs, num_samples=num_samples)
    if (idx_next.item() == 0):
        raise RuntimeError
    return idx_next

def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0):
  """

  Args:
      logits (torch.Tensorpe_): 2D tensor with shape (batch, vocab)
      top_k (int, optional): top_k. Defaults to 0.
      top_p (float, optional): top_p. Defaults to 0.0.

  Returns:
      torch.Tensor: a renormalized logits
  """
  if top_k > 0:
      filter = torch.topk(logits, min(top_k, logits.size(-1)))[0]
      logits[logits < filter[:, [-1]]] = float('-inf')
  if top_p > 0.0:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(
        F.softmax(sorted_logits, dim=-1), dim=-1)
    filter = cumulative_probs > top_p
    filter[..., 1:] = filter[..., :-1].clone()
    filter[..., 0] = 0
    indices_to_remove = filter.scatter(1, sorted_indices, filter)
    logits[indices_to_remove] = float('-inf')
  return logits

def norm_logits(logits : torch.Tensor, temperature : float, top_k : float, top_p : float) -> torch.Tensor:
  """

  Args:
      logits (torch.Tensor): shape (1, vocab)
      temperature (float): temperature
      top_k (float): top_k
      top_p (float): top_p

  Returns:
      torch.Tensor: next token with shape as (batch,  1)
  """
  assert logits.dim() == 2
  logits = logits / temperature
  logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
  probs = F.softmax(logits, dim=1)
  return probs

def generate(input_text, model_name):
  torch_device = 'cpu'
  # 步骤1: 对输入进行编码
  tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
  input_ids = tokenizer.encode(input_text, return_tensors='pt').to(torch_device)

  # 步骤2: 初始化语言模型

  ## 采样参数初始化
  N = 20
  top_k = 10
  top_p = 0.9
  temperature = 1.0


  ## 模型初始化
  demo_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                      torch_dtype=torch.float32,
                                                      device_map="cpu",
                                                      trust_remote_code=True)

  n = len(input_ids)
  T = len(input_ids) + N
  output = input_ids
  past_key_values = None  # 这里使用KVCache进行优化
  torch.manual_seed(123)  # 设置生成随机数的种子，保证实验结果是可福先到
  while n < T:
    # 步骤3: 语言模型推理
    if past_key_values:
        last_ids = output[:, -1]
        if last_ids.dim() == 1:
            last_ids = torch.unsqueeze(last_ids, 0)
        # torch.onnx.export(demo_model, (last_ids, past_key_values), "./model.onnx", input_names=["input", "cache"], output_names=["output"])
        outputs = demo_model(last_ids, past_key_values = past_key_values, use_cache = True)
    else:
        outputs = demo_model(output)
    # 步骤4: 推理结果解码
    last_p = norm_logits(outputs.logits[::, -1, :], temperature, top_k, top_p)
    past_key_values = outputs.past_key_values
    idx_next = sample(last_p)
    output = torch.cat((output, idx_next), dim=1)
    n += 1
  # 步骤5：推理结果解码
  generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
  logger.debug("%s %s %s" % (Fore.RED, generated_text, Style.RESET_ALL))

if __name__ == "__main__":
  input = "A red hat "
  model_name = "./models/TinyStories-1M"
  generate(input, model_name)
