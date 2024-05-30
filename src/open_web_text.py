import numpy as np
import torch

from datasets import load_dataset
from autointerpretability import tokenize_and_concatenate
from transformers import AutoTokenizer

dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
dataset = dataset.shuffle(seed=42, buffer_size=10_000)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

tokenized_owt = tokenize_and_concatenate(
    dataset, tokenizer, max_length=128, streaming=True
)
tokenized_owt = tokenized_owt.shuffle(42)
tokenized_owt = tokenized_owt.take(12800 * 2)
owt_tokens = np.stack([x["tokens"] for x in tokenized_owt])

open_web_text_tokens = torch.tensor(owt_tokens)
