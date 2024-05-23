import torch
import random

from transformer_lens import HookedTransformer
from data.eval_dataset import EvalItem
from typing import List


GT_GROUND_TRUTH_DATA = torch.load("data/gt_ground_truth.pt")

GT_GROUND_TRUTH_HEADS = torch.zeros(12, 12)

for layer, head in GT_GROUND_TRUTH_DATA:
    GT_GROUND_TRUTH_HEADS[layer, head] = 1


SINGLE_TOKEN_NOUNS = [
    "attack",
    "campaign",
    "case",
    "contact",
    "contract",
    "deal",
    "development",
    "effect",
    "employment",
    "existence",
    "fall",
    "growth",
    "impact",
    "marriage",
    "operation",
    "order",
    "plan",
    "practice",
    "process",
    "program",
    "progress",
    "project",
    "quest",
    "riot",
    "rise",
    "rule",
    "shift",
    "study",
    "test",
    "testing",
    "tests",
    "trial",
    "trip",
    "work",
]

model = HookedTransformer.from_pretrained("gpt2-small")


def generate_greater_than_sentence(noun: str, year: int):
    century = year // 100

    return f"The {noun} lasted from the year {year} to the year {century}"


def get_valid_years(
    tokenizer,
    start: int = 1000,
    end: int = 2150,
):
    """Get valid years (_abcd) between [start, end) that are tokenized into
    [_ab, cd] by the input tokenizer. Here _ denotes white space.
    """
    years = [" " + str(year) for year in range(start, end)]
    tokens = tokenizer(years)["input_ids"]
    detokenized = [tokenizer.convert_ids_to_tokens(year_toks) for year_toks in tokens]
    valid = torch.tensor(
        [(len(detok) == 2 and len(detok[1]) == 2) for detok in detokenized]
    )
    last_valid_index = None
    current_century = None
    for i, year in zip(range(len(valid)), range(start, end)):
        cent = year // 100
        if valid[i]:
            if current_century != cent:
                current_century = cent
                valid[i] = False
                if last_valid_index is not None:
                    valid[last_valid_index] = False
            last_valid_index = i
    if last_valid_index is not None:
        valid[last_valid_index] = False
    return torch.arange(start, end)[valid].tolist()


def generate_greater_than_dataset(N=10, counter_minus=10) -> List[EvalItem]:
    nouns = random.choices(SINGLE_TOKEN_NOUNS, k=N)
    years = random.choices(get_valid_years(model.tokenizer), k=N)

    sentences = [
        generate_greater_than_sentence(noun, year) for (noun, year) in zip(nouns, years)
    ]

    pre_toks = model.to_tokens(sentences, prepend_bos=True)

    all_next_toks = []

    for indices in torch.split(torch.arange(N), 100):
        toks = pre_toks[indices]

        logits = model(toks, return_type="logits")[:, -1, :]
        next_toks = logits.argmax(dim=-1)

        all_next_toks.append(next_toks)

    all_next_toks = torch.concat(all_next_toks, dim=0)

    print(pre_toks.shape, all_next_toks.shape)

    # final_toks = torch.concat([pre_toks, all_next_toks.unsqueeze(-1)], dim=-1)[:, 1:]

    str_prompts = []
    correct_toks = []

    for indices in torch.split(torch.arange(N), 100):
        str_prompts.extend(model.tokenizer.batch_decode(pre_toks[indices]))
        correct_toks.extend(model.tokenizer.batch_decode(all_next_toks[indices]))

    counter_toks = [str(max(0, (year % 100) - counter_minus)) for year in years]

    return [
        {"text": prompt, "correct": correct, "counter": counter}
        for prompt, correct, counter in zip(str_prompts, correct_toks, counter_toks)
    ]


# %%
model.to_single_token("h")
