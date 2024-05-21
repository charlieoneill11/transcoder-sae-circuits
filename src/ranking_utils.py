import torch.nn.functional as F

from rich import print as rprint
from rich.table import Table


def get_token_ranking(token, logits):
    logits = logits.squeeze()

    softmax = F.softmax(logits, dim=-1)

    return (softmax > softmax[token]).sum().item()


def get_top_token_rankings(tokens, logits, seq_index, k=10):
    tokens = tokens.squeeze()
    logits = logits.squeeze()

    log_probs = F.log_softmax(logits, dim=-1)[seq_index]
    softmax = F.softmax(logits, dim=-1)[seq_index]

    indices = softmax.topk(k).indices.tolist()

    return list(zip(indices, softmax[indices].tolist(), (-log_probs[indices]).tolist()))


def visualize_top_tokens(model, tokens, logits, seq_index, token=None, k=5):
    tokens = tokens.squeeze()
    logits = logits.squeeze()

    if seq_index < 0:
        seq_index += tokens.size(0)

    if token is None:
        token_title = "Correct Token"

        token = tokens[seq_index + 1]
    else:
        token_title = "Selected Token"

    log_probs = F.log_softmax(logits, dim=-1)[seq_index]
    softmax = F.softmax(logits, dim=-1)[seq_index]

    indices = softmax.topk(k).indices.tolist()

    token_rank = (softmax > softmax[token]).sum().item()

    selected_token_table = Table(
        show_header=True, header_style="bold yellow", show_lines=True, title=token_title
    )
    selected_token_table.add_column("Rank")
    selected_token_table.add_column("Token")
    selected_token_table.add_column("Prob")
    selected_token_table.add_column("-Log Prob")
    selected_token_table.add_column("Token Index")

    selected_token_table.add_row(
        str(token_rank),
        f"'{model.tokenizer.decode(token)}'",
        f"{softmax[token].item() * 100:.3g}%",
        f"{-log_probs[token].item():.3g}",
        str(int(token)),
    )

    top_table = Table(
        show_header=True,
        header_style="bold yellow",
        show_lines=True,
        title="Top Tokens",
    )
    top_table.add_column("Rank")
    top_table.add_column("Token")
    top_table.add_column("Prob")
    top_table.add_column("-Log Prob")
    top_table.add_column("Token Index")

    for i, index in enumerate(indices):
        top_table.add_row(
            str(i),
            f"'{model.tokenizer.decode(index)}'",
            f"{softmax[index].item() * 100:.3g}%",
            f"{-log_probs[index].item():.3g}",
            str(index),
        )

    rprint(selected_token_table)
    rprint(top_table)
