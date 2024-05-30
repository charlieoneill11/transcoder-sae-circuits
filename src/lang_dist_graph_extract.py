import torch
import circuitsvis as cv

from open_web_text import open_web_text_tokens
from tqdm import trange, tqdm
from circuit_lens import get_model_encoders
from circuit_discovery import CircuitDiscovery
from IPython.display import display


class LanguageDistributionGraphExtract:
    seq_batch_size: int
    num_seqs: int

    def __init__(
        self,
        strategy,
        component_filter,
        filter_threshold=0.6,
        seq_batch_size=10,
        num_seqs=1000,
        head_layer_threshold=0,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.strategy = strategy
        self.component_filter = component_filter
        self.filter_threshold = filter_threshold
        self.num_seqs = num_seqs
        self.seq_batch_size = seq_batch_size
        self.head_layer_threshold = head_layer_threshold

        self.model = get_model_encoders(self.device)[0]

    def get_circuit_discovery_for_seq(
        self, seq_index, index, is_target_index=True, visualize=True
    ):
        toks = open_web_text_tokens[seq_index].to("cuda")
        logits = self.model(toks)
        prob = logits.softmax(dim=-1)

        values, indices = torch.topk(prob, k=1, dim=-1)
        values, indices = values.squeeze(), indices.squeeze()

        mask = torch.logical_and(
            toks[1:] == indices[:-1], values[:-1] > self.filter_threshold
        ).float()

        targets = mask.nonzero().squeeze()

        if is_target_index:
            target = targets[index] + 1
            target = target.item()
        else:
            target = index

        cd = CircuitDiscovery(
            prompt=self.model.tokenizer.decode(toks[1:target]),
            seq_index=-1,
            token=self.model.tokenizer.decode(toks[target]),
            allowed_components_filter=self.component_filter,
        )
        self.strategy(cd)

        if visualize:
            print(
                "Target: ",
                self.model.tokenizer.decode(toks[target]),
                "::",
                target,
            )
            print(
                "Context:",
                self.model.tokenizer.decode(toks[max(target - 3, 0) : target + 3]),
            )
            masked_vals = mask * values[:-1]

            display(
                cv.tokens.colored_tokens(
                    tokens=self.model.to_str_tokens(toks[1:]), values=masked_vals
                )
            )

        return cd

    def run(self, visualize=False):
        feature_vecs = []
        seq_pos = []

        for k, range_indices in tqdm(
            list(
                enumerate(
                    torch.split(torch.arange(0, self.num_seqs), self.seq_batch_size)
                )
            )
        ):
            toks = open_web_text_tokens[range_indices]  # .to("cuda")
            logits = self.model(toks)
            prob = logits.softmax(dim=-1)

            values, indices = torch.topk(prob, k=1, dim=-1)
            values, indices = values.squeeze(dim=-1), indices.squeeze(dim=-1)

            mask = torch.logical_and(
                toks[:, 1:] == indices[:, :-1].cpu(),
                values[:, :-1].cpu() > self.filter_threshold,
            ).float()

            for i, seq_i in enumerate(range_indices):
                if visualize:
                    print("Sequence: ", seq_i.item())

                targets = mask[i].nonzero().squeeze()

                total = 0

                for target in targets:
                    target += 1

                    if target < 2:
                        continue

                    cd = CircuitDiscovery(
                        prompt=self.model.tokenizer.decode(toks[i, 1:target]),
                        seq_index=-1,
                        token=self.model.tokenizer.decode(toks[i, target]),
                        allowed_components_filter=self.component_filter,
                    )
                    self.strategy(cd)

                    if not cd.are_heads_above_layer(self.head_layer_threshold):
                        continue

                    feature_vecs.append(cd.get_graph_edge_matrix())
                    seq_pos.append((seq_i.item(), target.item()))
                    total += 1

                if visualize:
                    print(f"Total added for Seq {seq_i}: {total}")

        return feature_vecs, seq_pos
