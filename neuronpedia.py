import html
from collections import defaultdict
from json.decoder import JSONDecodeError
import os
import requests
from collections import defaultdict
import torch as t
from accelerate import Accelerator

def deduplicate_sequences(
    contexts_and_acts: defaultdict[int, list[tuple[list[str], list[float]]]]
) -> defaultdict[int, list[tuple[list[str], list[float]]]]:
    """Deduplicate sequences when necessary."""

    deduplicated_contexts_and_acts = defaultdict(list)

    for dim_idx, contexts_acts in contexts_and_acts.items():
        for context, acts in contexts_acts:
            if (context, acts) not in deduplicated_contexts_and_acts[dim_idx]:
                deduplicated_contexts_and_acts[dim_idx].append((context, acts))

    return deduplicated_contexts_and_acts


def top_k_contexts(
    contexts_and_activations: defaultdict[
        int, list[tuple[list[str], list[float]]]
    ],
    view: int,
    top_k: int,
) -> defaultdict[int, list[tuple[list[str], list[float]]]]:
    """
    Select the top-k contexts for each feature.

    The contexts are sorted by their max activation values, and are trimmed to
    a specified distance around each top activating token. Then, we only keep
    the top-k of those trimmed contexts.
    """

    top_k_contexts_acts = defaultdict(list)
    top_k_views = defaultdict(list)

    for dim_idx, contexts_acts in contexts_and_activations.items():
        ordered_contexts_acts: list[tuple[list[str], list[float]]] = sorted(
            contexts_acts,
            key=lambda x: max(x[-1]),
            reverse=True,
        )
        top_k_contexts_acts[dim_idx] = ordered_contexts_acts[:top_k]

        for context, acts in top_k_contexts_acts[dim_idx]:
            # index() should always return a unique index. It will prioritize
            # the first, in case of collisions.
            max_position = acts.index(max(acts))
            # To complete the open end of the slice, we add 1 to that side.
            view_slice = slice(max_position - view, max_position + view + 1)
            # Fixes singleton unpadded _contexts_.
            if isinstance(context, int):
                context: list = [context]
            top_k_views[dim_idx].append(
                (context[view_slice], acts[view_slice])
            )

    return top_k_views


def unpad_activations(
    activations_block: t.Tensor, unpadded_prompts: list[list[int]]
) -> list[t.Tensor]:
    """
    Unpads activations to the lengths specified by the original prompts.

    Note that the activation block must come in with dimensions (batch x stream
    x embedding_dim), and the unpadded prompts as an array of lists of
    elements.
    """
    unpadded_activations: list = []

    for k, unpadded_prompt in enumerate(unpadded_prompts):
        try:
            # Fixes singleton unpadded _activations_.
            if isinstance(unpadded_prompt, int):
                unpadded_prompt: list = [unpadded_prompt]
            original_length: int = len(unpadded_prompt)
            # From here on out, activations are unpadded, and so must be
            # packaged as a _list of tensors_ instead of as just a tensor
            # block.
            unpadded_activations.append(
                activations_block[k, :original_length, :]
            )
        except IndexError:
            print(f"IndexError at {k}")
            # This should only occur when the data collection was interrupted.
            # In that case, we just break when the data runs short.
            break

    return unpadded_activations


def neuronpedia_api(
    layer_idx: int,
    dim_idx: int,
    neuronpedia_key: str,
    sublayer_type: str,
    top_k: int,
    view: int,
) -> str:
    """
    Pulls down Neuronpedia API annotations for given graph nodes.
    """

    url_prefix: str = "https://www.neuronpedia.org/api/feature/gpt2-small/"
    url_post_res: str = "-res-jb/"
    url_post_attn: str = "-att_128k-oai/"
    url_post_mlp: str = "-mlp_128k-oai/"

    # sublayer_type: str = "res" | "attn" | "mlp"
    if sublayer_type == "res":
        url_post: str = url_post_res
    elif sublayer_type == "attn":
        url_post: str = url_post_attn
    elif sublayer_type == "mlp":
        url_post: str = url_post_mlp
    elif "error" in sublayer_type:
        return ""
    else:
        raise ValueError("Sublayer type not recognized:", sublayer_type)

    url: str = url_prefix + str(layer_idx) + url_post + str(dim_idx)

    if not neuronpedia_key:
        neuronpedia_key = os.getenv("NEURONPEDIA_KEY")

    try:
        response = requests.get(
            url,
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip, deflate, br, zstd",
                "Accept-Language": "en-US,en;q=0.9",
                "Cache-Control": "max-age=0",
                "Upgrade-Insecure-Requests": "1",
                "User-Agent": "sparse_circuit_discovery",
                "X-Api-Key": neuronpedia_key,
            },
            timeout=300,
        )
    except requests.exceptions.RequestException as e:
        print(e)
        return ""

    # assert (
    #     response.status_code != 404
    # ), "Neuronpedia API connection failed: 404"

    try:
        neuronpedia_dict: dict = response.json()
    except JSONDecodeError as e:
        print(e)
        return ""

    if neuronpedia_dict is None:
        return ""

    data: list[dict] = neuronpedia_dict["activations"]

    label: str = ""

    # defaultdict[int, list[tuple[list[str], list[float]]]]
    contexts_and_activations = defaultdict(list)
    for seq_dict in data:
        tokens: list[str] = seq_dict["tokens"]
        values: list[float | int] = seq_dict["values"]

        contexts_and_activations[dim_idx].append((tokens, values))

    top_contexts = top_k_contexts(contexts_and_activations, view, top_k)
    top_contexts = deduplicate_sequences(top_contexts)

    for context, acts in top_contexts[dim_idx]:
        if not context:
            continue

        max_a: int | float = max(acts)
        label += "<tr>"
        # It is known that the context is not empty by here.
        for token, act in zip(context, acts):
            token = html.escape(token)
            token = token.encode("unicode_escape").decode("utf-8")

            if act <= 0.0:
                label += f'<td bgcolor="#ffffff">{token}</td>'
            else:
                blue_prop = act / max_a
                rg_prop = 1.0 - blue_prop

                rg_shade = f"{int(96 + (159*rg_prop)):02x}"
                b_shade = f"{255:02x}"
                shade = f"#{rg_shade}{rg_shade}{b_shade}"
                cell_tag = f'<td bgcolor="{shade}">'
                label += f"{cell_tag}{token}</td>"
        label += "</tr>"

    return label