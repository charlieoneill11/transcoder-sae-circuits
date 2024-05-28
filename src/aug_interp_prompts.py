from jinja2 import Template
from typing import List

explanation_delim = "EXPLANATION"


def main_aug_interp_prompt(
    examples: List[str],
    top_tokens: List[str],
    token_lr=("<<", ">>"),
    context_lr=("[[", "]]"),
):
    tl, tr = token_lr
    cl, cr = context_lr

    template = Template(
        """
{# You are a meticulous AI researcher conducting an important investigation into a certain neuron in a language model. Your task is to analyze the neuron and provide an explanation that thoroughly encapsulates its behavior.  Here's how you will complete this task: #}

You are a meticulous AI researcher conducting an important investigation into a certain neuron in a language model. This language model is trained to predict the text that will follow a given input. Your task is to figure out what sort of behavior this neuron is responsible for -- namely, when this neuron fires, what kind of predictions does this neuron promote? Here's how you'll complete the task:

INPUT_DESCRIPTION: 

You will be given two types of inputs: 1) Max Activating Examples and 2) Top Promoted Tokens.

- MAX_ACTIVATING_EXAMPLES_DESCRIPTION
You will be given several examples of text that activate the neuron.  First we'll provide the
example text without any annotations, and then we'll provide the same text with annotations that show the specific tokens that caused the neuron to activate and context about why the neuron fired.

The specific token that the neuron activates on will be the last token in the sequence, and will appear between {{tl}} and {{tr}} (like {{tl}}this{{tr}}).  

Additionally, each sequence will have tokens enclosed between {{cl}} and {{cr}} (like {{cl}}this{{cr}}).  From previous analysis, we know that the these tokens form the context for why our neuron fires on the token enclosed in {{tl}} and {{tr}} (in addition to the value of actual token itself).  Note that we treat the group of tokens enclosed between {{cl}} and {{cr}} as the "context" for why the neuron fired. (If no context tokens are provided, then the only relevant context is the token inside of {{tl}} and {{tr}}).

- TOP_PROMOTED_TOKENS_DESCRIPTION
Additionally, you'll be provided with the a list of the top tokens that this neuron promotes. Note that this list sometimes provides a very strong indication about what this neuron's role is.  However, often
this list isn't interpretable, and can be safely ignored.

OUTPUT_DESCRIPTION:
Given the inputs provided, complete the following tasks.

Step 1: Based on the MAX_ACTIVATING_EXAMPLES provided, write down observed patterns between the tokens that caused the neuron to activate (just the tokens enclosed in {{tl}} and {{tr}})
Step 2: Based on the MAX_ACTIVATING_EXAMPLES provided, write down patterns you see in the context for why the neuron fired. (Remember, the "context" for an example is the group of tokens enclosed in {{cl}} and {{cr}}).  Include any patterns in the relationships between different tokens in the context, and any patterns in the relationship between the context and the rest of the text.
Step 3: Write down any patterns you see in the TOP_PROMOTED_TOKENS.  If no patterns are apparent, just write "No apparent patterns -- Ignoring Top Promoted Tokens!" and ignore the top promoted tokens for the remainder of the analysis
Step 4: Write down several general shared features of the MAX_ACTIVATING_EXAMPLES, taking into account the TOP_PROMOTED_TOKENS if relevant.
{# Step 4: Based on the patterns you found between the activating token and the relevant context, write down an explanation for what causes this neuron to activate. Propose your explanation in the following form: #}
{# Step 4: Based on the patterns you found between the activating token and the relevant context, write down your explanation for what this neuron is responsible for.  #}
{#Step 5: Based on the patterns you found in the MAX_ACTIVATING_EXAMPLES and the TOP_PROMOTED_TOKENS (if the TOP_PROMOTED_TOKENS are interpretable), write down your explanation for what this neuron is responsible for.#}
Step 5: Based on the patterns you found in the MAX_ACTIVATING_EXAMPLES and the TOP_PROMOTED_TOKENS (if the TOP_PROMOTED_TOKENS are interpretable), write what the activation of this neuron causes the model to do. What kind of predictions does this neuron promote (either general or specific)?
{# Additionally, create an example that you think would cause this neuron to activate, and explain why you think this example would activate the neuron. #}
Propose your response in the following form: 
[{{explanation_delim}}]
<your explanation>
[\\{{explanation_delim}}]

Guidelines:
- Try to produce a final explanation that's both concise, and general to the examples provided
- Your explanation should be short: 1-2 sentences

INPUT:

- MAX_ACTIVATING_EXAMPLES:

{% for example in examples %}                         
EXAMPLE {{loop.index}}:
- Base Text -
=================================================
{{example[0]}}
=================================================

- Annotated Text -
=================================================
{{example[1]}}
=================================================

{% endfor %}

- TOP_PROMOTED_TOKENS:
{% for token in top_tokens %}
{{loop.index}}. '{{ token }}'{% endfor %}

OUTPUT:
                         
Step 1:
"""
    )

    return template.render(
        {
            "tl": tl,
            "tr": tr,
            "cl": cl,
            "cr": cr,
            "examples": examples,
            "top_tokens": top_tokens,
            "explanation_delim": explanation_delim,
        }
    )


def extract_explanation(
    res: str, replace_neuron_with_feature=True, delim=explanation_delim
):
    base = res.split(f"[{delim}]")[1].split(f"[\\{delim}]")[0]

    if replace_neuron_with_feature:
        base = base.replace("neuron", "feature")

    return base
