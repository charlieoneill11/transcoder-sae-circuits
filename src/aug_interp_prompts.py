from jinja2 import Template
from typing import List


def main_aug_interp_prompt(
    examples: List[str], token_lr=("<<", ">>"), context_lr=("[[", "]]")
):
    tl, tr = token_lr
    cl, cr = context_lr

    template = Template(
        """
{# You are a meticulous AI researcher conducting an important investigation into a certain neuron in a language model. Your task is to analyze the neuron and provide an explanation that thoroughly encapsulates its behavior.  Here's how you will complete this task: #}

You are a meticulous AI researcher conducting an important investigation into a certain neuron in a language model. This language model is trained to predict the text that will follow a given input. Your task is to figure out what sort of behavior this neuron is responsible for -- namely, when this neuron fires, what kind of predictions does this neuron promote? Here's how you'll complete the task:

INPUT_DESCRIPTION: 
You will be given several examples of text that activate the neuron.  First we'll provide the
example text without any annotations, and then we'll provide the same text with annotations that show the specific tokens that caused the neuron to activate and context about why the neuron fired.

The specific token that the neuron activates on will be the last token in the sequence, and will appear between {{tl}} and {{tr}} (like {{tl}}this{{tr}}).  

Additionally, each sequence will have tokens enclosed between {{cl}} and {{cr}} (like {{cl}}this{{cr}}).  From previous analysis, we know that the these tokens form the context for why our neuron fires on the token enclosed in {{tl}} and {{tr}} (in addition to the value of actual token itself).  Note that we treat the group of tokens enclosed between {{cl}} and {{cr}} as the "context" for why the neuron fired.

Given these examples, complete the following steps.

OUTPUT_DESCRIPTION:

Step 1: Based on the examples provided, write down observed patterns between the tokens that caused the neuron to activate (just the tokens enclosed in {{tl}} and {{tr}})
Step 2: Based on the examples provide write down patterns you see in the context for why the neuron fired. (Remember, the "context" for an example is the group of tokens enclosed in {{cl}} and {{cr}}).  Include any patterns in the relationships between different tokens in the context, and any patterns in the relationship between the context and the rest of the text.
Step 3: Write down several general shared features of the text examples
{# Step 4: Based on the patterns you found between the activating token and the relevant context, write down an explanation for what causes this neuron to activate. Propose your explanation in the following form: #}
Step 4: Based on the patterns you found between the activating token and the relevant context, write down your best explanation for what this neuron is responsible for.  Propose your explanation in the following form: 
[EXPLANATION]: <your explanation>

Guidelines:
- Try to produce a final explanation that's both concise, and general to the examples provided
- Your explanation should be short: 1-2 sentences

INPUT:

{% for example in examples %}                         
EXAMPLE {{loop.index + 1}}:
- Base Text -
=================================================
{{example[0]}}
=================================================

- Annotated Text -
=================================================
{{example[1]}}
=================================================

{% endfor %}

OUTPUT:
                         
Step 1:
"""
    )

    return template.render(
        {"tl": tl, "tr": tr, "cl": cl, "cr": cr, "examples": examples}
    )
