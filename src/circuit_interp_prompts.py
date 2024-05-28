attn_head_max_act_prompt = """
# CONTEXT
You are a meticulous AI researcher conducting an important investigation into a certain attention head in a transformer-based
language model. 

Right now, you're attempting to understand the computation performed by head {{head}} in layer {{layer}} of the transformer.
(we'll call this {{head_label}} from here on out).  From a previous analysis, we know that this attention head causes a
particular neuron to fire strongly, and we know what this neuron represents. Here's that description:

[HEAD_OUTPUT_NEURON_DESCRIPTION]
{{ head_output_description }}
[//HEAD_OUTPUT_NEURON_DESCRIPTION]

We're specifically interested in understanding *how* {{head_label}} causes this neuron to fire.  

In order to figure this out, you will:
1. Analyze the inputs to {{head_label}}
2. Deduce how the attention head uses that information to produce the neuron corresponding to HEAD_OUTPUT_VECTOR_DESCRIPTION.

Now then, let's break down how you'll complete these steps.

## STEP ONE:

### INPUT_ANALYSIS:

For some background, the purpose of an attention head is to move information from a previous token to a later token within 
a transformer.  Here's how this works: given a pair of tokens, the attention head computes a "query vector" for the later token, 
and it computes a "key vector" and a "value vector" for the earlier token. If the query vector of the later token matches with
the key vector of the earlier token, the attention head moves the earlier token's value vector to the later token.

For the sequence of tokens that we're studying, {{head_label}} moves information from '{{source_token}}' to '{{dest_token}}'.
Specifically, here's the sequence we're studying. We've put the "source" token in delimiters {{sr}} and {{sl}}, and we've put 
the "destination" token in delimiters {{dr}} and {{dl}}:

{{source_dest_annotated_seq}}

So {{head_label}} computes a query vector using the residual stream of token '{{dest_token}}', and it computes a key vector
and value vector using the residual stream of token '{{source_token}}'.

From a previous analysis, we've been able to isolate the particular neurons that the transformer uses to compute these
query, key, and value vectors.  Your job in this step is to figure out what these neurons represent, so we can 
then figure out how {{head_label}} uses this information to produce the output neuron.

### INPUT_DESCRIPTION:

For each of these neurons, you'll be given two types of inputs: 1) Max Activating Examples and 2) Top Promoted Tokens. Here's
a description of each input:

- MAX_ACTIVATING_EXAMPLES_DESCRIPTION
You will be given several examples of text that activate the neuron.  First we'll provide the example text without any 
annotations, and then we'll provide the same text with annotations that show the specific tokens that caused the neuron to 
activate and context about why the neuron fired.

The specific token that the neuron activates on will be the last token in the sequence, and will appear between {{tl}} 
and {{tr}} (like {{tl}}this{{tr}}).  

Additionally, each sequence will have tokens enclosed between {{cl}} and {{cr}} (like {{cl}}this{{cr}}).  From previous 
analysis, we know that the these tokens form the context for why our neuron fires on the token enclosed in {{tl}} and {{tr}} 
(in addition to the value of actual token itself).  Note that we treat the group of tokens enclosed between {{cl}} and {{cr}} 
as the "context" for why the neuron fired. (If no context tokens are provided, then the only relevant context is the token 
inside of {{tl}} and {{tr}}).

- TOP_PROMOTED_TOKENS_DESCRIPTION
You'll also be provided with the a list of the top tokens that this neuron promotes. Note that this list sometimes 
provides a very strong indication about what this neuron represents.  However, often this list isn't interpretable, and can 
be safely ignored.

## OUTPUT_DESCRIPTION:

Given the inputs provided, complete the following tasks for each neuron.

Step 1: Based on the MAX_ACTIVATING_EXAMPLES provided, write down observed patterns between the tokens that caused the 
neuron to activate (just the tokens enclosed in {{tl}} and {{tr}})

Step 2: Based on the MAX_ACTIVATING_EXAMPLES provided, write down patterns you see in the context for why the neuron 
fired. (Remember, the "context" for an example is the group of tokens enclosed in {{cl}} and {{cr}}).  Include any 
patterns in the relationships between different tokens in the context, and any patterns in the relationship between 
the context and the rest of the text.

Step 3: Write down any patterns you see in the TOP_PROMOTED_TOKENS.  If no patterns are apparent, just write "No 
apparent patterns -- Ignoring Top Promoted Tokens!" and ignore the top promoted tokens for the remainder of the analysis

Step 4: Write down several general shared features of the MAX_ACTIVATING_EXAMPLES, taking into account the 
TOP_PROMOTED_TOKENS if relevant.

Step 5: Based on the patterns you found in the MAX_ACTIVATING_EXAMPLES and the TOP_PROMOTED_TOKENS (if the 
TOP_PROMOTED_TOKENS are interpretable), write your best explanation for what this neuron represents. What kind of 
predictions does this neuron promote (either general or specific)?

Write your Step 5 explanation in the following form based on the neuron you're analyzing:

Use this form for the query neuron:
[QUERY_NEURON_EXPLANATION]
<your explanation>
[\\QUERY_NEURON_EXPLANATION]

Use this form for the key neuron:
[KEY_NEURON_EXPLANATION]
<your explanation>
[\\KEY_NEURON_EXPLANATION]

Use this form for the value neuron:
[VALUE_NEURON_EXPLANATION]
<your explanation>
[\\VALUE_NEURON_EXPLANATION]

Explanation Guidelines:
- Try to produce a final explanation that's both concise, and general to the examples provided
- Your explanation should be short: 1-2 sentences

## STEP 2:
Now that you've analyzed the query, key, and value that {{head_label}} uses, you should be able to figure out
how {{head_label}} computes the output neuron.  Perform the following tasks:

1. Start by rewriting HEAD_OUTPUT_NEURON_DESCRIPTION
2. Write an explanation that describes the reason why the query and key vectors for our source and destination 
tokens match, and explains what information {{head_label}} transfers from the source token to the output based 
on the value vector of the source token.

Provide your description in the following form:

[ATTENTION_HEAD_COMPUTATION_DESCRIPTION]
<your explanation>
[\\ATTENTION_HEAD_COMPUTATION_DESCRIPTION]

# TASK INPUT

[HEAD_OUTPUT_NEURON_DESCRIPTION]
{{ head_output_description }}
[//HEAD_OUTPUT_NEURON_DESCRIPTION]

## START_MAX_ACTIVATING_EXAMPLES_FOR_QUERY_NEURON:

{% for example in query_examples %}                         
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

## END_MAX_ACTIVATING_EXAMPLES_FOR_QUERY_NEURON:

## START_TOP_PROMOTED_TOKENS_FOR_QUERY_NEURON:

{% for token in top_query_tokens %}
{{loop.index}}. '{{ token }}'{% endfor %}

## END_TOP_PROMOTED_TOKENS_FOR_QUERY_NEURON:

## START_MAX_ACTIVATING_EXAMPLES_FOR_KEY_NEURON:

{% for example in key_examples %}                         
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

## END_MAX_ACTIVATING_EXAMPLES_FOR_KEY_NEURON:

## START_TOP_PROMOTED_TOKENS_FOR_KEY_NEURON:

{% for token in top_key_tokens %}
{{loop.index}}. '{{ token }}'{% endfor %}

## END_TOP_PROMOTED_TOKENS_FOR_KEY_NEURON:

## START_MAX_ACTIVATING_EXAMPLES_FOR_VALUE_NEURON:

{% for example in value_examples %}                         
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

## END_MAX_ACTIVATING_EXAMPLES_FOR_VALUE_NEURON:

## START_TOP_PROMOTED_TOKENS_FOR_VALUE_NEURON:

{% for token in top_value_tokens %}
{{loop.index}}. '{{ token }}'{% endfor %}

## END_TOP_PROMOTED_TOKENS_FOR_VALUE_NEURON:

# TASK OUTPUT
"""
