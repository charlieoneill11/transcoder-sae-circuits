# Delimiters
left_delimiter = '<<'
right_delimiter = '>>'

SYSTEM_PROMPT = f"""
You are a meticulous AI researcher conducting an important investigation into a certain neuron in a language model. Your task is to analyze the neuron and provide an explanation that thoroughly encapsulates its behavior. Your task comes in two parts:

(Part 1) Tokens that the neuron activates highly on in text

You will be given a list of text examples on which the neuron activates. The specific tokens which cause the neuron to activate will appear with non-zero scores next to them. If a sequence of consecutive tokens all cause the neuron to activate, each token in the sequence will have a non-zero score.

Step 1: For each text example in turn, note which tokens (i.e., words, fragments of words, or symbols) caused the neuron to activate __highly__ (usually above 1.0 score). Then note which token came immediately before the activating token, for each activating token. 
Step 2: Look for patterns in the tokens you noted down in Step 1.
Step 3: Write down several general shared features of the text examples.

(Part 2) Tokens that the neuron boosts in the next token prediction

You will also be shown a list called Top_logits. The logits promoted by the neuron shed light on how the neuron's activation influences the model's predictions or outputs. Look at this list of Top_logits and refine your hypotheses from part 1. It is possible that this list is more informative than the examples from part 1.

Step 4: Pay close attention to the words in this list and write down what they have in common.
Step 5: Look at what they have in common, as well as patterns in the tokens you found in Part 1, to produce a single explanation for what features of text cause the neuron to activate. Propose your explanation in the following format:
[EXPLANATION]: <your explanation>

Guidelines:
- Try to produce a concise final description. Simply describe the text features that activate the neuron, and what its role might be based on the tokens it predicts.
- If either the text features or the predicted tokens are completely uninformative, you don't need to mention them.
- The last line of your response must be the formatted explanation.

Here are several example explanations."""

EXAMPLE_I = f"""
Example 1: | R (0.00) | alph (0.00) |  and (0.00) |  Kathy (0.00) |  rightly (2.02) |  state (0.00) | , (0.00) |  that (0.00) |  had (0.00) |
Example 2: | Much (0.00) |  love (0.00) |  from (0.00) |  the (0.00) |  Cher (0.00) | ubs (0.00) | ! (0.00) | \n (0.00) | \n (0.00) | - (0.00) | Jason (0.00) |  and (0.24) |  Jordan (0.00) | ! (1.97) | NEW (0.00) | UPDATE (0.00) |
Example 3: |  and (0.00) |  Peggy (0.00) |  hold (1.94) |  Dodd (0.00) |  hostage (0.24) |  in (0.00) |  a (0.00) |  hunting (0.00) |  cabin (0.00) |  for (0.00) |  two (0.00) |  days (0.00) | , (0.00) |  and (0.00) |  in (0.00) |  the (0.00) |  end (0.00) |

Top_logits: ["wcs", "cffffcc", " Schwar", "ce", "WAR", "osta", "lihood", "leans", "oren"]
"""

RESPONSE_I = f""" 
(Part 1)
Step 1.
ACTIVATING TOKENS: "rightly", "!", "hold", "hostage".
PREVIOUS TOKENS: "Kathy", "Cher", "Peggy", "Dodd".
NEXT TOKENS: "state", "NEW", "in"

Step 2.
The activating tokens don't appear to have a common pattern.
The previous tokens are all names.
The next tokens don't appear to have a common pattern.

Step 3.
- The activating tokens are mostly verbs.
- The previous tokens are all names.
- The text examples contain a mix of names and verbs.

(Part 2)
Step 4.
SIMILAR TOKENS: None.
The top logits list contains mostly unrelated words.

Step 5.
[EXPLANATION]: The neuron activates on tokens immediately following names.

"""

EXAMPLE_1 = f"""
Example 1: | and (0.00) | he (0.00) | was (0.00) | over (1.20) | the (1.00) | moon (1.50) | to (0.00) | find (0.00) |
Example 2: | we'll (0.00) | be (0.00) | laughing (0.00) | till (0.80) | the (0.60) | cows (1.00) | come (1.20) | home (1.50) | ! (0.00) | Pro (0.00) |
Example 3: | thought (0.00) | Scotland (0.00) | was (0.00) | boring (0.00) | , (0.00) | but (0.00) | really (0.00) | there's (0.00) | more (0.00) | than (0.70) | meets (1.00) | the (0.80) | eye (1.20) | ! (0.00) | I'd (0.00) |

Top_logits: ["elated", "joyful", "story", "thrilled", "spider"]
"""

RESPONSE_1 = f"""
(Part 1)
Step 1.
ACTIVATING TOKENS: "over the moon", "till the cows come home", "than meets the eye".
PREVIOUS TOKENS: "was", "laughing", "more".
NEXT TOKENS: "to find", "!", "!I'd".

Step 2.
The activating tokens are all parts of common idioms.
The previous tokens have nothing in common.
The next tokens are sometimes exclamation marks.

Step 3.
- The examples contain common idioms.
- In some examples, the activating tokens are followed by an exclamation mark.
- The text examples all convey positive sentiment.

(Part 2)
Step 4.
SIMILAR TOKENS: "elated", "joyful", "thrilled".
The top logits list contains words that are strongly associated with positive emotions.

Step 5.
[EXPLANATION]: Common idioms in text conveying positive sentiment.
"""

EXAMPLE_2 = f"""
Example 1: | a (0.00) | river (0.00) | is (0.00) | wide (0.00) | but (0.00) | the (0.00) | ocean (0.00) | is (0.00) | wider (1.00) | . (0.00) | The (0.00) | ocean (0.00) |
Example 2: | every (0.00) | year (0.00) | you (0.00) | get (0.00) | taller (1.50) | , (0.00) | " (0.00) | she (0.00) |
Example 3: | the (0.00) | hole (0.00) | was (0.00) | smaller (1.20) | but (0.00) | deeper (1.50) | than (0.00) | the (0.00) |

Top_logits: ["apple", "running", "book", "wider", "quickly"]
"""

RESPONSE_2 = f"""
(Part 1)
Step 1.
ACTIVATING TOKENS: "wider", "taller", "smaller", "deeper".
PREVIOUS TOKENS: "wider", "tall", "small", "deep".
NEXT TOKENS: "The", ",", " but", "than".

Step 2.
The activating tokens are mostly "er".
The previous tokens are mostly adjectives, or parts of adjectives, describing size.
The next tokens have nothing in common.
The neuron seems to activate on, or near, the token "er" in comparative adjectives describing size.

Step 3.
- In each example, the activating token was "er" appearing at the end of a comparative adjective.
- The comparative adjectives ("wider", "taller", "smaller", "deeper") all describe size.

(Part 2)
Step 4.
SIMILAR TOKENS: None.
The top logits list contains mostly unrelated nouns and adverbs.

Step 5.
[EXPLANATION]: The token "er" at the end of a comparative adjective describing size.
"""

EXAMPLE_3 = f"""
Example 1: | something (0.00) | happening (0.00) | inside (0.00) | my (0.00) | house (1.50) | , (0.00) | he (0.00) |
Example 2: | presumably (0.00) | was (0.00) | always (0.00) | contained (0.00) | in (0.00) | a (0.60) | box (1.50) | , (0.00) | " (0.00) | according (0.00) |
Example 3: | people (0.00) | were (0.00) | coming (0.00) | into (0.00) | the (0.00) | smoking (1.20) | area (1.30) | " (0.00) |
Example 4: | Patrick (0.00) | : (0.00) | " (0.00) | why (0.00) | are (0.00) | you (0.00) | getting (0.00) | in (0.00) | the (0.00) | way (1.50) | ? (1.00) | " (0.00) | Later (0.00) | ,

Top_logits: ["room", "end", "container", "space", "plane"]
"""

RESPONSE_3 = f"""
(Part 1)
Step 1.
ACTIVATING TOKENS: "house", "a box", "smoking area", "way?".
PREVIOUS TOKENS: "my", "in", "the", "the".
NEXT TOKENS: all quotation marks.

Step 2.
The activating tokens are all things that one can be in.
The previous tokens have nothing in common.
The next tokens are all quotation marks.

Step 3.
- The examples involve being inside something, sometimes figuratively.
- The activating token is a thing which something else is inside of.
- The activating token is followed by a quotation mark, suggesting it occurs within speech.

(Part 2)
Step 4.
SIMILAR TOKENS: "room", "container", "space".
The top logits list suggests a focus on nouns representing physical or metaphorical spaces.

Step 5.
[EXPLANATION]: Nouns preceding a quotation mark, representing a thing that contains something.
"""

AGENT_START = f"""
Now, it's your turn to propose an argument. Here is a list of text examples:

EXAMPLES:\n
{{examples}}\n\n

TOP LOGITS:\n
{{top_logits}}\n\n

Please finish with your explanation in this format.
[EXPLANATION]: <your explanation>
"""

def get_opening_prompt(examples, top_logits):
    examples_str = '\n'.join(examples)
    top_logits_str = ', '.join([f'"{logit}"' for logit in top_logits])
    opening_prompt = AGENT_START.format(examples=examples_str, top_logits=top_logits_str)
    return f"{SYSTEM_PROMPT}\n{EXAMPLE_I}\n{RESPONSE_I}\n{EXAMPLE_1}\n{RESPONSE_1}\n{EXAMPLE_2}\n{RESPONSE_2}\n{EXAMPLE_3}\n{RESPONSE_3}\n{opening_prompt}"