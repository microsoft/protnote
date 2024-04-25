# Use a pipeline as a high-level helper
from transformers import pipeline
from fastchat import conversation

pipe = pipeline("text-generation", model="lmsys/vicuna-13b-v1.5-16k",device=0)

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-13b-v1.5-16k")
# model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-13b-v1.5-16k")


MAIN_INSTRUCTION = """
Put these three sentences together using the same language and style as original sentences. Avoid redundancy. If it's, better, separate it into multiple sentences.

#START EXAMPLES#
<Example 1>
Input:
Oxidoreductases
Acting on the CH-OH group of donors
With NAD(+) or NADP(+) as acceptor

Output:
Oxidoreductases acting on the CH-OH group of donors, with NAD(+) or NADP(+) as acceptor.

<Example 2>
Input:
Transferases
Transferring phosphorus-containing groups
Phosphotransferases with a nitrogenous group as acceptor

Output:
Transferases, transferring phosphorus-containing groups. Phosphotransferases with a nitrogenous group as acceptor.

<Example 3>
Input:
Hydrolases
Acting on peptide bonds (peptidases)
Omega peptidases

Output:
Hydrolases acting on peptide bonds (peptidases).  Omega peptidases.

<Example 4>
Input:
Transferases
Glycosyltransferases
Hexosyltransferases

<Example 5>
Input:
Lyases
Carbon-carbon lyases
Carboxy-lyases

Output:
Carbon-carbon lyases, carboxy-lyases

<Example 7>
Input:
Translocases
Catalysing the translocation of hydrons
Hydron translocation or charge separation linked to oxidoreductase reactions

Output:
Translocases catalysing the translocation of hydrons or charge separation linked to oxidoreductase reactions.

<Example 8>
Input:
Ligases
Forming carbon-nitrogen bonds
Acid--ammonia (or amine) ligases (amide synthases)

Output:
Ligases forming carbon-nitrogen bonds, specifically acid--ammonia (or amine) ligases (amide synthases)

<Example 9>
Input:
Isomerases
Racemases and epimerases
Acting on carbohydrates and derivatives

Output:
Isomerases, racemases and epimerases, acting on carbohydrates and derivatives

<Example 10>
Input:
Isomerases
Other isomerases
Other isomerases

Output:
Isomerases

#END EXAMPLES#

"""

queries = [
"""
Input:
Lyases
ATP-independent chelatases
Forming coordination complexes
""",
"""
Input:
Oxidoreductases
Acting on paired donors, with incorporation or reduction of molecular oxygen. The oxygen incorporated need not be derived from O2
With NADH or NADPH as one donor, and the other dehydrogenated
""",
"""
Hydrolases
Acting on ester bonds
Exoribonucleases producing 5'-phosphomonoesters
"""
]



full_queries = []
for i in queries:
    conv = conversation.get_conv_template("vicuna_v1.1")
    q = MAIN_INSTRUCTION + i 
    conv.append_message(conv.roles[0], q)
    conv.append_message(conv.roles[1], None)
    full_queries.append(conv.get_prompt())
print(full_queries)
print(pipe(full_queries))