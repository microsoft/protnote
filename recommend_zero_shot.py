# Use a pipeline as a high-level helper
from transformers import pipeline
from fastchat import conversation

pipe = pipeline("text-generation", model="lmsys/vicuna-13b-v1.5-16k")

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-13b-v1.5-16k")
model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-13b-v1.5-16k")


MAIN_INSTRUCTION = ""
conv = conversation.get_conv_template("vicuna_v1.1")
conv.append_message(conv.roles[0], "Hello!")
conv.append_message(conv.roles[1], "Hello dear!")
conv.append_message(conv.roles[0], "How are you?")
conv.append_message(conv.roles[1], None)
print(conv.get_prompt())