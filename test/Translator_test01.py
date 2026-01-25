from transformers import AutoModelForCausalLM, AutoTokenizer
import utuil
from utuil import process_tencent_HY

model_name_or_path = "tencent/HY-MT1.5-7B"

text = '''
It’s on the house.
'''

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")  # You may want to use bfloat16 and/or move to GPU here
messages = [
    {"role": "user", "content": f"Translate the following segment into Chinese, without additional explanation.\n\n{text}"},
]
tokenized_chat = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,
    return_tensors="pt"
)

outputs = model.generate(tokenized_chat.to(model.device), max_new_tokens=2048)
output_text = tokenizer.decode(outputs[0])
print(output_text)
print('-----------------------------------------------------------------------')
print(process_tencent_HY(output_text))
