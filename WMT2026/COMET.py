from comet import download_model, load_from_checkpoint
from huggingface_hub import whoami
import json
import argparse
import inspect
from pathlib import Path
from dotenv import load_dotenv
import os


env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)


print("Token loaded:", os.environ.get("HF_TOKEN") is not None)
print(whoami())


model_path = download_model("Unbabel/XCOMET-XL")
model = load_from_checkpoint(model_path)
'''
data = [
    {
        "src": "Boris Johnson teeters on edge of favour with Tory MPs",
        "mt": "Boris Johnson ist bei Tory-Abgeordneten völlig in der Gunst",
        "ref": "Boris Johnsons Beliebtheit bei Tory-MPs steht auf der Kippe"
    }
]
model_output = model.predict(data, batch_size=8)
# Segment-level scores
print (model_output.scores)

# System-level score
print (model_output.system_score)

# Score explanation (error spans)
print (model_output.metadata.error_spans)
'''


'''
Test tokenizer
'''
token_ids = model.encoder.tokenizer.encode("I am me.")
print(token_ids)#[0, 87, 444, 163, 5, 2]
length = len(token_ids)
print(len(token_ids)) #6
sentence = model.encoder.tokenizer.decode(token_ids)
print(sentence) #<s> I am me.</s>

'''
tests slicing of subword_probs using true MT lengths
'''
