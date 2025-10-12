# python code to automate chunk extraction 
# python learn_dictionary.py
import numpy as np
import json
import matplotlib.pyplot as plt
import itertools
import transformers
import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import pickle
import re
from util import *

login(token="YOURAPITOKENHERE") # Log in with API token
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model_id = "meta-llama/Meta-Llama-3-8B"

# load the dictionary of frequent words 
with open('./hidden_unit_activity/top_freq_word.json', 'r') as file: top_freq_word = json.load(file)

# load the prompt bank data 
with open('./hidden_unit_activity/prompt_bank.json', 'r') as file:
    prompt_bank = json.load(file)

trainkey = 'prompt_frequent_words_train'
testkey = 'prompt_frequent_words_test'

for word in top_freq_word: 
    for step in [-2,-1,0,1,2]:
        print('word = ', word, 'step = ', step)
        decode_chunks(word, trainkey, testkey, prompt_bank,device, n_training = 5, step = step)
