# data conllection with other models including T5, mamba, 
# load the models and the different tokenizers 

import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer
import re

with open('./hidden_unit_activity/prompt_bank.json', 'r') as file:
    prompt_bank = json.load(file)


# Define the device
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

def load_llama3(model_id="meta-llama/Meta-Llama-3-8B", device=device): # 33 layers, 4096 embedding dimension
    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer


def load_T5(model_id="t5-small", device=device): # 7 encoder layer, 512 embedding dimension 
    # Load the model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
    return model, tokenizer


def load_mamba(model_id="state-spaces/mamba-130m-hf", device=device): # mamba has 25 layers and  768 embedding dimension
    # Load the model and tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = MambaForCausalLM.from_pretrained(model_id).to(device)
    return model, tokenizer

def load_RWKV(model_id= "RWKV/rwkv-4-169m-pile", device = device): # RWKV has 13 layers
    tokenizer = AutoTokenizer.from_pretrained(model_id)    
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    return model, tokenizer 



def collect_sentence_by_sentence_data(model, tokenizer, prompt_bank, device):
    for key in prompt_bank.keys():    
        paragraph = prompt_bank[key]
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', paragraph)
        if model_name == 'llama3': nlayer=33
        elif model_name == 'T5': nlayer=7
        elif model_name == 'mamba': nlayer = 25
        elif model_name == 'rwkv': nlayer = 13
        hidden_states_accumulated =[[] for _ in range(nlayer)] # This will be a list of lists, one list per layer
        for sentence in sentences:
            input_text = sentence
            inputs = tokenizer(input_text, return_tensors="pt").to(device)
        
            # Forward pass through the model and access hidden states
            if model_name == 'llama3':
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True, return_dict=True)
                hidden_states = outputs.hidden_states  # Tuple of hidden states at each layer

            elif model_name == 'T5':
                with torch.no_grad():
                    encoder_outputs = model.encoder(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        output_hidden_states=True,
                        return_dict=True)
                # Get hidden states from all encoder layers
                encoder_hidden_states = encoder_outputs.hidden_states  # Tuple of length num_layers + 1
                last_encoder_hidden = encoder_hidden_states[-1]  # shape: (batch_size, seq_len, hidden_dim)
                # Start token for T5 decoder is usually </s> (id=1)
                decoder_input_ids = torch.tensor([[tokenizer.pad_token_id]], device = device)
                with torch.no_grad():
                    outputs = model(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        decoder_input_ids=decoder_input_ids,
                        output_hidden_states=True,
                        return_dict=True
                    )
                
                hidden_states = outputs.encoder_hidden_states
                #decoder_hidden_states = outputs.decoder_hidden_states
                # print(len(encoder_hidden_states))# len(hidden_states) = 7 # n_layer 
                # print(encoder_hidden_states[0].shape) # (1, sequence_length, embedding_dim=512)
                
            elif model_name == 'mamba':# mamba has 25 layers and the embedding dimension of 768
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states  # Tuple of hidden states at each layer
                print(len(hidden_states))
            elif model_name == 'rwkv': 
                with torch.no_grad():
                   outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states  # Tuple of hidden states at each layer
                print(len(hidden_states))# 13 layers 
                
            print(hidden_states[0].shape) # len(hidden_states) = 33 # n_layer 
            # hidden_states[0].shape # (1, sequence_length, embedding_dim)
            
            for i, hidden_state in enumerate(hidden_states):
                # Detach, move to CPU, and append the hidden state
                hidden_state_np = hidden_state.detach().cpu().float().numpy()
                hidden_states_accumulated[i].append(hidden_state_np)

        # Concatenate hidden states across all sentences for each layer
        for i, layer_states in enumerate(hidden_states_accumulated):
            # Concatenate along the sequence length (axis 1)
            concatenated_hidden_state = np.concatenate(layer_states, axis=1)  # Shape: (batch_size, total_sequence_length, hidden_size)
            
            # Save the concatenated hidden state
            np.save(f'./hidden_unit_activity/{key}_concatenated_hidden_state_layer_{i}_model={model_name}.npy', concatenated_hidden_state)
            print(f"Layer {i} concatenated hidden state shape: {concatenated_hidden_state.shape}")
    return 



for model_name in ['rwkv','T5','mamba','llama3']:
    if model_name == 'llama3':
        model, tokenizer = load_llama3(model_id="meta-llama/Meta-Llama-3-8B", device=device)
    if model_name == 'T5':
        model, tokenizer = load_T5(model_id="t5-small", device=device)
    if model_name == 'mamba':
        model, tokenizer = load_mamba(model_id="state-spaces/mamba-130m-hf", device=device)
    if model_name == 'rwkv':
        model, tokenizer = load_RWKV(model_id= "RWKV/rwkv-4-169m-pile", device = device)
        
    collect_sentence_by_sentence_data(model, tokenizer, prompt_bank, device)
    print('Finished collecting neural data')
    
