###########################################################
# Official Code to Discover Chunks in an Unsupervised Way #
###########################################################

import json
from util import *
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim


with open('./hidden_unit_activity/prompt_bank.json', 'r') as file:
    prompt_bank = json.load(file)

version = 'no_sparsity'

key = 'austen-emma.txt'
testkey = 'austen-persuasion.txt'
K = 2000  # Number of chunk embeddings
embedding_dim = 4096

input_text = prompt_bank[key]
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model_id = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
inputs = tokenizer(input_text, return_tensors="pt").to(device)
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
tokens = [clean_token(token) for token in tokens]
print('start program')

# Function to clean up the tokens for plotting
def clean_token(token):
    token = token.replace('Ġ', ' ')  # Replace subword indicator
    if token == '<|begin_of_text|>':
        token = '[START]'
    return token


# chunk_emb: K x 4096
# x = N x 4096
# sim = cossim(chunk_emb, x.T)/temp
# L = cross_entropy(sim, argmax(sim))
class ChunkEmbeddingModel(nn.Module):
    def __init__(self, K, embedding_dim=4096):
        super(ChunkEmbeddingModel, self).__init__()
        self.chunk_emb = nn.Parameter(torch.randn(K, embedding_dim))  # Learnable chunk embeddings
    def forward(self, x): # Normalize chunk embeddings and input features
        chunk_emb_norm = F.normalize(self.chunk_emb, p=2, dim=1) # [2000, 1]
        x_norm = F.normalize(x, p=2, dim=1)
        sim = torch.mm(chunk_emb_norm, x_norm.T) # Compute cosine similarity
        sim = sim 
        return sim # [2000, 215919]

# Define sparsity regularization
def sparsity_regularization(embeddings, alpha=1e-3):
    # L1 regularization for sparsity
    return alpha * torch.sum(torch.abs(embeddings))

# Loss function
def compute_loss(sim, chunk_emb, alpha=1e-3):
    # Cross-entropy loss
    target = torch.argmax(sim, dim=0)  # [215919] Argmax along the similarity axis 
    target_expanded = target.unsqueeze(0)  # Shape becomes [1, 215919]
    loss = -sim.gather(0, target_expanded).squeeze(0).mean()  # Shape becomes [215919]
    # sparsity_loss = sparsity_regularization(chunk_emb, alpha)
    print('similarity loss ', loss)
    # print('sparsity loss ', sparsity_loss)
    return loss #+ sparsity_loss

# Training setup
def train_model(model, x, epochs=100, lr=1e-3, alpha=1e-4):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        sim = model(x)
        loss = compute_loss(sim, model.chunk_emb, alpha)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

def train_model_batch_based(model, x, batch_size=32, epochs=100, lr=1e-2, alpha=1e-6):
    # Create DataLoader for batching
    dataset = TensorDataset(x)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch in dataloader:
            batch_x = batch[0]  # Since TensorDataset returns tuples

            # Forward pass
            sim = model(batch_x)
            loss = compute_loss(sim, model.chunk_emb, alpha)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss for the epoch
            epoch_loss += loss.item()

        # Print epoch statistics
        if (epoch + 1) % 100 == 0:
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

def test_model(model, x_test, batch_size=32):
    # Create DataLoader for testing
    dataset = TensorDataset(x_test)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            batch_x = batch[0]  # Since TensorDataset returns tuples

            # Forward pass
            sim = model(batch_x)
            loss = compute_loss(sim, model.chunk_emb, alpha=0)  # Alpha can be 0 for testing

            # Accumulate loss
            test_loss += loss.item()

    avg_loss = test_loss / len(dataloader)
    print(f"Test Loss: {avg_loss:.4f}")
    return avg_loss



#################################
# training and eval
#################################

for layer in range(0,33):
    hidden_state = np.load(f'./hidden_unit_activity/{key}_concatenated_hidden_state_layer_{layer}.npy')
    hidden_state = hidden_state[0,:,:]
    x = torch.tensor(hidden_state, dtype=torch.float32).to(device)
    model = ChunkEmbeddingModel(K, embedding_dim)
    model.to(device)
    train_model_batch_based(model, x, batch_size=64, epochs=100, lr=1e-2, alpha=1e-7)
    torch.save(model.state_dict(), f'./neural_chunk_dictionary/unsupervised_embedding_model{layer}_version_{version}.pth')
    hidden_state_test = np.load(f'./hidden_unit_activity/{testkey}_concatenated_hidden_state_layer_{layer}.npy')
    hidden_state_test = hidden_state_test[0,:,:]
    x_test = torch.tensor(hidden_state_test, dtype=torch.float32).to(device) #torch.randn(N, embedding_dim)
    test_model(model, x_test, batch_size=64)
    print('*************************************' + str(layer) + '*************************************')
    


