import numpy as np
import matplotlib.pyplot as plt
import json
import itertools
import transformers
import torch
import pandas as pd
import pickle
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re


def plot_alignment(deviations_template,deviations_control,word_id,temp_id):
    common_bins = np.histogram_bin_edges(deviations_template + deviations_control, bins=30)
    
    # Create a figure with two subplots (1 row, 2 columns)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot the histograms in the first subplot
    ax[0].hist(deviations_template, bins=common_bins, alpha=0.5, color='orange', label='Template', edgecolor='black')
    ax[0].hist(deviations_control, bins=common_bins, alpha=0.5, color='blue', label='Control', edgecolor='black')
    ax[0].set_xlabel('Deviation ||X-T||')
    ax[0].set_ylabel('Frequency')
    ax[0].legend()
    
    # Plot the scatter and regression line in the second subplot
    ax[1].scatter(word_id, temp_id, color='skyblue', label='Data points')
    ax[1].plot(word_id, word_id, color='blue', linewidth=1, label='Alignment')
    ax[1].set_xlabel('Word Occurrence')
    ax[1].set_ylabel('Population Trajectory Occurrence')
    ax[1].legend()
    
    # Adjust layout for better spacing
    plt.tight_layout()
    
    # Show the combined plot
    plt.show()
    return 


def visualize_hidden_layer(hidden_state,layer = 1, n_neuron = 30):
    # Load the hidden state from the saved file
    hidden_state = np.load(f'hidden_state_layer_{layer}.npy')  # Shape: (batch_size, sequence_length, hidden_size)
    
    # Print the shape to confirm it has been loaded correctly
    print(f"Loaded layer {layer} hidden state shape: {hidden_state.shape}")
    # # Select a specific sequence from the batch (e.g., first batch element)
    # batch_idx = 0
    # sequence_idx = 0  # Select the first sequence (time step)
    # # Extract the hidden state for the selected batch and sequence
    # hidden_activations = hidden_state[batch_idx, sequence_idx, :]  # Shape: (hidden_size,)
    
    # Plot the activations as a heatmap
    plt.figure(figsize=(5, 10))
    plt.imshow(hidden_state[0,:,2*n_neuron:3*n_neuron], aspect='auto', cmap='viridis')
    plt.colorbar(label="Activation")
    plt.xlabel("Hidden Units")
    plt.ylabel("Sequence Token")
    plt.title(f"Layer {layer}")
    plt.yticks(ticks=range(len(tokens)), labels=tagged_tokens)
    plt.show()


def get_dev_from_mean_average_neuron(H):
    m = 1
    p = H.shape[-1]
    stacked_matrices = H[:,:]# Stack matrices along a new dimension to calculate mode across matrices
    mean_matrix = np.mean(stacked_matrices, axis=0)
    mean_matrix = np.squeeze(mean_matrix)  # Remove single-dimensional entries
    mean_matrix = mean_matrix.reshape([-1, p])
    constant_mask = np.ones((m, p), dtype=bool)
    dev = []
    for i in range(0, stacked_matrices.shape[0]): # iterate over tokens 
        matrix = stacked_matrices[i,:] 
        tempdev = np.sum((matrix - mean_matrix)**2)/mean_matrix.size # squared differnce sum divided by the size of the mean matrix
        dev.append(tempdev)
    return dev



# Function to clean up the tokens for plotting
def clean_token(token):
    token = token.replace('Ġ', ' ')  # Replace subword indicator
    if token == '<|begin_of_text|>':
        token = '[START]'
    return token






def detection(constant_positions, constant_values, dev_threshold):
    # detect positive when the deviation is below the maximal deviation threshold 
    template_hidden_state = hidden_state[constant_positions[:, 0],i,constant_positions[:, 1]]
    dev = np.sum((template_hidden_state - constant_values)**2)/template_hidden_state.size        
    if dev <=dev_threshold:return True
    else:return False


# Function to clean up the tokens for plotting
def clean_token(token):
    token = token.replace('Ġ', ' ')  # Replace subword indicator
    if token == '<|begin_of_text|>':
        token = '[START]'
    return token
    

def get_TP_FP(hidden_state,ST_indices,tokens, tolerance = 0.6,n_training = 4):
    # split indices into training and test set
    # parameters: tolerance; dev_threshold: 
    # deviations: how much the subpopulation activity deviates from the template to be classified as an article, dev can come from 
    
    n_training = n_training
    training_indices = np.random.choice(ST_indices,n_training, replace = False)
    
    ############# Training part: using training tagged token indices to identify invarying subpopulation
    # compute article mean population activity during training and then evaluate during test 
    m = 1
    p = hidden_state.shape[-1]
    
    stacked_matrices = hidden_state[0,training_indices,:] # Stack matrices along a new dimension
    n_seql = hidden_state.shape[1]

    mean_matrix = np.mean(stacked_matrices, axis=0) # Compute mean along the 0th axis 
    mean_matrix = np.squeeze(mean_matrix)  # Remove single-dimensional entries
    mean_matrix = mean_matrix.reshape([-1, p]) # Initialize a mask (True for positions where values are constant within tolerance, False otherwise)

    constant_mask = np.ones((m, p), dtype=bool)
    for i in range(0, stacked_matrices.shape[0]): # Compare each matrix with the learned reference (mode matrix)
        matrix = stacked_matrices[i,:]
        constant_mask &= (np.abs(matrix - mean_matrix) < tolerance)
    
    # The mask now contains True for positions with constant values within the tolerance, False otherwise.
    # Extract the constant values using the mask.
    # subpopulation specfied by their position and value 
    constant_positions = np.argwhere(constant_mask)
    constant_values = mean_matrix[constant_mask]
    
    ############ determine deviation threshold 
    max_dev = 0
    for i in range(0, stacked_matrices.shape[0]):
        template_hidden_state = stacked_matrices[i,constant_positions[:, 1]]# neural subpopulation
        #sometimes template_hidden_state.size will be 0
        dev = np.sum((template_hidden_state - constant_values)**2)/template_hidden_state.size
        if dev>max_dev:
            max_dev = dev 
    
    dev_threshold = max_dev
    
    ############ evaluation part, currently, evaluation is performed on the entire sequence
    deviations_control = [] 
    deviations_template = []
    word_count = 0
    template_matching_count = 0
    n_identification = 0
    correct = 0
    FP = 0
    word_id = []
    temp_id = []
    for i in range(0, len(tokens)):
        template_hidden_state = hidden_state[constant_positions[:, 0],i,constant_positions[:, 1]]
        dev = np.sum((template_hidden_state - constant_values)**2)/template_hidden_state.size
        if i in ST_indices:
            word_count = word_count + 1
            deviations_template.append(dev)
        else: 
            deviations_control.append(dev)
            
        if dev <=dev_threshold: # detect positive when the deviation is below the maximal deviation threshold 
            template_matching_count = template_matching_count + 1
            if i in ST_indices: # indice detected
                n_identification = n_identification + 1 
                correct = correct + 1
            if i not in ST_indices: FP = FP + 1

        if dev > dev_threshold:
            if i not in ST_indices: correct = correct + 1

                
        word_id.append(word_count)
        temp_id.append(template_matching_count)
    
    TP_rate = n_identification/len(ST_indices)
    FP_rate = FP/(n_seql - len(ST_indices))
    accuracy = correct / len(tokens)
    
    return constant_positions,constant_values, TP_rate, FP_rate, accuracy, dev_threshold 



# evaluate AUC on the test dataset, given the neural subspace captured by the model 
def eval_TP_FP(ST_indices, hidden_state, neural_chunk_dictionary, l):
    m = 1
    p = hidden_state.shape[-1]
    ## these are our detectors  
    constant_values = neural_chunk_dictionary['layer'][l]['constant_values']
    constant_positions = neural_chunk_dictionary['layer'][l]['constant_positions'] 
    tol = neural_chunk_dictionary['layer'][l]['tolerance'] 
    stacked_matrices = hidden_state[0,ST_indices,:] # Stack matrices along a new dimension
    n_seql = hidden_state.shape[1]
    ############ evaluation part, currently, evaluation is performed on the entire sequence
    word_count = template_matching_count = n_identification = correct = FP = 0
    word_id = temp_id = []
    for i in range(0, n_seql):
        template_hidden_state = hidden_state[constant_positions[:, 0],i,constant_positions[:, 1]]
        dev = np.sum((template_hidden_state - constant_values)**2)/template_hidden_state.size
        if i in ST_indices: 
            word_count = word_count + 1

        if dev <=tol: # detect positive when the deviation is below the maximal deviation threshold 
            template_matching_count = template_matching_count + 1
            if i in ST_indices: # indice detected
                n_identification = n_identification + 1 
                correct = correct + 1
            if i not in ST_indices: FP = FP + 1

        if dev > tol:
            if i not in ST_indices: correct = correct + 1

        word_id.append(word_count)
        temp_id.append(template_matching_count)
    
    TP_rate = n_identification/len(ST_indices)
    FP_rate = FP/(n_seql - len(ST_indices))
    accuracy = correct / n_seql
    return TP_rate, FP_rate, accuracy


def make_figure(fps_sorted, tps_sorted, exp_growth_list, tps, fps, n_neurons, accs, model_name=''):
    fig, ax = plt.subplots(1, 5, figsize=(12, 3))
    
    # Plot the histograms in the first subplot
    ax[0].plot(fps_sorted, tps_sorted, '-o')
    ax[0].set_ylim([0, 1])  # Replace ymin and ymax with desired limits for the y-axis
    ax[0].set_xlabel('False Positive')
    ax[0].set_ylabel('True Positive')
    ax[0].set_title('ROC Curve')
    
    # Plot the scatter and regression line in the second subplot
    ax[1].plot(exp_growth_list, tps, '-o')
    ax[1].set_xlabel('Tolerance Threshold')
    ax[1].set_xscale('log')  # Set x-axis to log scale
    ax[1].set_ylabel('True Positive Rate')
    ax[1].set_ylim([0, 1])  # Replace ymin and ymax with desired limits for the y-axis
    ax[1].set_title('TPR vs Tolerance')
    ax[1].legend()
    
    # Plot the scatter and regression line in the third subplot
    ax[2].plot(exp_growth_list, fps, '-o')
    ax[2].set_xlabel('Tolerance Threshold')
    ax[2].set_xscale('log')  # Set x-axis to log scale
    ax[2].set_ylabel('False Positive Rate')
    ax[2].set_ylim([0, 1])  # Replace ymin and ymax with desired limits for the y-axis
    ax[2].set_title('FPR vs Tolerance')
    ax[2].legend()
    
    # Plot the number of relevant neurons in the fourth subplot
    ax[3].plot(exp_growth_list, n_neurons, '-o')
    ax[3].set_xlabel('Tolerance Threshold')
    ax[3].set_xscale('log')  # Set x-axis to log scale
    ax[3].set_ylabel('Number of Relevant Neurons')
    ax[3].set_title('Relevant Neurons vs Tolerance')
    ax[3].legend()
    
    # Plot the accuracy in the fifth subplot
    ax[4].plot(exp_growth_list, accs, '-o')
    ax[4].set_ylabel('Accuracy')
    ax[4].set_xscale('log')  # Set x-axis to log scale
    ax[4].set_xlabel('Tolerance Threshold')
    ax[4].set_title('Accuracy vs Tolerance')
    ax[4].legend()

    
    # Adjust layout for better spacing
    plt.tight_layout()
    plt.title(f"layer{l}")
    plt.savefig(f"model={model_name}_layer{l}_plot.png", dpi=30, bbox_inches='tight')
    return




def plot_layer_statistics(layers, optimal_threshold,n_relevant_neurons,deviation_threshold, word,step,model_name=''):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3), sharex=True)
    
    # Bar plot for Optimal Threshold vs Layers
    ax[0].bar(layers, optimal_threshold)
    ax[0].set_yscale('log')  # Keep the y-axis in log scale if required
    ax[0].set_xlabel('Layers')
    ax[0].set_ylabel('Optimal Threshold')
    ax[0].set_title('Optimal Threshold Across Layers')
    ax[0].axhline(y=100, color='red', linestyle='--', linewidth=1.5, label='Max Threshold')
    
    # Bar plot for Number of Relevant Neurons vs Layers
    ax[1].bar(layers, n_relevant_neurons)
    ax[1].set_xlabel('Layers')
    ax[1].set_ylabel('Number of Encoding Neurons')
    ax[1].set_title('Number of Encoding Neurons Across Layers')
    ax[1].axhline(y=4096, color='red', linestyle='--', linewidth=1.5, label='')
    
    # Bar plot for Maximal Deviation vs Layers
    ax[2].bar(layers, deviation_threshold)
    ax[2].set_xlabel('Layers')
    ax[2].set_ylabel('Maximal Deviation')
    ax[2].set_title('Maximal Deviation Across Layers')
    
    # Adjust layout
    plt.tight_layout()
    plt.show()

    # Save the figure
    file_path = f"./plots/model={model_name}_word={word}_step={step}_layer_statistics_train.png"
    fig.savefig(file_path, bbox_inches='tight')
    
    return 



def plot_decoding_performance(layers, tps, fps, model_name='', word = 'N/A', step = 0, testdata = True): 
    # Create the figure and subplots
    fig, ax = plt.subplots(2, 1, figsize=(6, 2), sharex=True, gridspec_kw={'height_ratios': [1, 1]}, dpi = 150)
    
    # Upper bar plot (normal)
    ax[0].bar(layers, tps, color="skyblue", edgecolor="black")
    ax[0].set_ylabel((1-testdata)*'Max'+' TP ')
    ax[0].set_title('Decoding Performance Across Layers' + testdata*' (Test)' + (1-testdata)*' (Train with Optimal Tolerance Setting)')
    ax[0].set_ylim([0,1.1])
    ax[0].tick_params(axis='x')  # Rotate x-axis labels for better readability
    ax[0].axhline(y=1, color='brown', linestyle='--',)
    # Lower bar plot (reverse)
    ax[1].bar(layers, [-v for v in fps], color='lightcoral', edgecolor="black")  # Reverse the values (negative)
    ax[1].set_ylabel('-'+ (1-testdata)*'MinMax'+' FP')
    ax[1].set_ylim([-1.1,0])
    ax[1].tick_params(axis='x')  # Rotate x-axis labels for better readability
    ax[1].set_xlabel('Layer')
    
    plt.subplots_adjust(hspace=0)
    # minimal false positive given the maximal true positive 
    plt.show()
    # Save the figure
    if testdata:file_path = f"./plots/model={model_name}_word={word}_step={step}_test.png"
    else:file_path = f"./plots/model={model_name}_word={word}_step={step}_train.png"
    fig.savefig(file_path, bbox_inches='tight')

    return



def auc_across_layers(layers, aucs):
    plt.figure(figsize=(10, 3))  # Optional: set figure size for better visibility
    plt.bar(layers, aucs)
    plt.xlabel('Layer')
    plt.ylabel('AUC')
    plt.title('Signal Separability Across Layers (Training Data)')
    return 



def get_word_indices_in_sequence(tokens, word, step = 0):
    '''step:  the number of previous step (used to find components responsible for predicting an upcoming token)'''
    ######### find token occurrance ############
    variants = [word[0].lower(), word[0].upper(), word.upper(), word.lower()]
    spaces = ['',' ']
    combinations = [f"{space_before}{first_letter}{word[1:]}{space_after}" for first_letter, (space_before, space_after) in itertools.product(variants, itertools.product(spaces, repeat=2))]
    indices = [] # indices of identified token 
    for i in range(0, len(tokens)):
        # first letter lower, first letter upper, space in 
        if tokens[i] in combinations: 
            indices.append(i-step)
    return indices 



def get_word_indices_in_sequence_partword(tokens, word, step = 0):
    # step: int               step backward, population responsible for predicting the next token
    
    ######### find token occurrance ############
    limit = 5
    variants = [word[0].lower(), word[0].upper()]
    spaces = ['',' ']
    combinations = [f"{space_before}{first_letter}{word[1:]}{space_after}" for first_letter, (space_before, space_after) in itertools.product(variants, itertools.product(spaces, repeat=2))]
    print(combinations)
    indices = [] # indices of identified token 
    for i in range(0, len(tokens)):
        # first letter lower, first letter upper, space in 
        if any(comb in tokens[i] for comb in combinations):  # Substring matching
            indices.append(i-step)
            #print(tokens[i])
        else: # if the word is broken into parts, obtain the index of the token that marks the end of the word 
            check = False 
            if tokens[i] in word:
                k = 1
                tokenword = tokens[i]
                check = True 
            while check: 
                tokenword = tokens[i - k] + tokenword # check if tokenword belongs to any of the word variants 
                if tokenword in combinations: 
                    indices.append(i-step)# append the last index (the token finishing point) to the indice checker 
                k = k + 1
                if k>=limit:break
                # print(tokenword)
    print('=============')
    for i in indices:
        print(tokens[i])
    return indices 




def get_token_austin(tokenizer, input_text,device):
    '''needs special tokenization treatment, otherwise dimensionality does not match'''
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', input_text)
    all_tokens = []
    for sentence in sentences:
        input_text = sentence
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        tokens = [clean_token(token) for token in tokens]
        all_tokens = all_tokens + tokens
    return all_tokens


def decode_chunks(word, trainkey, testkey, prompt_bank,device, step = 0, n_training = 5, partword = False):     
    ######### find token occurrance ############
    model_id = "meta-llama/Meta-Llama-3-8B"
    input_text = prompt_bank[trainkey]
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if trainkey == 'austen-emma.txt':tokens = get_token_austin(tokenizer, input_text,device)
    else:tokens = obtain_sentence_wise_token(tokenizer, input_text,device)

    
    indices = get_word_indices_in_sequence(tokens, word, step = step) # n step predictive of the upcoming sequence
    if partword: indices = get_word_indices_in_sequence_partword(tokens, word, step = step) # n step predictive of the upcoming sequence

    
    print('length of indices = ', len(indices))
    if len(indices)>=10:
        plot = False
        nlayer = 33        
        neural_chunk_dictionary = {}
        try:
            with open("./neural_chunk_dictionary/neural_chunk_dictionary.pkl", "rb") as file:
                neural_chunk_dictionary = pickle.load(file)
        except FileNotFoundError:
            print("Dictionary file not found. Starting with an empty dictionary.")
        
        # Manually ensure intermediate levels exist
        if word not in neural_chunk_dictionary:neural_chunk_dictionary[word] = {}
        if step not in neural_chunk_dictionary[word]: neural_chunk_dictionary[word][step] = {}
        if 'layer' not in neural_chunk_dictionary[word][step]:neural_chunk_dictionary[word][step]['layer'] = {}
        print('finished loading files ')

        for l in range(0, nlayer): neural_chunk_dictionary[word][step]['layer'][l] = {}
        
        max_accs, max_tps, layers, optimal_threshold, deviation_threshold, n_relevant_neurons, minmax_FP, aucs = [],[],[],[],[],[],[],[]
        
        for l in range(0,nlayer)[:]: # 
            layer = l
            # print('layer = ', layer)
            hidden_state = np.load(f'./hidden_unit_activity/{trainkey}_concatenated_hidden_state_layer_{layer}.npy')
            exp_growth_list = [100, 30, 20, 10, 3] + [1 * 0.8**i for i in range(35)]# Create a list with exponential growth
        
            tps, fps, accs, n_neurons = [], [], [], []
            max_acc, max_tp, best_tol, best_threshold = (0,) * 4
        
            for tol in exp_growth_list:
                constant_positions,constant_values, TP_rate, FP_rate, accuracy,dev_threshold = get_TP_FP(hidden_state, indices, tokens, tolerance = tol,n_training = n_training)
                tps.append(TP_rate)
                fps.append(FP_rate)
                accs.append(accuracy)
                n_neurons.append(len(constant_positions))
        
                if accuracy>max_acc:
                    max_acc = accuracy
                    best_tol = tol
                    best_threshold = dev_threshold 
                    neural_chunk_dictionary[word][step]['layer'][l]['constant_values'] = constant_values
                    neural_chunk_dictionary[word][step]['layer'][l]['constant_positions'] = constant_positions
                    neural_chunk_dictionary[word][step]['layer'][l]['tolerance'] = dev_threshold
                if TP_rate > max_tp:max_tp = TP_rate
        
            
            temp = []
            for i in range(0, len(tps)):
                if tps[i]>=max_tp:temp.append(fps[i])
            
            fps_sorted, tps_sorted = zip(*sorted(zip(fps, tps)))
            fps_sorted = fps_sorted + (1,)
            tps_sorted = tps_sorted + (1,)
            auc = np.trapezoid(tps_sorted, fps_sorted)
        
            minmax_FP.append(min(temp))
            layers.append(layer)
            max_accs.append(max_acc)
            max_tps.append(max_tp)
            optimal_threshold.append(best_tol)
            n_relevant_neurons.append(n_neurons[np.argmax(accs)])
            deviation_threshold.append(best_threshold)
            # experiment_data.append({'Word': word, 'TP': max_tp, 'FP': min(temp), 'Layer': layer, 'Training': True, 'Test': False})
            aucs.append(auc) 
            print(auc)
            
            if plot: make_figure(fps_sorted, tps_sorted, exp_growth_list, tps, fps, n_neurons, accs)
        
        
        with open("./neural_chunk_dictionary/neural_chunk_dictionary.pkl", "wb") as file:
            pickle.dump(neural_chunk_dictionary, file)
    
        plot_decoding_performance(layers, max_tps, minmax_FP, step = step, word = word, testdata = False)
        plot_layer_statistics(layers, optimal_threshold,n_relevant_neurons,deviation_threshold,word,step)

        print('======================================== evaluate on test data ==============================================')

        if testkey == 'austen-persuasion.txt':tokens = get_token_austin(tokenizer, input_text,device)
        else:tokens = obtain_sentence_wise_token(tokenizer, input_text,device)

        input_text = prompt_bank[testkey]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokens = obtain_sentence_wise_token(tokenizer, input_text,device)
        indices = get_word_indices_in_sequence(tokens, word, step = step)
        if partword: indices = get_word_indices_in_sequence_partword(tokens, word, step = step) # n step predictive of the upcoming sequence

        if word in neural_chunk_dictionary:
            layers, accs, tps, fps = [], [], [], []
            for layer in range(0,nlayer): # 
                # Load the hidden state from the saved file
                hidden_state = np.load(f'./hidden_unit_activity/{testkey}_concatenated_hidden_state_layer_{layer}.npy')  # Shape: (batch_size, sequence_length, hidden_size)    
                TP_rate, FP_rate, accuracy = eval_TP_FP(indices, hidden_state, neural_chunk_dictionary[word][step], layer)
                print(TP_rate, FP_rate, accuracy)
                # experiment_data.append({'Word': word , 'TP': TP_rate, 'FP': FP_rate, 'Layer': layer, 'Training': False, 'Test': True})
                for lst, value in zip([accs, tps, fps, layers], [accuracy, TP_rate, FP_rate, layer]):
                    lst.append(value)
        
        plot_decoding_performance(layers, tps, fps, word = word, step = step, testdata = True)

def obtain_sentence_wise_token(tokenizer, input_text, device): 
    paragraph = input_text
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', paragraph)
    
    tokens = []
    for s in sentences:
        input_text = s
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        temp = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        tokens = tokens + [clean_token(token) for token in temp]
    return tokens 



def decode_chunks_all_models(word, trainkey, testkey, prompt_bank,device, tokenizer, model_name='T5', step = 0, n_training = 5, partword = False):     
    experiment_data = [] # record experiment data 
    
    ######### decode chunks in other models  ############
    input_text = prompt_bank[trainkey]
    if trainkey == 'austen-emma.txt':tokens = get_token_austin(tokenizer, input_text,device)
    else:tokens = obtain_sentence_wise_token(tokenizer, input_text,device)
    if model_name == 'T5': tokens = [token.replace('▁', ' ') for token in tokens]
    indices = get_word_indices_in_sequence(tokens, word, step = step) # n step predictive of the upcoming sequence
    if partword: indices = get_word_indices_in_sequence_partword(tokens, word, step = step) # n step predictive of the upcoming sequence

    if model_name == 'llama3': nlayer = 33
    elif model_name == 'T5': nlayer = 7
    elif model_name == 'mamba': nlayer = 25
    elif model_name == 'rwkv': nlayer = 13

    print('length of indices = ', len(indices))
    if len(indices)>=10: # word becomes eligible for population averaging when the nunber of occurrance exceed 10 in the training data
        plot = False
        neural_chunk_dictionary = {}
        try:
            with open(f"./neural_chunk_dictionary/neural_chunk_dictionary_model={model_name}.pkl", "rb") as file:
                neural_chunk_dictionary = pickle.load(file)
        except FileNotFoundError:
            print("Dictionary file not found. Starting with an empty dictionary.")
        
        # Manually ensure intermediate levels exist
        if word not in neural_chunk_dictionary:neural_chunk_dictionary[word] = {}
        if step not in neural_chunk_dictionary[word]: neural_chunk_dictionary[word][step] = {}
        if 'layer' not in neural_chunk_dictionary[word][step]:neural_chunk_dictionary[word][step]['layer'] = {}
        print('finished loading files ')

        for l in range(0, nlayer): neural_chunk_dictionary[word][step]['layer'][l] = {}
        
        max_accs, max_tps, layers, optimal_threshold, deviation_threshold, n_relevant_neurons, minmax_FP, aucs = [],[],[],[],[],[],[],[]
        
        for l in range(0,nlayer)[:]: # 
            layer = l
            # print('layer = ', layer)
            hidden_state = np.load(f'./hidden_unit_activity/{trainkey}_concatenated_hidden_state_layer_{layer}_model={model_name}.npy')
            exp_growth_list = [100, 30, 20, 10, 3] + [1 * 0.8**i for i in range(35)]# Create a list with exponential growth
        
            tps, fps, accs, n_neurons = [], [], [], []
            max_acc, max_tp, best_tol, best_threshold = (0,) * 4
        
            for tol in exp_growth_list:
                constant_positions,constant_values, TP_rate, FP_rate, accuracy,dev_threshold = get_TP_FP(hidden_state, indices, tokens, tolerance = tol,n_training = n_training)
                tps.append(TP_rate)
                fps.append(FP_rate)
                accs.append(accuracy)
                n_neurons.append(len(constant_positions))
        
                if accuracy>max_acc:
                    max_acc = accuracy
                    best_tol = tol
                    best_threshold = dev_threshold 
                    neural_chunk_dictionary[word][step]['layer'][l]['constant_values'] = constant_values
                    neural_chunk_dictionary[word][step]['layer'][l]['constant_positions'] = constant_positions
                    neural_chunk_dictionary[word][step]['layer'][l]['tolerance'] = dev_threshold
                if TP_rate > max_tp:max_tp = TP_rate
        
            
            temp = []
            for i in range(0, len(tps)):
                if tps[i]>=max_tp:temp.append(fps[i])
            
            fps_sorted, tps_sorted = zip(*sorted(zip(fps, tps)))
            fps_sorted = fps_sorted + (1,)
            tps_sorted = tps_sorted + (1,)
            auc = np.trapezoid(tps_sorted, fps_sorted)
        
            minmax_FP.append(min(temp))
            layers.append(layer)
            max_accs.append(max_acc)
            max_tps.append(max_tp)
            optimal_threshold.append(best_tol)
            n_relevant_neurons.append(n_neurons[np.argmax(accs)])
            deviation_threshold.append(best_threshold)
            experiment_data.append({'Model': model_name, 'Word': word ,'step': step, 'TP': TP_rate, 'FP': FP_rate,'n_training': n_training, 'Layer': layer, 'Training': True, 'Test': False})

            aucs.append(auc) 
            
            if plot: make_figure(fps_sorted, tps_sorted, exp_growth_list, tps, fps, n_neurons, accs,model_name = model_name)
        
        
        with open(f"./neural_chunk_dictionary/neural_chunk_dictionary_model={model_name}.pkl", "wb") as file:
            pickle.dump(neural_chunk_dictionary, file)
    
        plot_decoding_performance(layers, max_tps, minmax_FP, step = step, word = word, testdata = False,model_name = model_name)
        plot_layer_statistics(layers, optimal_threshold,n_relevant_neurons,deviation_threshold,word,step,model_name = model_name)

        print('======================================== evaluate on test data ==============================================')

        if testkey == 'austen-persuasion.txt':tokens = get_token_austin(tokenizer, input_text,device)
        else:tokens = obtain_sentence_wise_token(tokenizer, input_text,device)

        input_text = prompt_bank[testkey]
        tokens = obtain_sentence_wise_token(tokenizer, input_text,device)

        if model_name == 'T5': tokens = [token.replace('▁', ' ') for token in tokens]

        indices = get_word_indices_in_sequence(tokens, word, step = step)
        if partword: indices = get_word_indices_in_sequence_partword(tokens, word, step = step) # n step predictive of the upcoming sequence

        if word in neural_chunk_dictionary:
            layers, accs, tps, fps = [], [], [], []
            for layer in range(0,nlayer): # 
                # Load the hidden state from the saved file
                hidden_state = np.load(f'./hidden_unit_activity/{testkey}_concatenated_hidden_state_layer_{layer}_model={model_name}.npy')  # Shape: (batch_size, sequence_length, hidden_size)    
                TP_rate, FP_rate, accuracy = eval_TP_FP(indices, hidden_state, neural_chunk_dictionary[word][step], layer)
                print(TP_rate, FP_rate, accuracy)
                experiment_data.append({'Model': model_name, 'Word': word ,'step': step, 'TP': TP_rate, 'FP': FP_rate,'n_training': n_training, 'Layer': layer, 'Training': False, 'Test': True})
                for lst, value in zip([accs, tps, fps, layers], [accuracy, TP_rate, FP_rate, layer]):
                    lst.append(value)
        
        plot_decoding_performance(layers, tps, fps, word = word, step = step, testdata = True, model_name = model_name)

        
        return experiment_data # returns a list of dictionary 



def decode_sae(word, trainkey, testkey, prompt_bank,device, tokenizer,  step = 0, n_training = 5, partword = False):     
    experiment_data = [] # record experiment data 
    
    ######### decode chunks in other models  ############
    input_text = prompt_bank[trainkey]
    if trainkey == 'austen-emma.txt':tokens = get_token_austin(tokenizer, input_text,device)
    else:tokens = obtain_sentence_wise_token(tokenizer, input_text,device)
    indices = get_word_indices_in_sequence(tokens, word, step = step) # n step predictive of the upcoming sequence
    if partword: indices = get_word_indices_in_sequence_partword(tokens, word, step = step) # n step predictive of the upcoming sequence

    # the indice at the end token as a part of the word, the maximially activating neuron when the indice is present 
    nlayer = 33 # we fix this for llama

    print('length of indices = ', len(indices))
    if len(indices)>=10: # word becomes eligible for population averaging when the nunber of occurrance exceed 10 in the training data
        plot = True

        # each layer has an neuron index       
        max_accs, max_tps, layers, optimal_threshold, deviation_threshold, n_relevant_neurons, minmax_FP, aucs = [],[],[],[],[],[],[],[]
        
        for l in range(0,nlayer)[:]: # 
            layer = l
            # print('layer = ', layer)
            hidden_state = np.load(f'./hidden_unit_activity/{trainkey}_concatenated_hidden_state_layer_{layer}_model={model_name}.npy')
        
            max_acc, max_tp, best_tol, best_threshold = (0,) * 4
        
            constant_positions,constant_values, TP_rate, FP_rate, accuracy,dev_threshold = get_TP_FP(hidden_state, indices, tokens, tolerance = tol,n_training = n_training)

            if accuracy>max_acc:
                max_acc = accuracy
                best_threshold = dev_threshold 
                neural_chunk_dictionary[word][step]['layer'][l]['constant_values'] = constant_values
                neural_chunk_dictionary[word][step]['layer'][l]['constant_positions'] = constant_positions
                neural_chunk_dictionary[word][step]['layer'][l]['tolerance'] = dev_threshold
            if TP_rate > max_tp:max_tp = TP_rate
        
            
            temp = []
            for i in range(0, len(tps)):
                if tps[i]>=max_tp:temp.append(fps[i])
            
            fps_sorted, tps_sorted = zip(*sorted(zip(fps, tps)))
            fps_sorted = fps_sorted + (1,)
            tps_sorted = tps_sorted + (1,)
            auc = np.trapezoid(tps_sorted, fps_sorted)
        
            minmax_FP.append(min(temp))
            layers.append(layer)
            max_accs.append(max_acc)
            max_tps.append(max_tp)
            optimal_threshold.append(best_tol)
            n_relevant_neurons.append(n_neurons[np.argmax(accs)])
            deviation_threshold.append(best_threshold)
            experiment_data.append({'Model': model_name, 'Word': word ,'step': step, 'TP': TP_rate, 'FP': FP_rate,'n_training': n_training, 'Layer': layer, 'Training': True, 'Test': False})

            aucs.append(auc) 
            
            if plot: make_figure(fps_sorted, tps_sorted, exp_growth_list, tps, fps, n_neurons, accs,model_name = model_name)
        
        
        with open(f"./neural_chunk_dictionary/neural_chunk_dictionary_model={model_name}.pkl", "wb") as file:
            pickle.dump(neural_chunk_dictionary, file)
    
        plot_decoding_performance(layers, max_tps, minmax_FP, step = step, word = word, testdata = False,model_name = model_name)
        plot_layer_statistics(layers, optimal_threshold,n_relevant_neurons,deviation_threshold,word,step,model_name = model_name)

        print('======================================== evaluate on test data ==============================================')

        if testkey == 'austen-persuasion.txt':tokens = get_token_austin(tokenizer, input_text,device)
        else:tokens = obtain_sentence_wise_token(tokenizer, input_text,device)

        input_text = prompt_bank[testkey]
        tokens = obtain_sentence_wise_token(tokenizer, input_text,device)

        indices = get_word_indices_in_sequence(tokens, word, step = step)
        if partword: indices = get_word_indices_in_sequence_partword(tokens, word, step = step) # n step predictive of the upcoming sequence

        if word in neural_chunk_dictionary:
            layers, accs, tps, fps = [], [], [], []
            for layer in range(0,nlayer): # 
                # Load the hidden state from the saved file
                TP_rate, FP_rate, accuracy = eval_TP_FP(indices, hidden_state, neural_chunk_dictionary[word][step], layer)
                print(TP_rate, FP_rate, accuracy)
                experiment_data.append({'Model': model_name, 'Word': word ,'step': step, 'TP': TP_rate, 'FP': FP_rate,'n_training': n_training, 'Layer': layer, 'Training': False, 'Test': True})
                for lst, value in zip([accs, tps, fps, layers], [accuracy, TP_rate, FP_rate, layer]):
                    lst.append(value)
        
        plot_decoding_performance(layers, tps, fps, word = word, step = step, testdata = True, model_name = model_name)

        
        return experiment_data # returns a list of dictionary 






        