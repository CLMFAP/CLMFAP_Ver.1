import torch
from transformers import AutoModel, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np

# Load the pre-trained smiles_transformer model and tokenizer
model = AutoModel.from_pretrained("ibm/smiles_transformer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("ibm/smiles_transformer-XL-both-10pct", trust_remote_code=True)

# Example SMILES input
smiles = "COc1ccccc1N1CCN(c2ncc3c(n2)C[C@@H](c2ccccc2Cl)CC3=O)CC1"  # Ethanol
inputs = tokenizer(smiles, return_tensors="pt")

# Forward pass to get the outputs with attention weights
model.eval()  # Set the model to evaluation mode
outputs = model(**inputs, output_attentions=True)
attentions = outputs.attentions  # List of attention matrices from each layer

# Calculate the average pooled attention matrix from the last layer
# Get the last layer's attentions (assuming it's the most relevant)
last_layer_attentions = attentions[7]  # shape: (batch_size, num_heads, seq_length, seq_length)

def grid_show(to_shows, cols):
    rows = (len(to_shows)-1) // cols + 1
    it = iter(to_shows)
    fig, axs = plt.subplots(rows, cols, figsize=(rows*8.5, cols*2))
    for i in range(rows):
        for j in range(cols):
            try:
                image, title = next(it)
            except StopIteration:
                image = np.zeros_like(to_shows[0][0])
                title = 'pad'
            axs[i, j].imshow(image, cmap='gray')
            axs[i, j].set_title(title)
            axs[i, j].set_yticks([])
            axs[i, j].set_xticks([])
    plt.show()

def visualize_heads(att_map, cols):
    to_shows = []
    att_map = att_map.squeeze()
    for i in range(att_map.shape[0]):
        to_shows.append((att_map[i], f'Head {i}'))
    average_att_map = att_map.mean(axis=0)
    to_shows.append((average_att_map, 'Head Average'))
    grid_show(to_shows, cols=cols)

last_layer_attentions = last_layer_attentions.detach().numpy()

print(last_layer_attentions.shape)

last_layer_attentions = last_layer_attentions[:,:,1:-1,1:-1]

# CC(C)C(C)(C)O
# print(last_layer_attentions.shape)
# rows_to_remove = [2,4,6,8,9,11]
# last_layer_attentions = np.delete(last_layer_attentions, rows_to_remove, axis=2)
# print(last_layer_attentions.shape)
# last_layer_attentions = np.delete(last_layer_attentions, rows_to_remove, axis=3)
# print(last_layer_attentions.shape)
# visualize_heads(last_layer_attentions, cols=4)


# To only show the average head attention:
# The different is that above visualize use numpy array, and below 
# average_attention = torch.mean(attentions, dim=1)  # Average over heads and select the first item in batch

average_attention = np.mean(last_layer_attentions, axis=1)
average_attention = np.squeeze(average_attention)
labels = ["C", "C", "C", "C", "C", "C", "O"]
labels = [i for i in "COc1ccccc1N1CCN(c2ncc3c(n2)C[C@@H](c2ccccc2Cl)CC3=O)CC1"]

# Visualize the average attention matrix
plt.figure(figsize=(10, 8))
# plt.imshow(average_attention, cmap='viridis')
plt.imshow(average_attention, cmap='gray')
plt.title(f"smiles_transformer Average Attention Map for SMILES: COc1ccccc1N1CCN(c2ncc3c(n2)C[C@@H](c2ccccc2Cl)CC3=O)CC1")
plt.xlabel("Token Positions")
plt.ylabel("Token Positions")
# plt.xticks(ticks=np.arange(len(labels)), labels=labels)
# plt.yticks(ticks=np.arange(len(labels)), labels=labels)
plt.colorbar()
plt.show()