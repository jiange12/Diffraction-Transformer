import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import glob
import pandas as pd
import pickle as pkl
from torch_geometric.data import Data
import math
from utils.data import create_pattern_data_sets, create_torch_data_dicts
import time
import torch.optim as optim
import random

class InputEmbeddings(nn.Module):
    def __init__(self, Is_dim, hklss_dim, fs_dim, frac_coords_dim, qs_dim, embed_dim):
        super(InputEmbeddings, self).__init__()
        self.Is_embed = nn.Linear(Is_dim, embed_dim)
        self.hklss_embed = nn.Linear(hklss_dim, embed_dim)
        self.fs_embed = nn.Linear(fs_dim, embed_dim)
        self.frac_coords_embed = nn.Linear(frac_coords_dim, embed_dim)
        self.qs_embed = nn.Linear(qs_dim, embed_dim)

    def forward(self, Is, hklss, fs, frac_coords, qs):
        Is_embedded = self.Is_embed(Is)
        hklss_embedded = self.hklss_embed(hklss)
        fs_embedded = self.fs_embed(fs)
        frac_coords_embedded = self.frac_coords_embed(frac_coords)
        qs_embedded = self.qs_embed(qs)
        
        return Is_embedded, hklss_embedded, fs_embedded, frac_coords_embedded, qs_embedded

class CombinedEmbeddings(nn.Module):
    def __init__(self, embed_dim):
        super(CombinedEmbeddings, self).__init__()
        self.embed_dim = embed_dim

    def forward(self, Is_embedded, hklss_embedded, fs_embedded, frac_coords_embedded, qs_embedded):
        combined = torch.cat([Is_embedded, hklss_embedded, fs_embedded, frac_coords_embedded, qs_embedded], dim=-1)
        return combined

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

def build_data(data):
    frac_coords = torch.from_numpy(data.frac_coords)
    disp_coords = frac_coords.reshape(-1, 1, 3)
    disp_tensor = disp_coords - disp_coords.transpose(0, 1)

    return Data(id=data.id,
                formula=data.formula,
                structure=data.structure,
                Is=torch.tensor(data.Is)[:5],
                hklss=[torch.stack([torch.tensor(hkl, dtype=torch.float64) for hkl in hkls]) for hkls in data.hklss][:5],
                fs=torch.from_numpy(data.fs),
                frac_coords=frac_coords,
                disp_tensor=disp_tensor,
                qs=torch.tensor(data.qs),
                target=None)  # No separate target in self-supervised learning

def create_torch_data_dicts(data_sets, model_folder, torch_data_file):
    torch_data_path = os.path.join(model_folder, torch_data_file)
    torch_data = dict()
    if len(glob.glob(torch_data_path)) == 0:
        for set_name, data_set in data_sets.items():
            N = len(data_set)
            set_data = dict()

            for i, (_, data) in enumerate(data_set.iterrows()):
                if len(data.frac_coords) == 1:
                    continue
                set_data[data.id] = build_data(data)
                print(f'num {i+1:4d}/{N}', end='\r')
            print('')

            torch_data[set_name] = set_data

        pkl.dump(torch_data, open(torch_data_path, 'wb'))
        return torch_data
    else:
        return pkl.load(open(torch_data_path, 'rb'))

class EmbeddingModel(nn.Module):
    def __init__(self, example_data, embed_dim):
        super(EmbeddingModel, self).__init__()
        
        # Determine the lengths based on the example data instance
        self.fs_length = len(example_data.fs)
        self.frac_coords_length = example_data.frac_coords.shape[1]
        self.disp_tensor_length = example_data.disp_tensor.shape[1] * example_data.disp_tensor.shape[2]

        self.Is_embed = nn.Linear(5, embed_dim)
        self.hklss_embed = nn.Linear(15, embed_dim)  # Adjust according to actual size needed
        self.fs_embed = nn.Linear(self.fs_length, embed_dim)
        self.frac_coords_embed = nn.Linear(self.frac_coords_length, embed_dim)
        self.disp_tensor_embed = nn.Linear(self.disp_tensor_length, embed_dim)
        self.qs_embed = nn.Linear(1, embed_dim)

    def forward(self, Is, hklss, fs, frac_coords, disp_tensor, qs):
        Is_embedded = self.Is_embed(Is)
        hklss_embedded = self.hklss_embed(hklss.view(-1, self.hklss_embed.in_features))  # Adjust reshape based on actual size
        fs_embedded = self.fs_embed(fs)
        frac_coords_embedded = self.frac_coords_embed(frac_coords)
        disp_tensor_embedded = self.disp_tensor_embed(disp_tensor.view(-1, self.disp_tensor_embed.in_features))  # Adjust reshape based on actual size
        qs_embedded = self.qs_embed(qs.view(-1, 1))

        return Is_embedded, hklss_embedded, fs_embedded, frac_coords_embedded, disp_tensor_embedded, qs_embedded

def create_embeddings(torch_data, embed_dim):
    # Extract a sample data instance to determine the lengths
    example_data = next(iter(torch_data['tr'].values()))
    
    model = EmbeddingModel(example_data, embed_dim)
    embeddings = {}

    for dataset in ['tr', 'va', 'te']:
        embeddings[dataset] = {}

        for key, data in torch_data[dataset].items():
            # Ensure hklss is reshaped properly
            hklss = torch.cat([torch.tensor(hkl, dtype=torch.float64).view(-1) for hkl in data.hklss])[:15] #sourceTensor.copy()
            
            Is_embedded, hklss_embedded, fs_embedded, frac_coords_embedded, disp_tensor_embedded, qs_embedded = model(
                data.Is[:5],  # Ensure proper slicing if needed
                hklss,
                data.fs,
                data.frac_coords,
                data.disp_tensor,
                data.qs
            )
            
            embeddings[dataset][key] = {
                'Is': Is_embedded,
                'hklss': hklss_embedded,
                'fs': fs_embedded,
                'frac_coords': frac_coords_embedded,
                'disp_tensor': disp_tensor_embedded,
                'qs': qs_embedded
            }

    return embeddings


# Usage
run_name = time.strftime('%y%m%d-%H%M%S', time.localtime())

structure_data_path = '../structure_data/structure_data_2_atoms.pkl'

data_folder = "C:/Users/ejian/Summer 2024 UROP/X-Ray diffraction prediction models/01_diff2struct_lstm_2atom_test (Earth's version)/data"
data_file = 'data.pkl'

model_folder = f"C:/Users/ejian/Summer 2024 UROP/X-Ray diffraction prediction models/01_diff2struct_lstm_2atom_test (Earth's version)/models/{run_name}/"
torch_data_file = "torch_data.pkl"#"C:/Users/ejian/Summer 2024 UROP/X-Ray diffraction prediction models/01_diff2struct_lstm_2atom_test (Earth's version)/models/240712-152237/torch_data.pkl"


try:
    os.mkdir(model_folder)
except FileExistsError:
    pass

tr_ratio = 0.8
va_ratio = 0.1
te_ratio = 0.1

ratios = [tr_ratio, va_ratio]

data_sets = create_pattern_data_sets(structure_data_path, data_folder, data_file, ratios)
torch_data = create_torch_data_dicts(data_sets, model_folder, torch_data_file)
print(torch_data)
embeddings = create_embeddings(torch_data, embed_dim=128)

# Accessing/Creating embeddings
train_embeddings = embeddings['tr']
val_embeddings = embeddings['va']
test_embeddings = embeddings['te']

tr_target_data = {}
val_target_data = {}

# Extracting frac_coords from the data
for id in train_embeddings:
    tr_target_data[id] = train_embeddings[id]['frac_coords']

for id in val_embeddings:
    val_target_data[id] = val_embeddings[id]['frac_coords']



# For a single tensor, you can concatenate them
# target_data_tensor = torch.cat(target_data, dim=0)


"""

# Assuming data_dict is a dictionary with your data objects
# Example structure of data_dict:
# data_dict = {
#     'data_1': DataObject_1,
#     'data_2': DataObject_2,
#     ...
# }

target_data = {key: value.frac_coords for key, value in data_dict.items()}

# If you have separate training and validation datasets, you can split the target data accordingly
# Example:
# tr_dict = {key: value for key, value in data_dict.items() if key in train_keys}
# va_dict = {key: value for key, value in data_dict.items() if key in val_keys}

tr_target_data = {key: value.frac_coords for key, value in tr_dict.items()}
va_target_data = {key: value.frac_coords for key, value in va_dict.items()}
"""

# print(target_data_tensor)

# print('train_embeddings:', train_embeddings)
# print('val_embeddings:', val_embeddings)
# print('test_embeddings', test_embeddings)

# Hyperparameters
batch_size = 16
block_size = 32
max_iters = 3000
eval_interval = 300
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 1
d = 128  # embedding dimension
n_head = 8  # number of heads
n_layer = 6  # number of transformer blocks

torch.manual_seed(1337)


class SelfAttention(nn.Module):
    def __init__(self, d, d_k):
        super(SelfAttention, self).__init__()
        self.Wk = nn.Linear(d, d_k, bias=False)
        self.Wq = nn.Linear(d, d_k, bias=False)
        self.Wv = nn.Linear(d, d_k, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        k = self.Wk(x)
        q = self.Wq(x)
        v = self.Wv(x)
        alpha = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        alpha = alpha.masked_fill(self.tril[:alpha.size(-2), :alpha.size(-1)] == 0, float('-inf'))
        alpha = F.softmax(alpha, dim=-1)
        out = alpha @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_k):
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList([SelfAttention(d, d_k) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

class FeedForward(nn.Module):
    def __init__(self, d):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.ReLU(),
            nn.Linear(4 * d, d),
        )

    def forward(self, x):
        return self.net(x)

# class TransformerBlock(nn.Module):
#     def __init__(self, d, n_head):
#         super(TransformerBlock, self).__init__()
#         d_k = d // n_head
#         self.attention = MultiHeadAttention(n_head, d_k)
#         self.feed_forward = FeedForward(d)
#         self.layer_norm1 = nn.LayerNorm(d)
#         self.layer_norm2 = nn.LayerNorm(d)

#     def forward(self, x):
#         x = x + self.attention(self.layer_norm1(x))
#         x = x + self.feed_forward(self.layer_norm2(x))
#         return x





    # Stuff from the LSTM model that we may need
    # from utils.plot import loss_train_plot, model_prediction

    # model.to(device)
    # checkpoint_generator = loglinspace(0.3, 5)
    # checkpoint = next(checkpoint_generator)
    # start_time = time.time()

    # record_lines = []

    # try:
    #     print('Use model.load_state_dict to load the existing model: ' + run_name + '.torch')
    #     results = torch.load(os.path.join(model_folder, run_name + '.torch'))
    #     history = results['history']
    #     s0 = history[-1]['step'] + 1
    #     model.load_state_dict(['state'])
    # except:
    #     print('There is no existing model')
    #     results = {}
    #     history = []
    #     s0 = 0

    # with open(os.path.join(model_folder, run_name + '.torch'), 'wb') as f:
    #     torch.save(results, f)

    # record_line = '%d\t%.20f\t%.20f'%(step,train_avg_loss,valid_avg_loss)
    # record_lines.append(record_line)




# # Stochastic Gradient Descent Training Loop
# def train_sgd():
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#     model.train()
    
#     for epoch in range(epochs): # Instead of doing a max, do until the improvement is minimal
#         for sample_id in train_embeddings.keys(): # Processed in the order of train_embeddings and not fully randomnized
#             data = train_embeddings[sample_id]
#             Is = data['Is'].to(device)
#             hklss = data['hklss'].to(device)
#             fs = data['fs'].to(device)
#             frac_coords = data['frac_coords'].to(device)
#             disp_tensor = data['disp_tensor'].to(device)
#             qs = data['qs'].to(device)
            
#             optimizer.zero_grad()
#             output = model(Is, hklss, fs, frac_coords, disp_tensor, qs)
            
#             # Assuming 'output' and 'target' need to be compatible in shape for loss calculation
#             target = target_data[sample_id].to(device)
#             loss = F.mse_loss(output, target)
#             loss.backward()
#             optimizer.step()
        
#         # Optional: Evaluate the model on the validation set after each epoch
#         val_loss = estimate_loss()
#         print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss}")

# def estimate_loss():
#     model.eval()
#     losses = []
#     with torch.no_grad():
#         for sample_id in val_embeddings.keys():
#             data = val_embeddings[sample_id]
#             Is = data['Is'].to(device)
#             hklss = data['hklss'].to(device)
#             fs = data['fs'].to(device)
#             frac_coords = data['frac_coords'].to(device)
#             disp_tensor = data['disp_tensor'].to(device)
#             qs = data['qs'].to(device)
            
#             output = model(Is, hklss, fs, frac_coords, disp_tensor, qs)
            
#             target = target_data[sample_id].to(device)
#             loss = F.mse_loss(output, target)
#             losses.append(loss.item())
#     model.train()
#     return sum(losses) / len(losses)

# train_sgd()
"""
# True SGD
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SubNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SubNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return torch.relu(self.fc(x))


class Transformer(nn.Module):
    def __init__(self, d, n_head, n_layer,
                 dim_Is, dim_hklss, dim_fs, dim_frac_coords, dim_disp_tensor, dim_qs):
        super(Transformer, self).__init__()
        
        # Define separate subnetworks for each input type
        self.embed_Is = SubNetwork(dim_Is, d)
        self.embed_hklss = SubNetwork(dim_hklss, d)
        self.embed_fs = SubNetwork(dim_fs, d)
        self.embed_frac_coords = SubNetwork(dim_frac_coords, d)
        self.embed_disp_tensor = SubNetwork(dim_disp_tensor, d)
        self.embed_qs = SubNetwork(dim_qs, d)
        
        # Transformer block
        self.transformer = nn.Sequential(*[TransformerBlock(d, n_head) for _ in range(n_layer)])
        self.layer_norm = nn.LayerNorm(d)
        self.fc = nn.Linear(d, 1)

    def forward(self, Is, hklss, fs, frac_coords, disp_tensor, qs):
        # Process each input type separately
        x_Is = self.embed_Is(Is)
        x_hklss = self.embed_hklss(hklss)
        x_fs = self.embed_fs(fs)
        x_frac_coords = self.embed_frac_coords(frac_coords)
        x_disp_tensor = self.embed_disp_tensor(disp_tensor)
        x_qs = self.embed_qs(qs)
        
        # Concatenate processed inputs along the feature dimension
        combined_input = torch.cat((x_Is, x_hklss, x_fs, x_frac_coords, x_disp_tensor, x_qs), dim=-1)
        
        # Pass through transformer layers
        x = self.transformer(combined_input.unsqueeze(0)).squeeze(0)
        x = self.layer_norm(x)
        x = self.fc(x)
        return x


# Define the Transformer model
class Transformer(nn.Module):
    def __init__(self, d, n_head, n_layer):
        super(Transformer, self).__init__()
        self.I_network = SubNetwork(Is_input_dim, d)
        self.hkls_network = SubNetwork(hklss_input_dim, d)
        self.fs_network = SubNetwork(fs_input_dim, d)
        self.frac_coords_network = SubNetwork(frac_coords_input_dim, d)
        self.disp_tensor_network = SubNetwork(disp_tensor_input_dim, d)
        self.qs_network = SubNetwork(qs_input_dim, d)

        self.layers = nn.Sequential(*[TransformerBlock(d, n_head) for _ in range(n_layer)])
        self.layer_norm = nn.LayerNorm(d)
        self.fc = nn.Linear(d, 1)

    def forward(self, Is, hklss, fs, frac_coords, disp_tensor, qs):
        # Process each input separately
        Is_embed = self.I_network(Is)
        hklss_embed = self.hkls_network(hklss)
        fs_embed = self.fs_network(fs)
        frac_coords_embed = self.frac_coords_network(frac_coords)
        disp_tensor_embed = self.disp_tensor_network(disp_tensor)
        qs_embed = self.qs_network(qs)

        # Combine the embeddings
        combined_input = torch.cat((Is_embed, hklss_embed, fs_embed, frac_coords_embed, disp_tensor_embed, qs_embed), dim=-1)
        
        x = self.layers(combined_input)
        x = self.layer_norm(x)
        x = self.fc(x)
        return x


# Initialize the model, optimizer, and hyperparameters
d = 128  # Dimension of the model, adjust as necessary
n_head = 8  # Number of attention heads
n_layer = 6  # Number of transformer layers
learning_rate = 1e-4
epochs = 10

model = Transformer(d, n_head, n_layer).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Training function using stochastic gradient descent
def train_sgd(model, optimizer, train_embeddings, target_data, epochs, device):
    model.train()
    
    for epoch in range(epochs):
        sample_ids = list(train_embeddings.keys())
        random.shuffle(sample_ids)  # Shuffle the order of samples for each epoch
        
        for sample_id in sample_ids:
            data = train_embeddings[sample_id]
            Is = data['Is'].to(device)
            hklss = data['hklss'].to(device)
            fs = data['fs'].to(device)
            frac_coords = data['frac_coords'].to(device)
            disp_tensor = data['disp_tensor'].to(device)
            qs = data['qs'].to(device)
            
            optimizer.zero_grad()
            output = model(Is, hklss, fs, frac_coords, disp_tensor, qs)
            
            target = target_data[sample_id].to(device)
            loss = F.mse_loss(output, target)
            loss.backward()
            optimizer.step()
        
        # Optional: Evaluate the model on the validation set after each epoch
        val_loss = estimate_loss(model, val_embeddings, device)
        print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss}")


# Function to estimate the loss on the validation set
def estimate_loss(model, val_embeddings, device):
    model.eval()
    loss_cumulative = 0.
    with torch.no_grad():
        for sample_id, data in val_embeddings.items():
            Is = data['Is'].to(device)
            hklss = data['hklss'].to(device)
            fs = data['fs'].to(device)
            frac_coords = data['frac_coords'].to(device)
            disp_tensor = data['disp_tensor'].to(device)
            qs = data['qs'].to(device)
            
            pred = model(Is, hklss, fs, frac_coords, disp_tensor, qs)
            target = data['frac_coords'].to(device)  # Assuming 'frac_coords' is the target
            
            loss = F.mse_loss(pred, target)
            loss_cumulative += loss.item()
    return loss_cumulative / len(val_embeddings)


# Run the training function
train_sgd()
"""

class SubNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SubNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return torch.relu(self.fc(x))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, n_head)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.layer_norm1(x + attn_output)
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + ff_output)
        return x

class Transformer(nn.Module):
    def __init__(self, d_model, n_head, n_layer, max_dims):
        super(Transformer, self).__init__()
        
        self.subnet_Is = SubNetwork(max_dims['Is'], d_model) if 'Is' in max_dims else None
        self.subnet_hklss = SubNetwork(max_dims['hklss'], d_model) if 'hklss' in max_dims else None
        self.subnet_fs = SubNetwork(max_dims['fs'], d_model) if 'fs' in max_dims else None
        self.subnet_frac_coords = SubNetwork(max_dims['frac_coords'], d_model) if 'frac_coords' in max_dims else None
        self.subnet_disp_tensor = SubNetwork(max_dims['disp_tensor'], d_model) if 'disp_tensor' in max_dims else None
        self.subnet_qs = SubNetwork(max_dims['qs'], d_model) if 'qs' in max_dims else None
        
        self.layers = nn.Sequential(*[TransformerBlock(d_model, n_head) for _ in range(n_layer)])
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, Is, hklss, fs, frac_coords, disp_tensor, qs):
        x_Is = self.subnet_Is(Is) if self.subnet_Is and Is is not None else 0
        x_hklss = self.subnet_hklss(hklss) if self.subnet_hklss and hklss is not None else 0
        x_fs = self.subnet_fs(fs) if self.subnet_fs and fs is not None else 0
        x_frac_coords = self.subnet_frac_coords(frac_coords) if self.subnet_frac_coords and frac_coords is not None else 0
        x_disp_tensor = self.subnet_disp_tensor(disp_tensor) if self.subnet_disp_tensor and disp_tensor is not None else 0
        x_qs = self.subnet_qs(qs) if self.subnet_qs and qs is not None else 0
        
        # print('x_Is: ', x_Is,"x_hklss: ", x_hklss,"x_fs: ", x_fs,"x_frac_coords: ", x_frac_coords,"x_disp_tensor: ", x_disp_tensor,"x_qs: ", x_qs)
        combined_input = x_Is + x_hklss + x_fs + x_frac_coords + x_disp_tensor + x_qs
        combined_input = combined_input.unsqueeze(0)  # Add batch dimension for transformer
        x = self.layers(combined_input)
        x = self.layer_norm(x)
        x = self.fc(x)
        return x

# Function to inspect data dimensions
def inspect_data_dimensions(data_dict):
    dimensions = {
        'Is': [],
        'hklss': [],
        'fs': [],
        'frac_coords': [],
        'disp_tensor': [],
        'qs': []
    }
    
    for sample_id, data in data_dict.items():
        for key in dimensions.keys():
            if key in data:
                dimensions[key].append(data[key].shape[-1])
    
    return dimensions

# Function to get maximum dimensions
def get_max_dimensions(dimensions):
    max_dims = {key: max(dims) for key, dims in dimensions.items() if dims}
    return max_dims

# Function to train the model using SGD
def train_sgd(model, optimizer, train_embeddings, target_data, epochs, device):
    model.train()
    
    for epoch in range(epochs):
        sample_ids = list(train_embeddings.keys())
        random.shuffle(sample_ids)  # Shuffle the order of samples for each epoch
        
        for sample_id in sample_ids:
            data = train_embeddings[sample_id]

            # Ensure all required keys are present in data
            Is = data['Is'].to(device) if 'Is' in data else None
            hklss = data['hklss'].to(device) if 'hklss' in data else None
            fs = data['fs'].to(device) if 'fs' in data else None
            frac_coords = data['frac_coords'].to(device) if 'frac_coords' in data else None
            disp_tensor = data['disp_tensor'].to(device) if 'disp_tensor' in data else None
            qs = data['qs'].to(device) if 'qs' in data else None

            optimizer.zero_grad()
            output = model(Is, hklss, fs, frac_coords, disp_tensor, qs)

            if sample_id not in target_data:
                print(f"Missing target for sample {sample_id}. Skipping this sample.")
                continue

            target = target_data[sample_id].to(device)
            loss = F.mse_loss(output, target)
            loss.backward()
            optimizer.step()
        
        # Optional: Evaluate the model on the validation set after each epoch
        val_loss = estimate_loss(model, val_embeddings, device)
        print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss}")

# Function to estimate the validation loss
def estimate_loss(model, val_embeddings, device):
    model.eval()
    loss_cumulative = 0.
    with torch.no_grad():
        for sample_id, data in val_embeddings.items():
            Is = data['Is'].to(device) if 'Is' in data else None
            hklss = data['hklss'].to(device) if 'hklss' in data else None
            fs = data['fs'].to(device) if 'fs' in data else None
            frac_coords = data['frac_coords'].to(device) if 'frac_coords' in data else None
            disp_tensor = data['disp_tensor'].to(device) if 'disp_tensor' in data else None
            qs = data['qs'].to(device) if 'qs' in data else None

            pred = model(Is, hklss, fs, frac_coords, disp_tensor, qs)
            target = data['frac_coords'].to(device)  # Assuming 'frac_coords' is the target

            loss = F.mse_loss(pred, target)
            loss_cumulative += loss.item()
    return loss_cumulative / len(val_embeddings)

# Example usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Inspect data dimensions
dimensions = inspect_data_dimensions(train_embeddings)

# Determine maximum dimensions
max_dims = get_max_dimensions(dimensions)

# Initialize model with maximum dimensions
d_model = 128  # Model dimension
n_head = 8
n_layer = 6

model = Transformer(d_model, n_head, n_layer, max_dims).to(device)

# Initialize optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


# Train the model
train_sgd(model, optimizer, train_embeddings, tr_target_data, epochs=20, device=device)







# # Inference test
# # Assuming the TransformerModel, CustomDataset, and other necessary components are already defined

# # Load the trained model (make sure to provide the correct path to the saved model)
# model_path = 'path_to_your_saved_model.pth'
# model = TransformerModel(d, n_head, n_layer)
# model.load_state_dict(torch.load(model_path))
# model.to(device)
# model.eval()

# # Prepare the input data for inference
# # Assuming you have new data similar to train_embeddings format
# # For demonstration, we'll use the existing train_embeddings

# # Convert the input data to the same format used in training
# inference_embeddings = train_embeddings  # Replace this with your actual inference data

# class InferenceDataset(Dataset):
#     def __init__(self, embeddings):
#         self.embeddings = list(embeddings.values())

#     def __len__(self):
#         return len(self.embeddings)

#     def __getitem__(self, idx):
#         return self.embeddings[idx]

# inference_dataset = InferenceDataset(inference_embeddings)
# inference_loader = DataLoader(inference_dataset, batch_size=1, shuffle=False)  # Batch size 1 for inference

# # Perform inference
# model.eval()
# predictions = []

# with torch.no_grad():
#     for batch in inference_loader:
#         inputs = batch.to(device)
#         outputs = model(inputs)
#         predictions.append(outputs.cpu().numpy())

# # Process the output
# # Assuming you want to convert predictions back to a format similar to the input
# # Here, predictions will be a list of numpy arrays

# for idx, prediction in enumerate(predictions):
#     print(f"Sample {idx + 1}: {prediction}")

# # Optionally, save the predictions to a file
# import numpy as np
# np.save('predictions.npy', predictions)




