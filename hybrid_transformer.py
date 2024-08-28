import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import time
import os
from utils.data import create_pattern_data_sets, create_torch_data_dicts
import glob
import pickle as pkl
from torch_geometric.data import Data

"""
class CustomDataset(Dataset):
    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.keys = list(data_dict.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        return self.data_dict[key]

class InputProcessing(nn.Module):
    def __init__(self, embed_dim):
        super(InputProcessing, self).__init__()
        self.disp_tensor_cnn = nn.Conv1d(in_channels=6, out_channels=embed_dim, kernel_size=1)
        self.Is_rnn = nn.LSTM(input_size=1, hidden_size=embed_dim, batch_first=True)
        self.qs_rnn = nn.LSTM(input_size=1, hidden_size=embed_dim, batch_first=True)
        self.fs_fc = nn.Linear(2, embed_dim)

    def forward(self, disp_tensor, Is, qs, fs):
        disp_tensor = disp_tensor.view(disp_tensor.size(0), -1, 3)  # Flatten to 2D
        disp_tensor_emb = self.disp_tensor_cnn(disp_tensor)
        disp_tensor_emb = F.relu(disp_tensor_emb)
        
        Is = Is.unsqueeze(-1)  # Add feature dimension
        _, (Is_emb, _) = self.Is_rnn(Is)
        Is_emb = Is_emb.squeeze(0)
        
        qs = qs.unsqueeze(-1)  # Add feature dimension
        _, (qs_emb, _) = self.qs_rnn(qs)
        qs_emb = qs_emb.squeeze(0)
        
        fs_emb = self.fs_fc(fs)

        return disp_tensor_emb, Is_emb, qs_emb, fs_emb

class CombineEmbeddings(nn.Module):
    def __init__(self, embed_dim):
        super(CombineEmbeddings, self).__init__()
        self.embed_dim = embed_dim
        self.fc = nn.Linear(embed_dim * 4, embed_dim)

    def forward(self, disp_tensor_emb, Is_emb, qs_emb, fs_emb):
        combined = torch.cat([disp_tensor_emb, Is_emb, qs_emb, fs_emb], dim=-1)
        combined = self.fc(combined)
        return combined


class Transformer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super(Transformer, self).__init__()
        self.embedding_layer = nn.Linear(embed_dim, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, 6)  # Output dimension for frac_coord (2x3)

    def forward(self, x):
        x = self.embedding_layer(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = self.fc_out(x)
        return x


import math

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


class FullModel(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super(FullModel, self).__init__()
        self.input_processing = InputProcessing(embed_dim)
        self.combine_embeddings = CombineEmbeddings(embed_dim)
        self.transformer = Transformer(embed_dim, num_heads, num_layers)

    def forward(self, disp_tensor, Is, qs, fs):
        disp_tensor_emb, Is_emb, qs_emb, fs_emb = self.input_processing(disp_tensor, Is, qs, fs)
        combined_embedding = self.combine_embeddings(disp_tensor_emb, Is_emb, qs_emb, fs_emb)
        combined_embedding = combined_embedding.unsqueeze(1)  # Add sequence dimension
        output = self.transformer(combined_embedding)
        return output.squeeze(1)  # Remove sequence dimension

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
"""

class CustomDataset(Dataset):
    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.keys = list(data_dict.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        return self.data_dict[key]

# Input Processing Function
def process_inputs(data):
    disp_tensor = data.disp_tensor.view(-1)  # Flatten to 1D
    Is = data.Is
    hklss = torch.cat([torch.tensor(hkl, dtype=torch.float64).view(-1) for hkl in data.hklss])
    fs = data.fs
    qs = data.qs.view(-1)
    return disp_tensor, Is, hklss, fs, qs

# Embedding Model with RNNs/LSTMs
class EmbeddingModel(nn.Module):
    def __init__(self, embed_dim, rnn_hidden_dim, rnn_layers):
        super(EmbeddingModel, self).__init__()
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_layers = rnn_layers

        self.Is_rnn = nn.LSTM(input_size=1, hidden_size=rnn_hidden_dim, num_layers=rnn_layers, batch_first=True)
        self.hklss_rnn = nn.LSTM(input_size=1, hidden_size=rnn_hidden_dim, num_layers=rnn_layers, batch_first=True)
        self.qs_rnn = nn.LSTM(input_size=1, hidden_size=rnn_hidden_dim, num_layers=rnn_layers, batch_first=True)
        
        self.fs_embed = nn.Linear(1, embed_dim)
        self.disp_tensor_embed = nn.Linear(6, embed_dim)

        self.fc = nn.Linear(rnn_hidden_dim, embed_dim)

    def forward(self, disp_tensor, Is, hklss, fs, qs):
        disp_tensor_embedded = self.disp_tensor_embed(disp_tensor)

        Is = Is.unsqueeze(-1)
        _, (Is_h, _) = self.Is_rnn(Is)
        Is_embedded = self.fc(Is_h[-1])

        hklss = hklss.unsqueeze(-1)
        _, (hklss_h, _) = self.hklss_rnn(hklss)
        hklss_embedded = self.fc(hklss_h[-1])

        fs_embedded = self.fs_embed(fs.unsqueeze(-1))

        qs = qs.unsqueeze(-1)
        _, (qs_h, _) = self.qs_rnn(qs)
        qs_embedded = self.fc(qs_h[-1])

        return disp_tensor_embedded, Is_embedded, hklss_embedded, fs_embedded, qs_embedded

# Full Transformer Model
class FullTransformerModel(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, rnn_hidden_dim, rnn_layers):
        super(FullTransformerModel, self).__init__()
        self.embedding_model = EmbeddingModel(embed_dim, rnn_hidden_dim, rnn_layers)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(embed_dim, 6)  # Predicting frac_coords (6 coordinates for 2 atoms, 3 each)

    def forward(self, disp_tensor, Is, hklss, fs, qs):
        disp_tensor_embedded, Is_embedded, hklss_embedded, fs_embedded, qs_embedded = self.embedding_model(
            disp_tensor, Is, hklss, fs, qs
        )

        combined = torch.stack([
            disp_tensor_embedded,
            Is_embedded,
            hklss_embedded,
            fs_embedded,
            qs_embedded
        ], dim=0)

        transformer_output = self.transformer(combined.unsqueeze(1))  # Add batch dimension
        output = self.output_layer(transformer_output.mean(dim=0))  # Aggregate the transformer output

        return output

# Example Usage
# def create_torch_data_dicts(data_sets):
#     torch_data = dict()
    
#     for set_name, data_set in data_sets.items():
#         set_data = dict()
#         for i, (key, data) in enumerate(data_set.items()):
#             if len(data.frac_coords) == 1:
#                 continue
#             set_data[key] = process_inputs(data)
#             print(f'Processing {set_name} data: {i+1}/{len(data_set)}', end='\r')
#         torch_data[set_name] = set_data

#     return torch_data

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
                target=None)

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

def create_embeddings(torch_data, embed_dim, rnn_hidden_dim, rnn_layers):
    model = FullTransformerModel(embed_dim, num_heads=8, num_layers=6, rnn_hidden_dim=rnn_hidden_dim, rnn_layers=rnn_layers)
    embeddings = {'tr': {}, 'va': {}, 'te': {}}

    for dataset in ['tr', 'va', 'te']:
        dataset_obj = CustomDataset(torch_data[dataset])
        dataloader = DataLoader(dataset_obj, batch_size=1, shuffle=False)

        for i, data in enumerate(dataloader):
            disp_tensor, Is, hklss, fs, qs = data
            output = model(disp_tensor.squeeze(0), Is.squeeze(0), hklss.squeeze(0), fs.squeeze(0), qs.squeeze(0))
            
            embeddings[dataset][i] = {
                'output': output.squeeze(0),
                'frac_coords': torch_data[dataset][list(torch_data[dataset].keys())[i]][4]  # frac_coords
            }

    return embeddings

# Example Data Structure
torch.set_default_dtype(torch.float64)

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

run_name = time.strftime('%y%m%d-%H%M%S', time.localtime())

structure_data_path = '../structure_data/structure_data_2_atoms.pkl'

data_folder = "C:/Users/ejian/Summer 2024 UROP/X-Ray diffraction prediction models/01_diff2struct_lstm_2atom_test (Earth's version)/data"
data_file = 'data.pkl'

model_folder = f"C:/Users/ejian/Summer 2024 UROP/X-Ray diffraction prediction models/01_diff2struct_lstm_2atom_test (Earth's version)/models/{run_name}/"
torch_data_file = "torch_data.pkl"

try:
    os.mkdir(model_folder)
except FileExistsError:
    pass

tr_ratio = 0.8
va_ratio = 0.1

ratios = [tr_ratio, va_ratio]

data_sets = create_pattern_data_sets(structure_data_path, data_folder, data_file, ratios)
# Build the Torch Data Dictionaries
torch_data = create_torch_data_dicts(data_sets, model_folder, torch_data_file)
print(torch_data)
# Create Embeddings
embeddings = create_embeddings(torch_data, embed_dim=128, rnn_hidden_dim=64, rnn_layers=2)
print(embeddings)

