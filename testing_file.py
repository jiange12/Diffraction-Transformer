import torch
import torch.nn as nn

class EmbeddingModel(nn.Module):
    def __init__(self, embed_dim, rnn_hidden_dim, rnn_layers):
        super(EmbeddingModel, self).__init__()
        self.embed_dim = embed_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_layers = rnn_layers
        
        # LSTM layer for multi-dimensional input
        self.lstm_disp = nn.LSTM(input_size=3, hidden_size=rnn_hidden_dim, num_layers=rnn_layers, batch_first=True)
        
        # Fully connected layer to map LSTM output to embedding size for disp_tensor
        self.fc_disp = nn.Linear(rnn_hidden_dim, embed_dim)
        
        # Linear layers for one-dimensional inputs (Is, fs, qs)
        self.fc_Is = nn.Linear(1, embed_dim)  # Directly map 1D input to the embedding dimension
        self.fc_fs = nn.Linear(1, embed_dim)
        self.fc_qs = nn.Linear(1, embed_dim)
        
    def forward(self, disp_tensor, Is, fs, qs):
        # Processing disp_tensor with LSTM
        disp_out, _ = self.lstm_disp(disp_tensor)
        disp_embedding = self.fc_disp(disp_out[:, -1, :])  # Use the last output of LSTM
        disp_embedding = disp_embedding.flatten()  # Flatten to a 1D vector
        # print("disp_embedding:", disp_embedding.shape)

        # Processing Is with a linear layer
        Is = Is.unsqueeze(-1)  # Adding extra dimension for linear layer input
        Is_embedding = self.fc_Is(Is)
        Is_embedding = Is_embedding.flatten()  # Flatten to a 1D vector
        # print("Is_embedding:", Is_embedding.shape)

        # Processing fs with a linear layer
        fs = fs.unsqueeze(-1)  # Adding extra dimension for linear layer input
        fs_embedding = self.fc_fs(fs)
        fs_embedding = fs_embedding.flatten()  # Flatten to a 1D vector
        # print("fs_embedding:", fs_embedding.shape)

        # Processing qs with a linear layer
        qs = qs.unsqueeze(-1)  # Adding extra dimension for linear layer input
        qs_embedding = self.fc_qs(qs)
        qs_embedding = qs_embedding.flatten()  # Flatten to a 1D vector
        # print("qs_embedding:", qs_embedding.shape)
        
        # Concatenating all embeddings into a single vector
        embedding = torch.cat([disp_embedding, Is_embedding, fs_embedding, qs_embedding], dim=-1)
        # print("Concatenated embedding:", embedding.shape)
        
        return embedding

# Example usage
def create_embeddings(data, embed_dim, rnn_hidden_dim, rnn_layers):
    model = EmbeddingModel(embed_dim, rnn_hidden_dim, rnn_layers)
    embeddings = {}
    # print("data: ", data)
    for key, value in data.items():
        # print("key: ", key, "value: ", value)
        disp_tensor, Is, fs, qs = value
        embedding = model(disp_tensor, Is, fs, qs)
        embeddings[key] = embedding
    return embeddings

# Example data structure
data_example = {
    'sample1': (
        torch.tensor([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]]),  # disp_tensor
        torch.tensor([1.0, 2.0, 3.0]),  # Is
        torch.tensor([0.5, 0.6]),  # fs
        torch.tensor([0.4, 0.3, 0.2])  # qs
    )
}

# Create embeddings
# embed_dim should be large enough to accommodate the concatenated embeddings
embeddings = create_embeddings(data_example, embed_dim=128, rnn_hidden_dim=64, rnn_layers=2)
print(embeddings)
