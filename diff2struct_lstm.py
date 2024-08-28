import torch
import time
import os

from utils.data import create_pattern_data_sets, create_torch_data_dicts
from utils.lstm_model import PhaseLoss, Diff2StructLSTM, train

torch.set_default_dtype(torch.float64)

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

run_name = time.strftime('%y%m%d-%H%M%S', time.localtime())
# run_name = '240403-170951'

structure_data_path = '../structure_data/structure_data_2_atoms.pkl'

data_folder = './data/'
data_file = 'data.pkl'

model_folder = f'./models/{run_name}/'
torch_data_file = 'torch_data.pkl'

try:
    os.mkdir(model_folder)
except FileExistsError:
    pass

tr_ratio = 0.8
va_ratio = 0.1

ratios = [tr_ratio, va_ratio]

data_sets = create_pattern_data_sets(structure_data_path, data_folder, data_file, ratios)
tr_dict, va_dict, te_dict = create_torch_data_dicts(data_sets, model_folder, torch_data_file).values()
# print("tr_dict: ", tr_dict, "va_dict: ", va_dict, "te_dict: ", te_dict)

loss_fn = PhaseLoss()

max_iter = 100
lr = 0.001
weight_decay = 0.05
schedule_gamma = 0.96

# embedding network for each atomic scatterring factor (f)
f_emb_lyr = 3
f_inp_dim = 2       # fixed since f is a scalar
f_hid_dim = 6       # determine the embedding hidden state dimensions
                    # over writen by f_emb_dim if f_emb_lyr = 1
f_emb_dim = 6       # determine the main LSTM hidden state dimensions 
                    # over writen by main_out_dim if main_lyr = 1

# embedding network for each peak intensity (I)
I_emb_lyr = 3
I_inp_dim = 1       # fixed since I is a scalar
I_hid_dim = 6       # determine the embedding hidden state dimensions
                    # over writen by I_emb_dim if I_emb_lyr = 1
I_emb_dim = 6

# embedding network for each peak's scatterring vectors (q)
hkls_emb_lyr = 3
hkls_inp_dim = 3    # fixed since q is a vector in 3D
hkls_hid_dim = 6    # determine the embedding LSTM hidden state dimensions
                    # over writen by q_emb_dim if q_emb_lyr = 1
hkls_emb_dim = 6

# embedding network for combined embedding of I, and q into the input of main LSTM
inp_emb_lyr = 3
inp_inp_dim = I_emb_dim + hkls_emb_dim
inp_hid_dim = 12     # determine the combined embedding hidden state dimensions 
                    # over writen by inp_emb_dim if inp_emb_lyr = 1
inp_emb_dim = 12

# main LSTM network for the prediction
main_lyr = 3
main_inp_dim = inp_emb_dim
main_hid_dim = f_emb_dim
main_out_dim = 6    # fixed since atomic coordinates are vector in 3D

model = Diff2StructLSTM(f_emb_lyr,
                        f_inp_dim,
                        f_hid_dim,
                        f_emb_dim,
                        I_emb_lyr,
                        I_inp_dim,
                        I_hid_dim,
                        I_emb_dim,
                        hkls_emb_lyr,
                        hkls_inp_dim,
                        hkls_hid_dim,
                        hkls_emb_dim,
                        inp_emb_lyr,
                        inp_inp_dim,
                        inp_hid_dim,
                        inp_emb_dim,
                        main_lyr,
                        main_inp_dim,
                        main_hid_dim,
                        main_out_dim)

opt = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = weight_decay)
scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma = schedule_gamma)

train(model,
      opt,
      tr_dict,
      va_dict,
      loss_fn,
      run_name,
      max_iter,
      scheduler,
      device,
      model_folder)