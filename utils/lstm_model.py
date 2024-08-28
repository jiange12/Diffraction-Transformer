import torch
import math
import time
import os
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torch_geometric.loader import DataLoader

torch.autograd.set_detect_anomaly(True)

# class PhaseLoss(_Loss):
#     '''
#     Diaplacement Matrix
#     '''
#     def __init__(self, size_average = None, reduce = None, reduction: str = 'mean') -> None:
#         super(PhaseLoss, self).__init__(size_average, reduce, reduction)

#     def forward(self, input, target):
#         return torch.sum(torch.pow(torch.abs(input - target), 2))/torch.numel(target)

class PhaseLoss(_Loss):
    '''
    Same Translation
    '''
    def __init__(self, size_average = None, reduce = None, reduction: str = 'mean') -> None:
        super(PhaseLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        disp = target - input
        disp_zero = disp - disp[0]
        disp_zero = torch.remainder(disp_zero, 1.0)
        disp_inv = 1.0 - disp_zero
        disp = torch.minimum(disp_zero, disp_inv)
        return torch.sum(torch.pow(disp, 2))/torch.numel(target)

class EmbedLinear(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super().__init__()
        self.inp_dim = inp_dim
        self.out_dim = out_dim

        self.W = nn.Parameter(torch.Tensor(inp_dim, out_dim))
        self.b = nn.Parameter(torch.Tensor(1, out_dim))

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.out_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return x @ self.W + self.b

class EmbedLSTM(nn.Module):
    def __init__(self, inp_dim, hid_dim):
        super().__init__()
        self.inp_dim = inp_dim
        self.hid_dim = hid_dim

        self.W_xg = nn.Parameter(torch.Tensor(inp_dim, 4*hid_dim))
        self.W_hg = nn.Parameter(torch.Tensor(hid_dim, 4*hid_dim))
        self.b_g  = nn.Parameter(torch.Tensor(1, 4*hid_dim))

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hid_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        seq_dim, _ = x.size()
        hid_seq = []

        h_t = torch.zeros(1, self.hid_dim).to(x.device)
        c_t = torch.zeros(1, self.hid_dim).to(x.device)

        hid_dim = self.hid_dim
        for t in range(seq_dim):
            x_t = x[t, :].unsqueeze(0)
            gates = x_t @ self.W_xg + h_t @ self.W_hg + self.b_g

            o_t = torch.sigmoid(gates[:, :hid_dim])
            f_t = torch.sigmoid(gates[:, hid_dim:2*hid_dim])
            i_t = torch.sigmoid(gates[:, 2*hid_dim:3*hid_dim])
            g_t = torch.tanh(gates[:, 3*hid_dim:])

            c_t = f_t * c_t + i_t * g_t

            h_t = o_t * torch.tanh(c_t)
            hid_seq.append(h_t)
        
        hid_seq = torch.cat(hid_seq, dim = 0)
        hid_seq = hid_seq.contiguous()
        return hid_seq

class MainLSTM(nn.Module):
    def __init__(self, inp_dim, hid_dim):
        super().__init__()
        self.inp_dim = inp_dim
        self.hid_dim = hid_dim

        self.W_xg = nn.Parameter(torch.Tensor(inp_dim, 4*hid_dim))
        self.W_hg = nn.Parameter(torch.Tensor(hid_dim, 4*hid_dim))
        self.b_g  = nn.Parameter(torch.Tensor(1, 4*hid_dim))

        self.W_cr = nn.Parameter(torch.Tensor(hid_dim, hid_dim))
        self.W_hr = nn.Parameter(torch.Tensor(hid_dim, hid_dim))
        self.b_r  = nn.Parameter(torch.Tensor(1, hid_dim))

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hid_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, init_states = None):
        seq_dim, _ = x.size()
        hid_seq = []

        if init_states is None:
            h_t = torch.zeros(1, self.hid_dim).to(x.device)
            c_t = torch.zeros(1, self.hid_dim).to(x.device)
        else:
            h_t, c_t = init_states
        hid_dim = self.hid_dim
        for t in range(seq_dim):
            x_t = x[t, :].unsqueeze(0)
            gates = x_t @ self.W_xg + h_t @ self.W_hg + self.b_g

            gate_o = gates[:, :hid_dim]
            # gate_c = torch.sum(gates[:, hid_dim:], axis = 0, keepdim = True)
            gate_c = gates[:, hid_dim:]
            o_t = torch.sigmoid(gate_o)

            f_t = torch.sigmoid(gate_c[:, :hid_dim])
            i_t = torch.sigmoid(gate_c[:, hid_dim:2*hid_dim])
            g_t = torch.tanh(gate_c[:, 2*hid_dim:])

            c_t = f_t * c_t + i_t * g_t

            h_t = o_t * torch.tanh(h_t @ self.W_hr + c_t @ self.W_cr + self.b_r)
            hid_seq.append(h_t)
        
        hid_seq = torch.stack(hid_seq, dim = 0)
        hid_seq = hid_seq.contiguous()
        return hid_seq
    
class Diff2StructLSTM(nn.Module):
    def __init__(self,
                 f_emb_lyr,
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
                 main_out_dim
                 ):
        super().__init__()
        
        self.I_network = nn.Sequential(*self.stacking(EmbedLinear, I_emb_lyr, I_inp_dim, I_hid_dim, I_emb_dim, nn.ReLU()))
        self.hkls_network = nn.Sequential(*self.stacking(EmbedLSTM, hkls_emb_lyr, hkls_inp_dim, hkls_hid_dim, hkls_emb_dim, nn.ReLU()))
        self.inp_network = nn.Sequential(*self.stacking(EmbedLinear, inp_emb_lyr, inp_inp_dim, inp_hid_dim, inp_emb_dim, nn.ReLU()))

        self.f_networks = nn.ModuleList([nn.Sequential(*self.stacking(EmbedLinear, f_emb_lyr, f_inp_dim, f_hid_dim, f_emb_dim, nn.ReLU())) for _ in range(main_lyr)])
        self.main_network = nn.ModuleList(self.stacking(MainLSTM, main_lyr, main_inp_dim, main_hid_dim, main_out_dim))

    def stacking(self, Class, lyr, inp_dim, hid_dim, out_dim, nonlinear = None):
        Layers = []
        if lyr == 1:
            Layers.append(Class(inp_dim, out_dim))
        else:
            Layers.append(Class(inp_dim, hid_dim))
            if nonlinear:
                Layers.append(nonlinear)
            for _ in range(lyr - 2):
                Layers.append(Class(hid_dim, hid_dim))
                if nonlinear:
                    Layers.append(nonlinear)
            Layers.append(Class(hid_dim, out_dim))
        return Layers

    def forward(self, data):
        Is = data.Is
        hklss = data.hklss
        fs = data.fs.reshape(-1, 2)
        I_embed = torch.relu(torch.vstack([self.I_network(I.unsqueeze(0)) for I in Is]))
        hkls_embed = torch.relu(torch.vstack([self.hkls_network(hkls)[-1] for hkls in hklss]))
        
        x = self.inp_network(torch.concat([I_embed.reshape(-1, 6), hkls_embed.reshape(-1, 6)], dim = 1))

        f_embeds = [torch.relu(f_network(fs)) for f_network in self.f_networks]
        for main_layer, f_embed in zip(self.main_network, f_embeds):
            if len(x.shape) == 3:
                x = torch.sum(x, dim = 1)
            x = torch.relu(x)
            x = main_layer(x, (f_embed, torch.zeros(f_embed.shape).to(f_embed.device)))
        x = x.reshape(-1, 2, 3)
        return torch.tanh(x[-1])
    
def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    loss_cumulative = 0.
    with torch.no_grad():
        for data in dataloader:
            data.to(device)
            pred_frac_coords = model(data)
            loss = loss_fn(pred_frac_coords, data.frac_coords).cpu()
            loss_cumulative += loss.detach().item()
    return loss_cumulative/len(dataloader)

def loglinspace(rate, step, end=None):
    t = 0
    while end is None or t <= end:
        yield t
        t = int(t + 1 + step*(1 - math.exp(-t*rate/step)))

def train(model,
          opt,
          tr_dict,
          va_dict,
          loss_fn,
          run_name,
          max_iter,
          scheduler,
          device,
          model_folder):
    
    from utils.plot import loss_train_plot, model_prediction

    model.to(device)
    checkpoint_generator = loglinspace(0.3, 5)
    checkpoint = next(checkpoint_generator)
    start_time = time.time()

    record_lines = []

    try:
        print('Use model.load_state_dict to load the existing model: ' + run_name + '.torch')
        results = torch.load(os.path.join(model_folder, run_name + '.torch'))
        history = results['history']
        s0 = history[-1]['step'] + 1
        model.load_state_dict(['state'])
    except:
        print('There is no existing model')
        results = {}
        history = []
        s0 = 0
        
    tr_loader = DataLoader(list(tr_dict.values()), shuffle = True)
    va_loader = DataLoader(list(va_dict.values()), shuffle = True)

    for step in range(max_iter):
        model.train()
        N = len(tr_loader)
        for i, data in enumerate(tr_loader):
            start = time.time()
            data.to(device)
            pred_frac_coords = model(data)
            loss = loss_fn(pred_frac_coords, data.frac_coords).cpu()
            opt.zero_grad()
            loss.backward()
            opt.step()

            print(f'num {i+1:4d}/{N}, loss = {loss:8.20f}, train time = {(time.time() - start):8.20f}', end = '\r')
        print('')

        end_time = time.time()
        epoch_time = end_time - start_time
        print(epoch_time)

        if step == checkpoint:
            checkpoint = next(checkpoint_generator)
            assert checkpoint > step

            train_avg_loss = evaluate(model, tr_loader, loss_fn, device)
            valid_avg_loss = evaluate(model, va_loader, loss_fn, device)

            history.append({
                            'step': s0 + step,
                            'epoch_time': epoch_time,
                            'batch': {
                                    'loss': loss.item(),
                                    },
                            'valid': {
                                    'loss': valid_avg_loss,
                                    },
                            'train': {
                                    'loss': train_avg_loss,
                                    },
                           })

            results = {
                        'history': history,
                        'state': model.state_dict()
                      }

            print(f'Iteration {step+1:4d}   ' +
                  f'train loss = {train_avg_loss:8.20f}   ' +
                  f'valid loss = {valid_avg_loss:8.20f}   ' +
                  f'elapsed time = {time.strftime("%H:%M:%S", time.gmtime(epoch_time))}')

            with open(os.path.join(model_folder, run_name + '.torch'), 'wb') as f: #
                torch.save(results, f) #

            record_line = '%d\t%.20f\t%.20f'%(step,train_avg_loss,valid_avg_loss) #
            record_lines.append(record_line) #

            loss_train_plot(model_folder, run_name, device)

            model_prediction(model, tr_loader, loss_fn, 4, 4, model_folder, 'train', device)
            model_prediction(model, va_loader, loss_fn, 4, 4, model_folder, 'valid', device)
        
        text_file = open(os.path.join(model_folder, run_name + '.txt'), 'w')
        for line in record_lines:
            text_file.write(line + '\n')
        text_file.close()

        if scheduler is not None:
            scheduler.step()