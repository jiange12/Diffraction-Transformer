import os
import glob
import pandas as pd
import pickle as pkl
import torch
from torch.utils.data import random_split
from torch_geometric.data import Data

from utils.neutron_diffraction import NeutronDiffraction

torch.set_default_dtype(torch.float64)

def create_pattern_data_sets(structure_data_path, data_folder, data_file, ratios, wavelength = 1):

    NDC = NeutronDiffraction(wavelength = wavelength)

    data_path = os.path.join(data_folder, data_file)
    if len(glob.glob(data_folder)) == 0:
        data = pd.read_pickle(structure_data_path)
        Iss, hklsss, fss, frac_coordss, qss = [], [], [], [], []

        N = len(data)
        exclude = []
        for i, structure in enumerate(data.structure):
            try:
                Is, hklss, fs, frac_coords, qs = NDC.get_pattern(structure)
                Iss.append(Is)
                hklsss.append(hklss)
                fss.append(fs)
                frac_coordss.append(frac_coords)
                qss.append(qs)
            except:
                exclude.append(i)

            print(f'num {i+1:4d}/{N}', end = '\r')
        print('')

        data = data.drop(exclude).reset_index(drop=True)
        data['Is'] = Iss
        data['hklss'] = hklsss
        data['fs'] = fss
        data['frac_coords'] = frac_coordss
        data['qs'] = qss

        num_data = len(data)
        num_tr = int(num_data*ratios[0])
        num_va = int(num_data*ratios[1])
        num_te = num_data - num_tr - num_va

        tr, va, te = random_split(data, [num_tr, num_va, num_te])

        tr_data = data.iloc[tr.indices]
        va_data = data.iloc[va.indices]
        te_data = data.iloc[te.indices]

        data_sets = {'tr': tr_data, 'va': va_data, 'te': te_data}

        pkl.dump(data_sets, open(data_path, 'wb'))
        return data_sets
    else:
        return pkl.load(open(data_path, 'rb'))

def build_data(data):
    frac_coords = torch.from_numpy(data.frac_coords)
    disp_coords = frac_coords.reshape(-1, 1, 3)
    disp_tensor = disp_coords - disp_coords.transpose(0, 1)

    return Data(id = data.id,
                formula = data.formula,
                structure = data.structure,
                Is = torch.tensor(data.Is)[:5], # Peak height
                hklss = [torch.stack([torch.tensor(hkl, dtype = torch.float64) for hkl in hkls]) for hkls in data.hklss][:5],
                fs = torch.from_numpy(data.fs),
                frac_coords = frac_coords,
                disp_tensor = disp_tensor,
                qs = torch.tensor(data.qs)) # Peak position

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
                print(f'num {i+1:4d}/{N}', end = '\r')
            print('')

            torch_data[set_name] = set_data

        pkl.dump(torch_data, open(torch_data_path, 'wb'))
        return torch_data
    else:
        return pkl.load(open(torch_data_path, 'rb'))