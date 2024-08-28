import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde

def loss_train_plot(model_folder, model_file, device):
    history = torch.load(os.path.join(model_folder, model_file + '.torch'), map_location = device)['history']
    steps = [d['step'] + 1 for d in history]
    loss_train = [d['train']['loss'] for d in history]
    loss_valid = [d['valid']['loss'] for d in history]

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    ax.plot(steps, loss_train, 'o-', label = 'Train', linewidth = 3)
    ax.plot(steps, loss_valid, 'o-', label = 'Valid', linewidth = 3)
    ax.set_xlabel('epochs []', fontsize = 26)
    ax.set_ylabel('loss []', fontsize = 26)
    ax.tick_params(axis = 'both', which = 'both', top = False, right = False, width = 1.5, length = 5, direction = 'in', labelsize = 26)
    ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0), useMathText = True)
    ax.yaxis.get_offset_text().set_size(26)
    ax.legend(fontsize = 26, loc = 'upper right')

    fig.tight_layout()
    fig.savefig(os.path.join(model_folder, 'loss_train_valid.png'), dpi = 300)
    plt.close()

def loss_test_plot(model, te_loader, loss_fn, model_folder, device):
    loss_test = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for data in te_loader:
            data.to(device)
            pred_frac_coords = model(data)
            loss = loss_fn(pred_frac_coords, data.frac_coords).cpu()
            loss_test.append(loss)

    loss_test = np.array(loss_test)

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    ax.plot(loss_test, 'o', label = 'Test: ' + str(np.round(np.mean(loss_test), 8)))
    ax.set_xlabel('data point []', fontsize = 26)
    ax.set_ylabel('loss []', fontsize = 26)
    ax.tick_params(axis = 'both', which = 'both', top = False, right = False, width = 1.5, length = 5, direction = 'in', labelsize = 26)
    ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0), useMathText = True)
    ax.yaxis.get_offset_text().set_size(26)
    ax.legend(fontsize = 26, loc = 'upper right')

    fig.tight_layout()
    fig.savefig(os.path.join(model_folder, 'loss_test.png'), dpi = 300)
    plt.close()

def predict(model, dataloader, loss_fn, device):
    with torch.no_grad():
        results = []
        for data in dataloader:
            data.to(device)
            pred_frac_coords = model(data)
            loss = loss_fn(pred_frac_coords, data.frac_coords).cpu()
            output = pred_frac_coords.cpu().numpy()
            target = data.frac_coords.cpu().numpy()
            
            results.append({'key': data.id, 'name': data.formula, 'loss': loss.item(), 'target': target, 'output': output})

    return pd.DataFrame(results)

def model_prediction(model, dataloader, loss_fn, quan_in, n, model_folder, plot_title, device):
    quan = quan_in
    results = predict(model, dataloader, loss_fn, device)
    sorted_results = results.sort_values(by = ['loss'])
    while True:
        try:
            quantiles = np.quantile(sorted_results.loss.values, [float(i+1)/quan for i in range(quan)])
            iquan = [0] + [np.argmin(np.abs(np.array(sorted_results.loss.values) - np.array(k))) for k in quantiles]
            selected_results = np.concatenate([np.sort(np.random.choice(np.arange(iquan[k], iquan[k + 1], 1), size = n, replace = False)) for k in range(quan)])
            break
        except ValueError:
            quan -= 1

    fig = plt.figure(figsize = (24, 21))
    gs = GridSpec(quan, n + 1, figure = fig)
    ax = fig.add_subplot(gs[:, 0])

    y_min, y_max = sorted_results.loss.min(), sorted_results.loss.max()
    y = np.linspace(y_min, y_max, 500)
    kde = gaussian_kde(sorted_results.loss)
    p = kde.pdf(y)
    ax.plot(p, y, color='black')
    qs =  list(quantiles)[::-1] + [0]
    for i in range(len(qs)-1):
        ax.fill_between([p.min(), p.max()], y1=[qs[i], qs[i]], y2=[qs[i+1], qs[i+1]],lw=0, alpha=0.5)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_ylabel('Loss', fontsize = 20)
    ax.set_xlim([p.min(), p.max()])
    ax.set_ylim([y_max, y_min])

    for k in range(quan*n):
        ax = fig.add_subplot(gs[k//n, 1 + k%n], projection = '3d')
        i = selected_results[k]
        
        pred = sorted_results.iloc[i].output.reshape(-1, 3)
        pred -= pred[0]
        pred = np.mod(pred, 1.0)
        real = sorted_results.iloc[i].target
        real -= real[0]
        real = np.mod(real, 1.0)
        print(real)
        print(pred)

        ax.scatter(real[:, 0], real[:, 1], real[:, 2], marker = 'o', s = 50, color = 'b')
        ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], marker = 'v', s = 50, color = 'r')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)

        ax.set_title(f'{sorted_results.iloc[i].key[0]}: {sorted_results.iloc[i].name}\nloss: {round(sorted_results.iloc[i].loss, 5)}', fontsize = 20) 

    fig.suptitle(plot_title, fontsize = 20)   
    fig.savefig(os.path.join(model_folder, f'{plot_title}_prediction.png'), dpi = 300)
    plt.close()