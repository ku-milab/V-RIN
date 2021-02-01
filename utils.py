import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_recall_curve
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Define the function to print and write log
def writelog(file, line):
    file.write(line + '\n')
    print(line)

# Normalization
def normalize(data, mean, std):
    n_patients = data.shape[0]
    n_hours = data.shape[1]
    n_variables = data.shape[2]

    mask = ~np.isnan(data) * 1
    mask = mask.reshape(n_patients * n_hours, n_variables)
    measure = data.copy().reshape(n_patients * n_hours, n_variables)

    # Log Transform
    # measure[np.where(measure == 0)] = measure[np.where(measure == 0)] + 1e-10
    # measure[np.where(mask == 1)] = np.log(measure[np.where(mask == 1)])

    isnew = 0
    if len(mean) == 0 or len(std) == 0:
        isnew = 1
        mean_set = np.zeros([n_variables])
        std_set = np.zeros([n_variables])
    else:
        mean_set = mean
        std_set = std
    for v in range(n_variables):
        idx = np.where(mask[:,v] == 1)[0]

        if idx.sum()==0:
            continue

        if isnew:
            measure_mean = np.mean(measure[:, v][idx])
            measure_std = np.std(measure[:, v][idx])

            # Save the Mean & STD Set
            mean_set[v] = measure_mean
            std_set[v] = measure_std
        else:
            measure_mean = mean[v]
            measure_std = std[v]

        for ix in idx:
            if measure_std != 0:
                measure[:, v][ix] = (measure[:, v][ix] - measure_mean) / measure_std
            else:
                measure[:, v][ix] = measure[:, v][ix] - measure_mean

    normalized_data = measure.reshape(n_patients, n_hours, n_variables)

    return normalized_data, mean_set, std_set

def collate_fn(recs):

    def to_tensor_dict(recs):

        values = torch.FloatTensor(np.array([r['values'] for r in recs]))
        masks = torch.FloatTensor(np.array([r['masks'] for r in recs]))
        deltas = torch.FloatTensor(np.array([r['deltas'] for r in recs]))

        evals = torch.FloatTensor(np.array([r['evals'] for r in recs]))
        eval_masks = torch.FloatTensor(np.array([r['eval_masks'] for r in recs]))

        return {'values': values,
                'masks': masks,
                'deltas': deltas,
                'evals': evals,
                'eval_masks': eval_masks}

    ret_dict = to_tensor_dict(recs)

    ret_dict['labels'] = torch.FloatTensor(np.array([r['label'] for r in recs]))

    return ret_dict

def parse_delta(masks):
    [T, D] = masks.shape
    deltas = []

    for t in range(T):
        if t == 0:
            deltas.append(np.ones(D))
        else:
            deltas.append(np.ones(D) + (1 - masks[t]) * deltas[-1])

    return np.array(deltas)

# Define Sample Loader
def sample_loader(set, data, label, batch_size, removal_percent, task, shuffle=True):
    # Random seed
    np.random.seed(1)
    torch.manual_seed(1)

    # Get Dimensionality
    [N, T, D] = data.shape

    # Reshape
    data = data.reshape(N, T*D)

    recs = []
    for i in range(N):

        values = data[i].copy()
        if task == 'I' or set == 'train':
            if removal_percent != 0:
                # randomly eliminate 10% values as the imputation ground-truth
                indices = np.where(~np.isnan(data[i]))[0].tolist()
                indices = np.random.choice(indices, len(indices) // removal_percent)
                values[indices] = np.nan

        masks = ~np.isnan(values)
        eval_masks = (~np.isnan(values)) ^ (~np.isnan(data[i]))

        evals = data[i].reshape(T, D)
        values = values.reshape(T, D)

        masks = masks.reshape(T, D)
        eval_masks = eval_masks.reshape(T, D)

        rec = {}
        rec['label'] = label[i]

        deltas = parse_delta(masks) 

        rec['values'] = np.nan_to_num(values).tolist()
        rec['masks'] = masks.astype('int32').tolist()
        rec['evals'] = np.nan_to_num(evals).tolist()
        rec['eval_masks'] = eval_masks.astype('int32').tolist()
        rec['deltas'] = deltas.tolist()

        recs.append(rec)

    # Define the loader
    loader = DataLoader(recs,
                        batch_size=batch_size,
                        num_workers=1,
                        shuffle=shuffle,
                        pin_memory=True,
                        collate_fn=collate_fn)

    return loader

def collate_fn_bidirectional(recs):

    def to_tensor_dict(recs):

        values = torch.FloatTensor(np.array([r['values'] for r in recs]))
        masks = torch.FloatTensor(np.array([r['masks'] for r in recs]))
        deltas_f = torch.FloatTensor(np.array([r['deltas_f'] for r in recs]))
        deltas_b = torch.FloatTensor(np.array([r['deltas_b'] for r in recs]))

        evals = torch.FloatTensor(np.array([r['evals'] for r in recs]))
        eval_masks = torch.FloatTensor(np.array([r['eval_masks'] for r in recs]))

        return {'values': values,
                'masks': masks,
                'deltas_f': deltas_f,
                'deltas_b': deltas_b,
                'evals': evals,
                'eval_masks': eval_masks}

    ret_dict = to_tensor_dict(recs)

    ret_dict['labels'] = torch.FloatTensor(np.array([r['label'] for r in recs]))

    return ret_dict

def parse_delta_bidirectional(masks, direction):
    if direction == 'backward':
        masks = masks[::-1]

    [T, D] = masks.shape
    deltas = []

    for t in range(T):
        if t == 0:
            deltas.append(np.ones(D))
        else:
            deltas.append(np.ones(D) + (1 - masks[t]) * deltas[-1])

    return np.array(deltas)

# Define Sample Loader
def sample_loader_bidirectional(set, data, label, batch_size, removal_percent, task, shuffle=True):

    # Random seed
    np.random.seed(1)
    torch.manual_seed(1)

    # Get Dimensionality
    [N, T, D] = data.shape

    # Reshape
    data = data.reshape(N, T*D)

    recs = []
    for i in range(N):

        values = data[i].copy()
        if task == 'I' or set == 'train':
            if removal_percent != 0:
                # randomly eliminate 10% values as the imputation ground-truth
                indices = np.where(~np.isnan(data[i]))[0].tolist()
                indices = np.random.choice(indices, len(indices) // removal_percent)
                values[indices] = np.nan

        masks = ~np.isnan(values)
        eval_masks = (~np.isnan(values)) ^ (~np.isnan(data[i]))

        evals = data[i].reshape(T, D)
        values = values.reshape(T, D)

        masks = masks.reshape(T, D)
        eval_masks = eval_masks.reshape(T, D)

        rec = {}
        rec['label'] = label[i]

        deltas_f = parse_delta_bidirectional(masks, direction='forward')
        deltas_b = parse_delta_bidirectional(masks, direction='backward')

        rec['values'] = np.nan_to_num(values).tolist()
        rec['masks'] = masks.astype('int32').tolist()
        rec['evals'] = np.nan_to_num(evals).tolist()
        rec['eval_masks'] = eval_masks.astype('int32').tolist()
        rec['deltas_f'] = deltas_f.tolist()
        rec['deltas_b'] = deltas_b.tolist()

        recs.append(rec)

    # Define the loader
    loader = DataLoader(recs,
                        batch_size=batch_size,
                        num_workers=1,
                        shuffle=shuffle,
                        pin_memory=True,
                        collate_fn=collate_fn_bidirectional)

    return loader

def calculate_performance(y, y_score, y_pred):
    # Calculate Evaluation Metrics
    acc = accuracy_score(y_pred, y) * 100
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0,1]).ravel()
    if tp == 0 and fn == 0:
        sen = 0.0
        recall = 0.0
        auprc = 0.0
    else:
        sen = tp / (tp + fn)
        recall = tp / (tp + fn)
        p, r, t = precision_recall_curve(y, y_score)
        auprc = np.nan_to_num(metrics.auc(r, p))
    spec = np.nan_to_num(tn / (tn + fp))
    balacc = ((spec + sen) / 2) * 100
    if tp == 0 and fp == 0:
        prec = 0
    else:
        prec = np.nan_to_num(tp / (tp + fp))

    try:
        auc = roc_auc_score(y, y_score)
    except ValueError:
        auc = 0

    return auc, auprc, acc, balacc, sen, spec, prec, recall

def plot_imputation_mae(dir, x, m, ex, eimp, x_imp_vae, x_imp_rnn, x_hat, u_hat, epoch):
    fig, axes = plt.subplots(4, 2, sharey=True)

    # cx1 = axes[0, 0].imshow(x.T)
    cx1 = axes[0, 0].imshow(x.T, vmin=0, vmax=1)
    axes[0, 0].set_ylabel('Variables')
    axes[0, 0].title.set_text('x')
    axes[0, 0].set_xticks([])

    cx2 = axes[0, 1].imshow(m.T, vmin=0, vmax=1)
    axes[0, 1].set_xlabel('')
    axes[0, 1].title.set_text('m')
    axes[0, 1].set_xticks([])

    cx3 = axes[1, 0].imshow(ex.T, vmin=0, vmax=1)
    axes[1, 0].set_ylabel('Variables')
    axes[1, 0].set_xlabel('')
    axes[1, 0].title.set_text('Eval X')
    axes[1, 0].set_xticks([])

    cx4 = axes[1, 1].imshow(eimp.T, vmin=0, vmax=1)
    axes[1, 1].set_xlabel('')
    axes[1, 1].title.set_text('Imputed X')
    axes[1, 1].set_xticks([])

    cx5 = axes[2, 0].imshow(x_hat.T, vmin=0, vmax=1)
    axes[2, 0].set_ylabel('Variables')
    axes[2, 0].set_xlabel('')
    axes[2, 0].title.set_text(r'$\hat{x}$ (VAE)')
    axes[2, 0].set_xticks([])

    cx6 = axes[2, 1].imshow(u_hat.T, vmin=0, vmax=1)
    axes[2, 1].set_xlabel('')
    axes[2, 1].title.set_text(r'$\bar{u}$ (VAE)')
    axes[2, 1].set_xticks([])

    cx7 = axes[3, 0].imshow(x_imp_vae.T, vmin=0, vmax=1)
    axes[3, 0].set_ylabel('Variables')
    axes[3, 0].set_xlabel('Hours')
    axes[3, 0].title.set_text(r'$\bar{x}$')

    cx8 = axes[3, 1].imshow(x_imp_rnn.T, vmin=0, vmax=1)
    axes[3, 1].set_xlabel('Hours')
    axes[3, 1].title.set_text(r'$x^c$ (VAE+RNN)')

    fig.colorbar(cx1, ax=axes.ravel().tolist(), orientation='vertical')
    stre = str(epoch)
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.75, top=0.95, bottom=0.1)
    plt.savefig(dir + '/x_' + stre.rjust(4, '0') + '.png',
                bbox_inches='tight')
    plt.close('all')
    plt.clf()




