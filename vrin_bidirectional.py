import pickle
import torch.optim as optim
import torch.nn as nn
from models import VAE, RIN
from losses import SVAELoss
from utils import *
import os
import random
import datetime
import numpy as np
import argparse
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

# Define Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id", type=str, default='0')
parser.add_argument("--model_name", type=str, default='VRIN')
parser.add_argument("--dataset", type=str, default='physionet')
parser.add_argument("--data_path", type=str, default='../../vrin_journal/data/physionet/data_nan.p')
parser.add_argument("--label_path", type=str, default='../../vrin_journal/data/physionet/label.p')
parser.add_argument("--hours", type=int, default='48')
parser.add_argument("--fold", type=int, default=0)
parser.add_argument("--note", type=str, default='Bi-Directional')
parser.add_argument("--epoch", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.0000)
parser.add_argument("--batchsize", type=int, default=64)
parser.add_argument("--lambda1", type=float, default=0.00001)
parser.add_argument("--lambda2", type=float, default=0.00001)
parser.add_argument("--vae_weight", type=float, default=1.0)
parser.add_argument("--rin_weight", type=float, default=1.0)
parser.add_argument("--rin_consistency_weight", type=float, default=1.0)
parser.add_argument("--beta", type=float, default=1.0)
parser.add_argument("--rin_hiddens", type=int, default=64)
parser.add_argument("--unc_flag", type=int, default=1) # 0: Without Uncertainty, C: With Uncertainty
parser.add_argument("--removal_percent", type=int, default=10)
parser.add_argument("--keep_prob", type=float, default=0.0)
parser.add_argument("--task", type=str, default='C') # I: Imputation, C: Classification
args = parser.parse_args()

# GPU Configuration
gpu_id = args.gpu_id
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Loading the kfold dataset
kfold_data = pickle.load(open(args.data_path, 'rb'))
kfold_label = pickle.load(open(args.label_path, 'rb'))

# Hidden Units
if args.dataset == 'mimic':
    args.vae_hiddens = [99, 128, 32, 16]
else:
    args.vae_hiddens = [35, 64, 24, 10]

# Change Learning Rate and Keep Prob
if args.dataset == 'mimic':
    if args.task == 'I':
        args.keep_prob = 0.3
        args.lr = 0.005
    elif args.task == 'C':
        args.keep_prob = 0.3
        args.lr = 0.0003
else:
    if args.task == 'I':
        args.keep_prob = 0.3
        args.lr = 0.005
    elif args.task == 'C':
        args.keep_prob = 0.1
        args.lr = 0.0005

# For logging purpose, create several directories
date_str = str(datetime.datetime.now().strftime('%Y%m%d.%H.%M.%S'))
dir = 'log/%s/%s_%d/T%s_B%.4f_A%.4f/%d/%s/' % (args.dataset,
                                               args.model_name,
                                               args.unc_flag,
                                               args.task,
                                               args.rin_weight,
                                               args.vae_weight,
                                               args.fold,
                                               date_str)
if not os.path.exists(dir):
    os.makedirs(dir)
    os.makedirs(dir + '/%s' % ('img/train'))
    os.makedirs(dir + '/%s' % ('img/valid'))
    os.makedirs(dir + '/%s' % ('img/test'))
    os.makedirs(dir + '/tflog/')
    os.makedirs(dir + '/model/')

# Text Logging
f = open(dir + 'log.txt', 'a')
writelog(f, '---------------')
writelog(f, 'Model: %s' % args.model_name)
writelog(f, 'Uncertainty: %d' % args.unc_flag)
writelog(f, 'Keep Prob: %f' % args.keep_prob)
for h in args.vae_hiddens:
    writelog(f, 'VAE Hidden Units: %d' % h)
writelog(f, 'RIN Hidden Units: %d' % args.rin_hiddens)
writelog(f, 'Task: %s' % args.task)
writelog(f, '---------------')
writelog(f, 'Dataset: %s' % args.dataset)
writelog(f, 'Hours: %s' % args.hours)
writelog(f, 'Removal: %s' % args.removal_percent)
writelog(f, '---------------')
writelog(f, 'Fold: %d' % args.fold)
writelog(f, 'Learning Rate: %.5f' % args.lr)
writelog(f, 'Batch Size: %d' % args.batchsize)
writelog(f, 'Lambda1: %.5f' % args.lambda1)
writelog(f, 'Lambda2: %.5f' % args.lambda2)
writelog(f, 'Beta: %.2f' % args.beta)
writelog(f, 'VAE Imputation Weight: %.3f' % args.vae_weight)
writelog(f, 'RIN Imputation Weight: %.3f' % args.rin_weight)
writelog(f, 'Consitency Loss Imputation Weight: %.3f' % args.rin_consistency_weight)
writelog(f, '---------------')
writelog(f, 'Note: %s' % args.note)
writelog(f, '---------------')
writelog(f, 'TRAINING LOG')

def train(data, dir='./', task='I'):
    # Set mode as Training
    vae.train()
    rin_f.train()
    rin_b.train()

    # Define training variables
    loss = 0
    loss_nll = 0
    loss_mae = 0
    loss_kld = 0
    loss_l1 = 0
    n_batches = 0

    # Loop over the minibatch
    for i, xdata in enumerate(data):

        # Data
        x = xdata['values'].to(device)
        m = xdata['masks'].to(device)
        d_f = xdata['deltas_f'].to(device)
        d_b = xdata['deltas_b'].to(device)
        eval_x = xdata['evals'].to(device)
        eval_m = xdata['eval_masks'].to(device)
        y = xdata['labels'].to(device)

        [B, T, V] = x.shape

        # Zero Grad
        optimizer.zero_grad()

        # VAE
        rx = x.contiguous().view(-1, V)
        rm = m.contiguous().view(-1, V)
        z, enc_mu, enc_logvar, x_hat, dec_mu, dec_logvar = vae(rx)
        unc = (m * torch.zeros(B, T, V).to(device)) + ((1 - m) * torch.exp(0.5 * dec_logvar).view(B, T, V))

        # RIN Forward
        x_imp_f, y_out_f, y_score_f, xreg_loss_f, _, _ = rin_f(x, x_hat.view(B, T, V), unc, m, d_f, y)

        # Set data to be backward
        x_b = x.flip(dims=[1])
        x_hat_b = x_hat.view(B, T, V).flip(dims=[1])
        unc_b = unc.flip(dims=[1])
        m_b = m.flip(dims=[1])

        # RIN Backward
        x_imp_b, y_out_b, y_score_b, xreg_loss_b, _, _ = rin_b(x_b, x_hat_b, unc_b, m_b, d_b, y)

        # Sum the regression loss
        xreg_loss = (xreg_loss_f + xreg_loss_b)/2

        # Loss
        if task == 'C':
            loss_rin_f = criterion_rin(y_out_f, y.unsqueeze(1))
            loss_rin_b = criterion_rin(y_out_b, y.unsqueeze(1))
            # Sum the prediction loss
            loss_rin = (loss_rin_f + loss_rin_b)/2
        else:
            loss_rin = 0
        loss_vae, lossnll, lossmae, losskld, lossl1 = criterion_vae(vae, rx, eval_x.view(B*T, V), x_hat.view(B*T, V), rm, eval_m.view(B*T, V), enc_mu, enc_logvar, dec_mu, dec_logvar, phase='train')

        # Add consistency loss
        loss_consistency = torch.abs(x_imp_f - x_imp_b.flip(dims=[1])).mean()

        # Imputation Loss
        loss_imp = (args.rin_weight * xreg_loss) + (args.vae_weight * loss_vae) + (args.rin_consistency_weight * loss_consistency)

        # Overall loss
        loss_total = loss_imp + loss_rin

        loss += loss_total.item()
        loss_nll += lossnll
        loss_mae += lossmae
        loss_kld += losskld
        loss_l1 += lossl1
        n_batches += 1

        # Bacward Propagation and Update the weights
        loss_total.backward()

        # Update the weights
        optimizer.step()

        # Visualize Imputation Result
        # if i == 0:
        #     plot_imputation_mae(dir + '/img/train',
        #                         x[0].to('cpu').detach().numpy(),
        #                         m[0].to('cpu').detach().numpy(),
        #                         eval_x[0].to('cpu').detach().numpy() * eval_m[0].to('cpu').detach().numpy(),
        #                         x_imp[0].to('cpu').detach().numpy() * eval_m[0].to('cpu').detach().numpy(),
        #                         x_bar.view(B, T, V)[0].to('cpu').detach().numpy(),
        #                         x_imp[0].to('cpu').detach().numpy(),
        #                         x_hat.view(B, T, V)[0].to('cpu').detach().numpy(),
        #                         unc[0].to('cpu').detach().numpy(),
        #                         epoch)

    # Averaging the loss
    loss = loss / n_batches
    loss_nll = loss_nll / n_batches
    loss_mae = loss_mae / n_batches
    loss_kld = loss_kld / n_batches
    loss_l1 = loss_l1 / n_batches
    writelog(f, 'Loss : ' + str(loss))
    writelog(f, 'Loss NLL : ' + str(loss_nll))
    writelog(f, 'Loss MAE : ' + str(loss_mae))
    writelog(f, 'Loss KLD : ' + str(loss_kld))
    writelog(f, 'Loss L1 : ' + str(loss_l1))

    # Tensorboard Logging
    info = {'loss': loss,
            'loss_nll': loss_nll,
            'loss_mae': loss_mae,
            'loss_kld': loss_kld,
            'loss_l1': loss_l1}
    for tag, value in info.items():
        summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=tag, simple_value=value)])
        tfw_train.add_summary(summary, epoch)

def evaluate(phase, data, dir='./', task='I'):
    # Set mode as Evaluation
    vae.eval()
    rin_f.eval()
    rin_b.eval()

    # Define training variables
    loss = 0
    loss_nll = 0
    loss_mae = 0
    loss_kld = 0
    loss_l1 = 0
    n_batches = 0

    y_gts = np.array([]).reshape(0)
    y_preds = np.array([]).reshape(0)
    y_scores = np.array([]).reshape(0)
    eval_xs = []
    imp_xs = []

    # Loop over the minibatch
    with torch.no_grad():
        for i, xdata in enumerate(data):

            # Data
            x = xdata['values'].to(device)
            m = xdata['masks'].to(device)
            d_f = xdata['deltas_f'].to(device)
            d_b = xdata['deltas_b'].to(device)
            eval_x = xdata['evals'].to(device)
            eval_m = xdata['eval_masks'].to(device)
            y = xdata['labels'].to(device)

            y_gts = np.hstack([y_gts, y.to('cpu').detach().numpy().flatten()])
            [B, T, V] = x.shape

            # VAE
            rx = x.contiguous().view(-1, V)
            rm = m.contiguous().view(-1, V)
            z, enc_mu, enc_logvar, x_hat, dec_mu, dec_logvar = vae(rx)
            unc = (m * torch.zeros(B, T, V).to(device)) + ((1 - m) * torch.exp(0.5 * dec_logvar).view(B, T, V))

            # RIN Forward
            x_imp_f, y_out_f, y_score_f, xreg_loss_f, _, _ = rin_f(x, x_hat.view(B, T, V), unc, m, d_f, y)

            # Set data to be backward
            x_b = x.flip(dims=[1])
            x_hat_b = x_hat.view(B, T, V).flip(dims=[1])
            unc_b = unc.flip(dims=[1])
            m_b = m.flip(dims=[1])

            # RIN Backward
            x_imp_b, y_out_b, y_score_b, xreg_loss_b, _, _ = rin_b(x_b, x_hat_b, unc_b, m_b, d_b, y)

            # Averaging the imputations and prediction
            x_imp = (x_imp_f + x_imp_b.flip(dims=[1])) / 2
            y_score = (y_score_f + y_score_b) / 2

            # Sum the regression loss
            xreg_loss = (xreg_loss_f + xreg_loss_b) / 2

            # Loss
            if task == 'C':
                loss_rin_f = criterion_rin(y_out_f, y.unsqueeze(1))
                loss_rin_b = criterion_rin(y_out_b, y.unsqueeze(1))
                # Sum the prediction loss
                loss_rin = (loss_rin_f + loss_rin_b)/2
            else:
                loss_rin = 0
            loss_vae, lossnll, lossmae, losskld, lossl1 = criterion_vae(vae, rx, eval_x.view(B*T, V), x_hat.view(B*T, V), rm, eval_m.view(B*T, V), enc_mu, enc_logvar, dec_mu, dec_logvar, phase=phase)

            # Add consistency loss
            loss_consistency = torch.abs(x_imp_f - x_imp_b.flip(dims=[1])).mean()

            # Imputation Loss
            loss_imp = (args.rin_weight * xreg_loss) + (args.vae_weight * loss_vae) + (args.rin_consistency_weight * loss_consistency)

            # Overall loss
            loss_total = loss_imp + loss_rin

            loss += loss_total.item()
            loss_nll += lossnll
            loss_mae += lossmae
            loss_kld += losskld
            loss_l1 += lossl1
            n_batches += 1

            # Calculate Evaluation Metric
            eval_m = eval_m.to('cpu').detach().numpy()
            if task == 'I':
                ex = eval_x.data.cpu().numpy()
                impx = x_imp.data.cpu().numpy()
                eval_xs += ex[np.where(eval_m == 1)].tolist()
                imp_xs += impx[np.where(eval_m == 1)].tolist()
            else:
                y_pred = np.round(y_score.to('cpu').detach().numpy())
                y_score = y_score.to('cpu').detach().numpy()
                y_preds = np.hstack([y_preds, y_pred.reshape(-1)])
                y_scores = np.hstack([y_scores, y_score.reshape(-1)])

            # Visualize Imputation Result
            # if i==0:
            #     plot_imputation_mae(dir + '/img/' + phase,
            #                         x[0].to('cpu').detach().numpy(),
            #                         m[0].to('cpu').detach().numpy(),
            #                         eval_x[0].to('cpu').detach().numpy() * eval_m[0],
            #                         x_imp[0].to('cpu').detach().numpy() * eval_m[0],
            #                         x_bar.view(B, T, V)[0].to('cpu').detach().numpy(),
            #                         x_imp[0].to('cpu').detach().numpy(),
            #                         x_hat.view(B, T, V)[0].to('cpu').detach().numpy(),
            #                         unc[0].to('cpu').detach().numpy(),
            #                         epoch)

    # Averaging the loss
    loss = loss / n_batches
    loss_nll = loss_nll / n_batches
    loss_mae = loss_mae / n_batches
    loss_kld = loss_kld / n_batches
    loss_l1 = loss_l1 / n_batches
    writelog(f, 'Loss : ' + str(loss))
    writelog(f, 'Loss NLL : ' + str(loss_nll))
    writelog(f, 'Loss MAE : ' + str(loss_mae))
    writelog(f, 'Loss KLD : ' + str(loss_kld))
    writelog(f, 'Loss L1 : ' + str(loss_l1))

    if task == 'I':
        eval_xs = np.asarray(eval_xs)
        imp_xs = np.asarray(imp_xs)
        mae = np.abs(eval_xs - imp_xs).mean()
        mre = np.abs(eval_xs - imp_xs).sum() / np.abs(eval_xs).sum()

        # Averaging & displaying the Evaluation Metric
        writelog(f, 'MAE : ' + str(mae))
        writelog(f, 'MRE : ' + str(mre))

        # Tensorboard Logging
        info = {'loss': loss,
                'loss_nll': loss_nll,
                'loss_mae': loss_mae,
                'loss_kld': loss_kld,
                'loss_l1': loss_l1,
                'mae': mae,
                'mre': mre}
    else:
        auc, auprc, acc, balacc, sens, spec, prec, recall = calculate_performance(y_gts,
                                                                                 y_scores,
                                                                                 y_preds)

        writelog(f, 'AUC : ' + str(auc))
        writelog(f, 'AUC PRC : ' + str(auprc))
        writelog(f, 'Accuracy : ' + str(acc))
        writelog(f, 'BalACC : ' + str(balacc))
        writelog(f, 'Sensitivity : ' + str(sens))
        writelog(f, 'Specificity : ' + str(spec))
        writelog(f, 'Precision : ' + str(prec))
        writelog(f, 'Recall : ' + str(recall))

        # Tensorboard Logging
        info = {'loss': loss,
                'balacc': balacc,
                'auc': auc,
                'auc_prc': auprc,
                'sens': sens,
                'spec': spec,
                'precision': prec,
                'recall': recall}

    for tag, value in info.items():
        summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=tag, simple_value=value)])
        if phase == 'valid':
            tfw_valid.add_summary(summary, epoch)
        else:
            tfw_test.add_summary(summary, epoch)

    if task == 'I':
        return mae, mre
    else:
        return auc, auprc, acc, balacc, sens, spec, prec, recall
# Process Defined Fold
writelog(f, '---------------')
writelog(f, 'FOLD ' + str(args.fold))

# Tensorboard Logging
tfw_train = tf.compat.v1.summary.FileWriter(dir + 'tflog/kfold_' + str(args.fold) + '/train_')
tfw_valid = tf.compat.v1.summary.FileWriter(dir + 'tflog/kfold_' + str(args.fold) + '/valid_')
tfw_test = tf.compat.v1.summary.FileWriter(dir + 'tflog/kfold_' + str(args.fold) + '/test_')

# Get dataset
train_data = kfold_data[args.fold][0]
train_label = kfold_label[args.fold][0]

valid_data = kfold_data[args.fold][1]
valid_label = kfold_label[args.fold][1]

test_data = kfold_data[args.fold][2]
test_label = kfold_label[args.fold][2]

# Normalization
writelog(f, 'Normalization')
train_data, mean_set, std_set = normalize(train_data, [], [])
valid_data, m, s = normalize(valid_data, mean_set, std_set)
test_data, m, s = normalize(test_data, mean_set, std_set)

# Tensor Seed
random.seed(1)
torch.manual_seed(1)

# Define Loaders
train_loader = sample_loader_bidirectional('train', train_data, train_label, args.batchsize, args.removal_percent, args.task)
valid_loader = sample_loader_bidirectional('valid', valid_data, valid_label, args.batchsize, args.removal_percent, args.task)
test_loader = sample_loader_bidirectional('test', test_data, test_label, args.batchsize, args.removal_percent, args.task)
dataloaders = {'train': train_loader,
               'valid': valid_loader,
               'test': test_loader}

# Remove Data
kfold_data = None
kfold_label = None
train_data = None
train_label = None
valid_data = None
valid_label = None
test_data = None
test_label = None

# Define Model
criterion_vae = SVAELoss(args).to(device)
criterion_rin = nn.BCEWithLogitsLoss().to(device)
vae = VAE(args).to(device)
rin_f = RIN(args).to(device)
rin_b = RIN(args).to(device)

total_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
total_params += sum(p.numel() for p in rin_f.parameters() if p.requires_grad)
total_params += sum(p.numel() for p in rin_b.parameters() if p.requires_grad)
writelog(f, 'Total params is {}'.format(total_params))

# Define Optimizer
# optimizer = optim.Adam(list(vae.parameters()) + list(rin.parameters()), lr=args.lr)
optimizer = optim.Adam(list(vae.parameters()) +
                       list(rin_f.parameters()) +
                       list(rin_b.parameters()), lr=args.lr, weight_decay=args.lambda2)

# Reset Best AUC
if args.task == 'I':
    valid = {
        'epoch': 0,
        'mae': 9999
    }
    test = {
        'epoch': 0,
        'mae': 0,
        'mre': 0,
    }
else:
    valid = {
        'epoch': 0,
        'auc': 0,
    }
    test = {
        'epoch': 0,
        'auc': 0, 'auprc': 0,
        'acc': 0, 'balacc': 0,
        'sens': 0, 'spec': 0,
        'prec': 0, 'recall': 0
    }
# Training & Validation Loop
for epoch in range(args.epoch):

    writelog(f, '------ Epoch ' + str(epoch))

    writelog(f, '-- Training')
    train(dataloaders['train'], dir=dir, task=args.task)

    writelog(f, '-- Validation')
    if args.task == 'I':
        mae, mre = evaluate('valid', dataloaders['valid'], dir=dir, task=args.task)

        if mae < valid['mae']:
            torch.save(vae, '%s/model/vae_%d_%d.pt' % (dir, args.fold, epoch))
            torch.save(rin_f, '%s/model/rnn_f_%d_%d.pt' % (dir, args.fold, epoch))
            torch.save(rin_b, '%s/model/rnn_b_%d_%d.pt' % (dir, args.fold, epoch))
            writelog(f, 'Best validation MAE is found! Validation MAE : %f' % mae)
            writelog(f, 'Models at Epoch %d are saved!' % epoch)
            valid['mae'] = mae
            valid['epoch'] = epoch

        writelog(f, '-- Test')
        mae, mre = evaluate('test', dataloaders['test'], dir=dir, task=args.task)

        # Save performance if current epoch is the best epoch based on validation's AUC
        if valid['epoch'] == epoch:
            test['epoch'] = epoch
            test['mae'] = mae
            test['mre'] = mre

    else:
        auc, auprc, acc, balacc, sens, spec, prec, recall = evaluate('valid', dataloaders['valid'], dir=dir, task=args.task)

        if auc > valid['auc']:
            torch.save(vae, '%s/model/vae_%d_%d.pt' % (dir, args.fold, epoch))
            torch.save(rin_f, '%s/model/rnn_f_%d_%d.pt' % (dir, args.fold, epoch))
            torch.save(rin_b, '%s/model/rnn_b_%d_%d.pt' % (dir, args.fold, epoch))
            writelog(f, 'Best validation AUC is found! Validation AUC : %f' % auc)
            writelog(f, 'Models at Epoch %d are saved!' % epoch)
            valid['auc'] = auc
            valid['epoch'] = epoch

        writelog(f, '-- Test')
        auc, auprc, acc, balacc, sens, spec, prec, recall = evaluate('test', dataloaders['test'], dir=dir, task=args.task)

        # Save performance if current epoch is the best epoch based on validation's AUC
        if valid['epoch'] == epoch:
            test['epoch'] = epoch
            test['auc'] = auc
            test['auprc'] = auprc
            test['acc'] = acc
            test['balacc'] = balacc
            test['sens'] = sens
            test['spec'] = spec
            test['prec'] = prec
            test['recall'] = recall

writelog(f, '-- Best Test')
for b in test:
    writelog(f, '%s:%f' % (b, test[b]))
writelog(f, 'END OF FOLD')
f.close()
