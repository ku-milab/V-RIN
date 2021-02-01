import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import math

class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()

        self.args = args
        self.hiddens = self.args.vae_hiddens

        # Encoder
        self.enc = nn.Sequential()
        for i in range(len(self.hiddens)-2):
            self.enc.add_module("fc_%d" % i, nn.Linear(self.hiddens[i], self.hiddens[i+1]))
            self.enc.add_module("bn_%d" % i, nn.BatchNorm1d(self.hiddens[i+1]))
            self.enc.add_module("do_%d" % i, nn.Dropout(self.args.keep_prob))
            self.enc.add_module("tanh_%d" % i, nn.Tanh())
        self.enc_mu = nn.Linear(self.hiddens[-2], self.hiddens[-1])
        self.enc_logvar = nn.Linear(self.hiddens[-2], self.hiddens[-1])

        # Decoder
        self.dec = nn.Sequential()
        for i in range(len(self.hiddens))[::-1][:-2]:
            self.dec.add_module("fc_%d" % i, nn.Linear(self.hiddens[i], self.hiddens[i-1]))
            self.dec.add_module("bn_%d" % i, nn.BatchNorm1d(self.hiddens[i-1]))
            self.dec.add_module("do_%d" % i, nn.Dropout(self.args.keep_prob))
            self.dec.add_module("tanh_%d" % i, nn.Tanh())
        self.dec_mu = nn.Linear(self.hiddens[1], self.hiddens[0])
        self.dec_logvar = nn.Linear(self.hiddens[1], self.hiddens[0])

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            if len(weight.size()) == 1:
                continue
            stv = 1. / math.sqrt(weight.size(1))
            nn.init.uniform_(weight, -stv, stv)

    # Reparameterize
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):

        # Encoding
        e = self.enc(x)
        enc_mu = self.enc_mu(e)
        enc_logvar =self.enc_logvar(e)
        z = self.reparameterize(enc_mu, enc_logvar)

        # Decoding
        d = self.dec(z)
        dec_mu = self.dec_mu(d)
        dec_logvar = self.dec_logvar(d)
        x_hat = dec_mu

        return z, enc_mu, enc_logvar, x_hat, dec_mu, dec_logvar

class FeatureRegression(nn.Module):
    def __init__(self, input_size):
        super(FeatureRegression, self).__init__()
        self.build(input_size)

    def build(self, input_size):
        self.W = Parameter(torch.Tensor(input_size, input_size))
        self.b = Parameter(torch.Tensor(input_size))

        m = torch.ones(input_size, input_size) - torch.eye(input_size, input_size)
        self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        z_h = F.linear(x, self.W * Variable(self.m), self.b)
        return z_h

class Decay(nn.Module):
    def __init__(self, input_size, output_size, diag=False):
        super(Decay, self).__init__()
        self.diag = diag
        self.build(input_size, output_size)

    def build(self, input_size, output_size):
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))

        if self.diag == True:
            assert(input_size == output_size)
            m = torch.eye(input_size, input_size)
            self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
        if self.diag == True:
            gamma = F.relu(F.linear(d, self.W * Variable(self.m), self.b))
        else:
            gamma = F.relu(F.linear(d, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma

class RIN(nn.Module):
    def __init__(self, args):#
        super(RIN, self).__init__()

        self.args = args

        # Define Input Size Depends on the Dataset
        if self.args.dataset == 'physionet':
            input_size = 35
        elif self.args.dataset == 'mimic':
            input_size = 99
        else:
            input_size = 35
        self.input_size = input_size
        self.hidden_size = self.args.rin_hiddens

        self.hist = nn.Linear(self.hidden_size, input_size)
        self.conv1 = nn.Conv1d(2, 1, kernel_size=1, stride=1)
        self.temp_decay_h = Decay(input_size=input_size, output_size=self.hidden_size)
        self.feat_reg_v = FeatureRegression(input_size)
        self.feat_reg_r = FeatureRegression(input_size)

        self.unc_flag = self.args.unc_flag
        self.gru = nn.GRUCell(input_size * 2, self.hidden_size)
        self.fc_out = nn.Linear(self.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

        # Activate only for the model with uncertainty
        if self.args.unc_flag == 1:
            self.unc_decay = Decay(input_size=input_size, output_size=input_size)

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            if len(weight.size()) == 1:
                continue
            stv = 1. / math.sqrt(weight.size(1))
            nn.init.uniform_(weight, -stv, stv)

    def forward(self, x, x_hat, u, m, d, y):
        # Get dimensionality
        [B, T, V] = x.shape

        # Initialize Hidden weights
        h = Variable(torch.zeros(B, self.hidden_size)).cuda()

        x_loss = 0
        # x_imp = torch.Tensor().cuda()
        x_imp = []
        xus = []
        xrs = []
        for t in range(T):
            x_t = x[:, t, :]
            x_hat_t = x_hat[:, t, :]
            u_t = u[:, t, :]
            d_t = d[:, t, :]
            m_t = m[:, t, :]

            # Decayed Hidden States
            gamma_h = self.temp_decay_h(d_t)
            h = h * gamma_h

            # Regression
            x_h = self.hist(h)
            x_r_t = (m_t * x_t) + ((1 - m_t) * x_h)

            if self.args.unc_flag == 1:
                xbar = (m_t * x_t) + ((1 - m_t) * x_hat_t)
                xu = self.feat_reg_v(xbar) * self.unc_decay(u_t)
            else:
                xbar = (m_t * x_t) + ((1 - m_t) * x_hat_t)
                xu = self.feat_reg_v(xbar)

            xr = self.feat_reg_r(x_r_t)

            x_comb_t = self.conv1(torch.cat([xu.unsqueeze(1), xr.unsqueeze(1)], dim=1)).squeeze(1)
            x_loss += torch.sum(torch.abs(x_t - x_comb_t) * m_t) / (torch.sum(m_t) + 1e-5)

            # Final Imputation Estimates
            x_imp_t = (m_t * x_t) + ((1 - m_t) * x_comb_t)

            # Set input the the RNN
            input_t = torch.cat([x_imp_t, m_t], dim=1)

            # Feed into GRU cell, get the hiddens
            h = self.gru(input_t, h)

            # Keep the imputation
            x_imp.append(x_imp_t.unsqueeze(dim=1))
            xus.append(xu.unsqueeze(dim=1))
            xrs.append(xr.unsqueeze(dim=1))

        x_imp = torch.cat(x_imp, dim=1)
        xus = torch.cat(xus, dim=1)
        xrs = torch.cat(xrs, dim=1)

        # Get the output
        if self.args.task == 'C':
            y_out = self.fc_out(h)
            y_score = self.sigmoid(y_out)
        else:
            y_out = 0
            y_score = 0

        return x_imp, y_out, y_score, x_loss, xus, xrs

# Define RNN Model
class RNN(nn.Module):
    def __init__(self, args):
        super(RNN, self).__init__()

        self.args = args

        # Define Input Size Depends on the Dataset
        if self.args.dataset == 'physionet':
            input_size = 35
        elif self.args.dataset == 'mimic':
            input_size = 99
        else:
            input_size = 35
        self.input_size = input_size
        self.hidden_size = self.args.rnn_hiddens

        self.rnn = nn.GRU(self.input_size,
                          self.hidden_size,
                          1,
                          batch_first=True,
                          bidirectional=False)
        self.fc_output = nn.Linear(self.hidden_size, 1)

    def forward(self, x):

        # RNN Forwarding and get the last output only as the prediction result
        h = self.rnn(x)
        y_out = self.fc_output(h[0].contiguous()[:,-1,:])
        y_score = torch.sigmoid(y_out)

        return y_out, y_score

