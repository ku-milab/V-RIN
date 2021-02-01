import torch
from torch.distributions.normal import Normal
import torch.nn.functional as F
import torch.nn as nn

# SVAELOSS
class SVAELoss(torch.nn.Module):

    def __init__(self, args):
        super(SVAELoss, self).__init__()
        self.args = args
        self.lambda1 = torch.tensor(args.lambda1).cuda()
        self.mae = nn.L1Loss()

    def forward(self, model, x, eval_x, x_bar, m, eval_m, enc_mu, enc_logvar, dec_mu, dec_logvar, phase='train'):

        # Reconstruction Loss
        nll = -Normal(dec_mu, torch.exp(0.5 * dec_logvar)).log_prob(x).sum(1)
        mae = torch.tensor([0.0]).cuda()
        recon_loss = nll

        # Variational Encoder Loss
        KLD_enc = - self.args.beta * 0.5 * torch.sum(1 + enc_logvar - enc_mu.pow(2) - enc_logvar.exp(), 1)

        # Regularization
        l1_regularization = torch.tensor(0).float().cuda()
        for name, param in model.named_parameters():
            if 'bias' not in name:
                l1_regularization += self.lambda1 * torch.norm(param.cuda(), 1)

        # Take the average
        loss = torch.mean(recon_loss) + torch.mean(KLD_enc) + l1_regularization

        return loss, torch.mean(nll).item(), torch.mean(mae).item(), torch.mean(KLD_enc).item(), l1_regularization.item()


