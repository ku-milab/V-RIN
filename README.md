# Uncertainty-Aware Variational-Recurrent Imputation Network for Clinical Time Series
A. W. Mulyadi, E. Jun, and H.-I. Suk, “Uncertainty-Aware Variational-Recurrent Imputation Network for Clinical Time Series,” arXiv preprint arXiv:2003.00662, 2020. https://arxiv.org/abs/2003.00662

_Accepted to IEEE Transactions on Cybernetics, 2021_
## Usage

For the V-RIN-full using unidirectional scenario:

python vrin.py --unc_flag=1 --dataset='physionet' --hours=48 --removal_percent=10 --gpu_id=0 --task='C' --vae_weight=1.0 --rin_weight=0.75 --fold=0

For the V-RIN-full using bidirectional scenario:

python vrin.py --unc_flag=1 --dataset='physionet' --hours=48 --removal_percent=10 --gpu_id=0 --task='C' --vae_weight=1.0 --rin_weight=0.75 --fold=0
