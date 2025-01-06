import pickle
import glob
from torch.utils.data.dataloader import DataLoader
import torch.distributions.multivariate_normal as torchdist
from pyDOE import lhs
from utils import *
from metrics import * 
from model import TrajectoryModel
import copy
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def box_muller_transform(x: torch.FloatTensor):
    r"""Box-Muller transform"""
    shape = x.shape
    x = x.view(shape[:-1] + (-1, 2))
    z = torch.zeros_like(x, device=x.device)
    z[..., 0] = (-2 * x[..., 0].log()).sqrt() * (2 * np.pi * x[..., 1]).cos()
    z[..., 1] = (-2 * x[..., 0].log()).sqrt() * (2 * np.pi * x[..., 1]).sin()
    return z.view(shape)


def compute_batch_metric(pred, gt):
    """Get ADE, FDE, TCC scores for each pedestrian"""
    # Calculate ADEs and FDEs
    temp = (pred - gt).norm(p=2, dim=-1)
    ADEs = temp.mean(dim=1).min(dim=0)[0]
    FDEs = temp[:, -1, :].min(dim=0)[0]

    # Calculate TCCs
    pred_best = pred[temp[:, -1, :].argmin(dim=0), :, range(pred.size(2)), :]
    pred_gt_stack = torch.stack([pred_best, gt.permute(1, 0, 2)], dim=0)
    pred_gt_stack = pred_gt_stack.permute(3, 1, 0, 2)
    covariance = pred_gt_stack - pred_gt_stack.mean(dim=-1, keepdim=True)
    factor = 1 / (covariance.shape[-1] - 1)
    covariance = factor * covariance @ covariance.transpose(-1, -2)
    variance = covariance.diagonal(offset=0, dim1=-2, dim2=-1)
    stddev = variance.sqrt()
    corrcoef = covariance / stddev.unsqueeze(-1) / stddev.unsqueeze(-2)
    corrcoef = corrcoef.clamp(-1, 1)
    corrcoef[torch.isnan(corrcoef)] = 0
    TCCs = corrcoef[:, :, 0, 1].mean(dim=0)
    return ADEs, FDEs, TCCs



def test(model, loader_test, KSTEPS=20):

    model.eval()
    ade_all, fde_all, tcc_all = [], [], []

    step =0
    pic_cnt = 0
    for batch in loader_test:
        step+=1
        #Get data
        batch = [tensor.cuda() for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
        loss_mask, V_obs, V_tr = batch

        identity_spatial = torch.ones((V_obs.shape[1], V_obs.shape[2], V_obs.shape[2])) * torch.eye(
            V_obs.shape[2])
        identity_temporal = torch.ones((V_obs.shape[2], V_obs.shape[1], V_obs.shape[1])) * torch.eye(
            V_obs.shape[1])
        identity_spatial = identity_spatial.cuda()
        identity_temporal = identity_temporal.cuda()
        identity = [identity_spatial, identity_temporal]

        V_pred = model(V_obs, identity)  # A_obs <8, #, #>

        V_pred = V_pred.squeeze()
        V_tr = V_tr.squeeze()
        num_of_objs = obs_traj_rel.shape[1]
        V_pred, V_tr = V_pred[:, :num_of_objs, :], V_tr[:, :num_of_objs, :]
        #
        # #For now I have my bi-variate parameters
        # #normx =  V_pred[:,:,0:1]
        # #normy =  V_pred[:,:,1:2]
        sx = torch.exp(V_pred[:,:,2]) #sx
        sy = torch.exp(V_pred[:,:,3]) #sy
        corr = torch.tanh(V_pred[:,:,4]) #corr
        #
        cov = torch.zeros(V_pred.shape[0],V_pred.shape[1],2,2).cuda()
        cov[:,:,0,0]= sx*sx
        cov[:,:,0,1]= corr*sx*sy
        cov[:,:,1,0]= corr*sx*sy
        cov[:,:,1,1]= sy*sy
        mean = V_pred[:,:,0:2]

        V_obs_traj = obs_traj.permute(0, 3, 1, 2).squeeze(dim=0)
        V_pred_traj_gt = pred_traj_gt.permute(0, 3, 1, 2).squeeze(dim=0)

        ade_stack, fde_stack, tcc_stack = [], [], []

        #### Latin hypercube sampling ####
        lhs_sample = torch.tensor(lhs(2, samples=20))
        qr_seq = torch.stack([box_muller_transform(lhs_sample) for _ in range(mean.size(0))], dim=1).unsqueeze(
            dim=2).type_as(mean)
        sample = mean + (torch.linalg.cholesky(cov) @ qr_seq.unsqueeze(dim=-1)).squeeze(dim=-1)

        # Evaluate trajectories
        V_absl = sample.cumsum(dim=1) + V_obs_traj[[-1], :, :]
        ADEs, FDEs, TCCs = compute_batch_metric(V_absl, V_pred_traj_gt)

        ade_stack.append(ADEs.detach().cpu().numpy())
        fde_stack.append(FDEs.detach().cpu().numpy())
        tcc_stack.append(TCCs.detach().cpu().numpy())

        ade_all.append(np.array(ade_stack))
        fde_all.append(np.array(fde_stack))
        tcc_all.append(np.array(tcc_stack))


    ade_all = np.concatenate(ade_all, axis=1)
    fde_all = np.concatenate(fde_all, axis=1)
    tcc_all = np.concatenate(tcc_all, axis=1)

    mean_ade, mean_fde, mean_tcc = ade_all.mean(axis=0).mean(), fde_all.mean(axis=0).mean(), tcc_all.mean(axis=0).mean()
    return mean_ade, mean_fde, mean_tcc






def main():

    KSTEPS = 20
    ade_ls = []
    fde_ls = []
    print('Number of samples:', KSTEPS)
    print("*" * 50)
    root_ = './checkpoints/'
    dataset = ['STIGCN/eth',
               'STIGCN/hotel',
               'STIGCN/univ',
               'STIGCN/zara1',
               'STIGCN/zara2']

    paths = list(map(lambda x: root_ + x, dataset))

    for feta in range(len(paths)):

        path = paths[feta]
        exps = glob.glob(path)
        print('Model being tested are:', exps)
        for exp_path in exps:
            print("*" * 50)
            print("Evaluating model:", exp_path)

            model_path = exp_path + '/val_best.pth'
            args_path = exp_path + '/args.pkl'
            with open(args_path, 'rb') as f:
                args = pickle.load(f)

            # Data prep
            obs_seq_len = args.obs_len
            pred_seq_len = args.pred_len
            data_set = './dataset/' + args.dataset + '/'

            dset_test = TrajectoryDataset(
                data_set + 'test/',
                obs_len=obs_seq_len,
                pred_len=pred_seq_len,
                skip=1)

            loader_test = DataLoader(
                dset_test,
                batch_size=1,  # This is irrelative to the args batch size parameter
                shuffle=False,
                num_workers=1)

            model = TrajectoryModel(embedding_dims=64, number_gcn_layers=1, dropout=0,
                                    obs_len=8, pred_len=12, n_tcn=5, out_dims=5).cuda()
            model.load_state_dict(torch.load(model_path))

            num_params = sum(p.numel() for p in model.parameters())
            print(f"Number of parameters: {num_params}")

            ad_ = 999999
            fd_ = 999999
            print("Testing ....")
            ade_,fde_,raw_data_dict = test(model, loader_test)
            ade_ = min(ade_, ad_)
            fde_ = min(fde_, fd_)
            ade_ls.append(ade_)
            fde_ls.append(fde_)
            print("ade:", ade_, " fde:", fde_)

        num_params = sum(p.numel() for p in model.parameters())
        print(f"Number of parameters: {num_params}")

        print("*" * 50)

    print("Avg ADE:", sum(ade_ls) / 5)
    print("Avg FDE:", sum(fde_ls) / 5)


if __name__ == '__main__':
    main()