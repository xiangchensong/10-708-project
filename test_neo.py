# Copyright (c) 2020, Ioana Bica

import argparse
import os
import numpy as np
import torch.nn.functional as F
import json
from scipy.optimize import minimize
from data_simulation import get_dataset_splits, TCGA_Data, get_iter
import torch
from torch.autograd import Variable
from TransTEE import TransTEE
from utils.utils import get_optimizer_scheduler
from utils.DisCri import DisCri
from utils.tsne import plot_tnse_ihdp
import random
from collections import defaultdict
from scipy.integrate import romb
def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def get_model_predictions(test_data,model):
    x = Variable(torch.from_numpy(test_data['x']).cuda().detach()).float()
    t = Variable(torch.from_numpy(test_data['t']).cuda().detach()).float()
    d = Variable(torch.from_numpy(test_data['d']).cuda().detach()).float()
    I_logits = model(x, t, d)
    return I_logits[1].cpu().detach().numpy()

w = None
wx = None
def get_y(x,t,d):
    te = w[t]*d
    xe = np.dot(wx[t],x)
    return te + xe

def get_patient_outcome(x,t,d):
    """
    x: patient data
    t: treatment
    d: dosage
    """
    return get_y(x,t,d)

def get_true_dose_response_curve(patient, treatment_idx):
    def true_dose_response_curve(dosage):
        y = get_patient_outcome(patient, treatment_idx, dosage)
        return y

    return true_dose_response_curve

def compute_eval_metrics(test_patients, num_treatments, model, train=False):
    mises = []
    ites = []
    dosage_policy_errors = []
    policy_errors = []
    pred_best = []
    pred_vals = []
    true_best = []

    samples_power_of_two = 6
    num_integration_samples = 2 ** samples_power_of_two + 1
    step_size = 1. / num_integration_samples
    treatment_strengths = np.linspace(np.finfo(float).eps, 1, num_integration_samples)

    for patient in test_patients:
        if train and len(pred_best) > 10:
            return np.sqrt(np.mean(mises)), np.sqrt(np.mean(dosage_policy_errors)), np.sqrt(np.mean(policy_errors)), np.mean(ites)
        for treatment_idx in range(num_treatments):
            test_data = dict()
            test_data['x'] = np.repeat(np.expand_dims(patient, axis=0), num_integration_samples, axis=0)
            test_data['t'] = np.repeat(treatment_idx, num_integration_samples)
            test_data['d'] = treatment_strengths
 
            pred_dose_response = get_model_predictions(test_data=test_data, model=model)[1].squeeze()
            # pred_dose_response = pred_dose_response * (
            #         dataset['metadata']['y_max'] - dataset['metadata']['y_min']) + \
            #                         dataset['metadata']['y_min']

            true_outcomes = np.array([get_patient_outcome(patient, treatment_idx, d) for d in
                                treatment_strengths])
            
            # if len(pred_best) < num_treatments and train == False:
            #     #print(true_outcomes)
            #     print([item[0] for item in pred_dose_response])
            mise = romb(np.square(true_outcomes - pred_dose_response).squeeze(), dx=step_size)
            inter_r = np.array(true_outcomes) - pred_dose_response.squeeze()
            ite = np.mean(inter_r ** 2)
            mises.append(mise)
            ites.append(ite)

            best_encountered_x = treatment_strengths[np.argmax(pred_dose_response)]

            def pred_dose_response_curve(dosage):
                test_data = dict()
                test_data['x'] = np.expand_dims(patient, axis=0)
                test_data['t'] = np.expand_dims(treatment_idx, axis=0)
                test_data['d'] = np.expand_dims(dosage, axis=0)

                ret_val = get_model_predictions(test_data=test_data, model=model)
                # ret_val = ret_val * (dataset['metadata']['y_max'] - dataset['metadata']['y_min']) + \
                #             dataset['metadata']['y_min']
                return ret_val.squeeze()

            true_dose_response_curve = get_true_dose_response_curve(patient, treatment_idx)

            min_pred_opt = minimize(lambda x: -1 * pred_dose_response_curve(x),
                                    x0=[best_encountered_x], method="SLSQP", bounds=[(0, 1)])

            max_pred_opt_y = - min_pred_opt.fun
            max_pred_dosage = min_pred_opt.x
            max_pred_y = true_dose_response_curve(max_pred_dosage)

            min_true_opt = minimize(lambda x: -1 * true_dose_response_curve(x),
                                    x0=[0.5], method="SLSQP", bounds=[(0, 1)])
            max_true_y = - min_true_opt.fun
            max_true_dosage = min_true_opt.x

            dosage_policy_error = (max_true_y - max_pred_y) ** 2
            dosage_policy_errors.append(dosage_policy_error)

            pred_best.append(max_pred_opt_y)
            pred_vals.append(max_pred_y)
            true_best.append(max_true_y)
            

        selected_t_pred = np.argmax(pred_vals[-num_treatments:])
        selected_val = pred_best[-num_treatments:][selected_t_pred]
        selected_t_optimal = np.argmax(true_best[-num_treatments:])
        optimal_val = true_best[-num_treatments:][selected_t_optimal]
        policy_error = (optimal_val - selected_val) ** 2
        policy_errors.append(policy_error)

    return np.sqrt(np.mean(mises)), np.sqrt(np.mean(dosage_policy_errors)), np.sqrt(np.mean(policy_errors)), np.mean(ites)


def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="GSM", type=str)
    parser.add_argument("--num_treatments", default=10, type=int)
    parser.add_argument("--treatment_selection_bias", default=2.0, type=float)
    parser.add_argument("--dosage_selection_bias", default=2.0, type=float)
    parser.add_argument("--save_dataset", default=True)
    # parser.add_argument("--validation_fraction", default=0.1, type=float)
    # parser.add_argument("--test_fraction", default=0.2, type=float)
    parser.add_argument("--model_name", default="TransTEE_tr")
    parser.add_argument("--verbose", default=False)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--h_dim", default=48, type=int)
    parser.add_argument("--rep", default=1, type=int)
    parser.add_argument("--h_inv_eqv_dim", default=64, type=int)
    parser.add_argument("--num_dosage_samples", default=5, type=int)
    parser.add_argument("--alpha", default=1.0, type=float)

    #optimizer and scheduler
    parser.add_argument('--beta', type=float, default=0.5, help='tradeoff parameter for advesarial loss')
    parser.add_argument('--p', type=int, default=0, help='dim for outputs of treatments discriminator, 1 for value, 2 for mean and var')
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--optimizer", type=str, default="adam", choices=["adam", "amsgrad"]
    )
    parser.add_argument(
            "--log_interval",
            type=int,
            default=10,
            help="How many batches to wait before logging training status",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        choices=["exponential", "cosine", "cycle", "none"],
    )
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--cov_dim", type=int, default=100)
    parser.add_argument(
        "--initialiser",
        type=str,
        default="xavier",
        choices=["xavier", "orthogonal", "kaiming", "none"],
    )
    return parser.parse_args()

def neg_guassian_likelihood(d, u):
    """return: -N(u; mu, var)"""
    B, dim = u.shape[0], 1
    assert (d.shape[1] == dim * 2)
    mu, logvar = d[:, :dim], d[:, dim:]
    return 0.5 * (((u - mu) ** 2) / torch.exp(logvar) + logvar).mean()
import pickle
def load_pkl_datasets(dataset_name):
    with open(f"datasets/{dataset_name}.pkl", 'rb') as f:
        dataset = pickle.load(f)
    return dataset
if __name__ == "__main__":

    args = init_arg()

    # dataset_params = dict()
    # dataset_params['num_treatments'] = args.num_treatments
    # dataset_params['treatment_selection_bias'] = args.treatment_selection_bias
    # dataset_params['dosage_selection_bias'] = args.dosage_selection_bias
    # dataset_params['save_dataset'] = args.save_dataset
    # dataset_params['validation_fraction'] = args.validation_fraction
    # dataset_params['test_fraction'] = args.test_fraction

    export_dir = 'saved_models/' + args.model_name +f'/{args.dataset}' +'/tr' +str(args.num_treatments) + '/trb' + str(args.treatment_selection_bias) +'/dob' +str(args.dosage_selection_bias)
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
    dataset_train, dataset_val, dataset_test,w_dict = load_pkl_datasets(args.dataset)
    dataset_train['x'] = torch.from_numpy(dataset_train['x']).float()
    dataset_train['t'] = torch.from_numpy(dataset_train['t']).float()
    dataset_train['d'] = torch.from_numpy(dataset_train['d']).float()
    dataset_train['y'] = torch.from_numpy(dataset_train['y']).float()#y_normalized
    w = np.array(w_dict['w'])
    wx = np.array(w_dict['wx'])
    args.num_treatments = wx.shape[0]
    result = {}
    result['in'] = []
    result['out'] = []
    result['val'] = {'mise':[],'dpe':[],'pe':[],'ate':[]}
    result['test'] = {'mise':[],'dpe':[],'pe':[],'ate':[]}
    result['train'] = {}
    for r in range(args.rep):
        result['train'][r] = defaultdict(list)
        params = {'num_features': dataset_train['x'].shape[1], 'num_treatments': args.num_treatments,
            'num_dosage_samples': args.num_dosage_samples, 'export_dir': export_dir,
            'alpha': args.alpha, 'batch_size': args.batch_size, 'h_dim': args.h_dim,
            'h_inv_eqv_dim': args.h_inv_eqv_dim, 'cov_dim':args.cov_dim, 'initialiser':args.initialiser}

        if 'tr' in args.model_name:
            TargetReg = DisCri(args.h_dim,dim_hidden=50, dim_output=args.p)#args.num_treatments
            TargetReg.cuda()
            tr_optimizer = torch.optim.SGD(TargetReg.parameters(), lr=0.001, weight_decay=5e-3)

        model = TransTEE(params)
        print("model have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
        model.cuda()
        optimizer, scheduler = get_optimizer_scheduler(args=args, model=model)
        

        train_loader = get_iter(dataset_train, batch_size=args.batch_size, shuffle=True)
        for it in range(args.max_epochs):
            for idx, (x, t, d, y) in enumerate(train_loader):
                X_mb = Variable(x).cuda().detach().float()
                T_mb = Variable(t).cuda().detach().float()
                D_mb = Variable(d).cuda().detach().float()
                Y_mb = Variable(y).cuda().detach().float().squeeze()

                optimizer.zero_grad()
                pred_outcome = model(X_mb, T_mb, D_mb)
                if 'tr' in args.model_name:
                    set_requires_grad(TargetReg, True)
                    tr_optimizer.zero_grad()
                    trg = TargetReg(pred_outcome[0].detach())
                    if args.p == 1:
                        loss_D = F.mse_loss(trg.squeeze(), T_mb)
                    elif args.p == 2:
                        loss_D = neg_guassian_likelihood(trg.squeeze(), T_mb)
                    loss_D.backward()
                    tr_optimizer.step()

                    set_requires_grad(TargetReg, False)
                    trg = TargetReg(pred_outcome[0])
                    if args.p == 1:
                        loss_D = F.mse_loss(trg.squeeze(), T_mb)
                    elif args.p == 2:
                        loss_D = neg_guassian_likelihood(trg.squeeze(), T_mb)
                    loss = F.mse_loss(input=pred_outcome[1].squeeze(), target=Y_mb) - args.beta * loss_D
                    loss.backward()
                    optimizer.step()
                else:
                    loss = F.mse_loss(input=pred_outcome[1].squeeze(), target=Y_mb)
                    loss.backward()
                    optimizer.step()
            if scheduler is not None:
                scheduler.step()
            if it % args.log_interval == 0:
                tr_mise, tr_dpe, tr_pe, tr_ite = compute_eval_metrics(X_mb.cpu().detach().numpy(), num_treatments=params['num_treatments'],model=model, train=True)
                mise, dpe, pe, ite = compute_eval_metrics(dataset_test['x'], num_treatments=params['num_treatments'],model=model, train=True)
                result['train'][r]['loss'].append(loss.item())
                result['train'][r]['tr_mise'].append(tr_mise)
                result['train'][r]['tr_dpe'].append(tr_dpe)
                result['train'][r]['tr_pe'].append(tr_pe)
                result['train'][r]['tr_ite'].append(tr_ite)
                result['train'][r]['mise'].append(mise)
                result['train'][r]['dpe'].append(dpe)
                result['train'][r]['pe'].append(pe)
                result['train'][r]['ite'].append(ite)
                print( "Train Epoch: [{}/{}]\tLoss: {:.6f}, tr_mise: {:.6f}, tr_dpe: {:.6f}, tr_pe: {:.6f}, tr_ite: {:.6}, mise: {:.6f}, dpe: {:.6f}, pe: {:.6f}, ite: {:.6}".format(it, args.max_epochs, loss.item(),  tr_mise, tr_dpe, tr_pe, tr_ite, mise, dpe, pe, ite )) 
        
        print('-----------------Test------------------')
        out = model(dataset_train['x'].cuda().detach().float(), dataset_train['t'].cuda().detach().float(), dataset_train['d'].cuda().detach().float())
        plot_tnse_ihdp(out[0], dataset_train['t'], model_name=args.model_name)
        mise, dpe, pe, ate = compute_eval_metrics(dataset_test['x'], num_treatments=args.num_treatments,model=model)
        print("Mise: %s" % str(mise))
        print("DPE: %s" % str(dpe))
        print("PE: %s" % str(pe))
        print("ATE: %s" % str(ate))
        result['out'].append(ate)
        result['test']['mise'].append(mise)
        result['test']['dpe'].append(dpe)
        result['test']['pe'].append(pe)
        result['test']['ate'].append(ate)
        mise, dpe, pe, ate = compute_eval_metrics(dataset_val['x'], num_treatments=args.num_treatments,model=model)
        print('-----------------Val------------------')
        print("Mise: %s" % str(mise))
        print("DPE: %s" % str(dpe))
        print("PE: %s" % str(pe))
        print("ATE: %s" % str(ate))
        result['in'].append(ate)
        result['val']['mise'].append(mise)
        result['val']['dpe'].append(dpe)
        result['val']['pe'].append(pe)
        result['val']['ate'].append(ate)
        with open(export_dir + '/p' + str(args.p) + str(args.beta) + 'result.json', 'w') as fp:
            json.dump(result, fp)