import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
from copy import deepcopy
import numpy as np
import os
import sys
from model import *
from utils import MMD_loss
import time
import json

# loss function definition
def loss_function(y_hat, y, x_recon, x, mu, logvar, 
                  MMD_sigma, kernel_num, 
                  gamma1=1, gamma2=0):

    # Compute MMD loss between predicted perturbed samples and true perturbed samples
    mmd_function_pred = MMD_loss(fix_sigma=MMD_sigma, kernel_num=kernel_num)
    if y_hat is None:
        pred_loss = 0
    else:
        pred_loss = mmd_function_pred(y_hat, y)
    
    # Compute reconstruction loss between reconstructed control samples and true control samples
    mmd_function_recon = MMD_loss(fix_sigma=MMD_sigma, kernel_num=kernel_num)
    if gamma1 > 0:
        recon_mmd = mmd_function_recon(x_recon, x)
    else:
        recon_mmd = 0
    if gamma2 > 0:
        recon_mse = nn.MSELoss()(x_recon, x)
    else:
        recon_mse = 0
    recon_loss = gamma1*recon_mmd + gamma2*recon_mse
    
    # Compute KL divergence
    if logvar is None:
        KLD = 0
    else:
        KLD = -0.5*torch.sum(logvar -(mu.pow(2)+logvar.exp())+1)/x.shape[0]
    
    return pred_loss, recon_loss, KLD


# Training the model
def train_validate(
    dataloader,
    dataloader_infer,
    dataloader_val,
    opts,
    device,
    savedir,
    model,
    log
    ):

    if log:
        project_name = f'{model}_{opts.dataset_name}_{opts.leave_out_test_set_id}'
        wandb.init(project=project_name, name=savedir.split('/')[-1])
        wandb.define_metric("batch_step")
        wandb.define_metric("epoch")
        wandb.define_metric("batch/*", step_metric="batch_step")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*", step_metric="epoch")
        global_batch_step = 0
    
    if model == 'MORPH':
        mvae = MORPH(
            dim = opts.dim,
            c_dim = opts.cdim,
            opts = opts,
            device = device
        )
    elif model == 'MORPH_no_residual1':
        mvae = MORPH_no_residual1(
            dim = opts.dim,
            c_dim = opts.cdim,
            opts = opts,
            device = device
        )
    elif model == "MORPH_moe_3expert":
        mvae = MORPH_moe_3expert(
            dim = opts.dim,
            c_dim = opts.cdim,
            opts = opts,
            device = device
        )
    else:
        raise ValueError(f"Unknown model type: {model}")
    
    # move model to device
    mvae.double()
    mvae.to(device)
    optimizer = torch.optim.Adam(params=mvae.parameters(), lr=opts.lr)
    mvae.train()
    print("Training for maximum {} epochs...".format(str(opts.epochs)))
    
    ## Loss parameters
    # Beta schedule (KL weight): start with a small positive floor to prevent
    # unconstrained KL inflation, then ramp to mxBeta.
    BETA_FLOOR = 0.01
    beta_schedule = torch.full((opts.epochs,), BETA_FLOOR)
    if opts.modality == 'rna':
        warmup_epochs = 10
        beta_schedule[:warmup_epochs] = BETA_FLOOR
        beta_schedule[warmup_epochs:] = torch.linspace(
            BETA_FLOOR, opts.mxBeta, opts.epochs - warmup_epochs
        )
    else:
        warmup_epochs = 5
        beta_schedule[:warmup_epochs] = BETA_FLOOR
        beta_schedule[warmup_epochs:] = torch.linspace(
            BETA_FLOOR, opts.mxBeta, opts.epochs - warmup_epochs
        )

    # Alpha schedule (MMD prediction weight)
    if opts.batch_size * len(dataloader) > 1e6:
        alpha_schedule = torch.zeros(opts.epochs)
        alpha_schedule[:1] = 0
        alpha_schedule[1:int(opts.epochs/2)] = torch.linspace(0, opts.mxAlpha, int(opts.epochs/2)-1) 
        alpha_schedule[int(opts.epochs/2):] = opts.mxAlpha
    else:
        alpha_schedule = torch.zeros(opts.epochs)
        alpha_schedule[:5] = 0
        alpha_schedule[5:int(opts.epochs/2)] = torch.linspace(0, opts.mxAlpha, int(opts.epochs/2)-5) 
        alpha_schedule[int(opts.epochs/2):] = opts.mxAlpha

    # Save schedules for debugging
    schedule_path = os.path.join(savedir, 'loss_schedules.json')
    with open(schedule_path, "w") as json_file:
        json.dump({
            "alpha_schedule": alpha_schedule.tolist(),
            "beta_schedule": beta_schedule.tolist(),
        }, json_file, indent=4)
    print(f"Loss schedules saved to {schedule_path}") 
    
    min_train_loss = np.inf
    best_model = deepcopy(mvae)
    min_val_loss = np.inf
    best_model_val = deepcopy(mvae)

    # Define tolenrance and patience for early stopping
    tolerance = opts.tolerance_epochs
    patience = 0

    # Start timing
    start_time = time.time()
    for epoch in range(0, opts.epochs):
        lossAv = 0
        ct = 0
        mmdAv = 0
        reconAv = 0
        klAv = 0

        # train
        for (i, X) in tqdm(enumerate(dataloader)):
            x = X[0] #control samples
            y = X[1] #perturbation samples
            c_1 = X[2] #perturbation labels (target 1)
            c_2 = X[3] #perturbation labels (target 2)
            if 'moe' in model:
                c_1_2 = X[5] #perturbation labels (target 1)
                c_2_2 = X[6] #perturbation labels (target 2)
                if '3expert' in model:
                    c_1_3 = X[7]
                    c_2_3 = X[8]
            
            if mvae.cuda:
                x = x.to(device)
                y = y.to(device)
                c_1 = c_1.to(device)
                c_2 = c_2.to(device)
                if 'moe' in model:
                    c_1_2 = c_1_2.to(device)
                    c_2_2 = c_2_2.to(device)
                    if '3expert' in model:
                        c_1_3 = c_1_3.to(device)
                        c_2_3 = c_2_3.to(device)
                
            optimizer.zero_grad()

            if 'moe' in model:
                if '3expert' in model:
                    y_hat, x_recon, z_mu, z_logvar = mvae(x,c_1, c_2, c_1_2, c_2_2, c_1_3, c_2_3)
                else:
                    y_hat, x_recon, z_mu, z_logvar = mvae(x,c_1, c_2, c_1_2, c_2_2)
            else:
                y_hat, x_recon, z_mu, z_logvar = mvae(x,c_1, c_2)

            mmd_loss, recon_loss, kl_loss = loss_function(y_hat=y_hat, y=y, x_recon=x_recon, x=x, mu=z_mu, logvar=z_logvar, 
                                                          MMD_sigma=opts.MMD_sigma, kernel_num=opts.kernel_num, 
                                                          gamma1=opts.Gamma1, gamma2=opts.Gamma2)
            loss = alpha_schedule[epoch] * mmd_loss + recon_loss + beta_schedule[epoch]*kl_loss

            if(recon_loss.isnan()):
                print('recon_loss: ',recon_loss)
                print('y_hat: ',y_hat)
                print('x_recon: ',x_recon)
                print('x: ',x)
                sys.exit()

            loss.backward()
            if opts.grad_clip:
                for param in mvae.parameters():
                    if param.grad is not None:
                        param.grad.data = param.grad.data.clamp(min=-0.5, max=0.5)
            optimizer.step()

            ct += 1
            lossAv += loss.detach().cpu().numpy()
            mmdAv += mmd_loss.detach().cpu().numpy()
            reconAv += recon_loss.detach().cpu().numpy()
            if z_logvar is not None:
                klAv += kl_loss.detach().cpu().numpy()
            else:
                klAv = 0

            if log:
                wandb.log({
                    'batch_step': global_batch_step,
                    'batch/loss': loss,
                    'batch/mmd_loss': mmd_loss,
                    'batch/recon_loss': recon_loss,
                    'batch/kl_loss': kl_loss,
                })
                global_batch_step += 1

        print('Epoch '+str(epoch)+': Loss='+str(lossAv/ct)+', '+'MMD='+str(mmdAv/ct)+', '+'MSE='+str(reconAv/ct)+', '+'KL='+str(klAv/ct))
        
        epoch_metrics = {}
        if log:
            epoch_metrics.update({
                'epoch': epoch,
                'train/loss': lossAv / ct,
                'train/mmd_loss': mmdAv / ct,
                'train/recon_loss': reconAv / ct,
                'train/kl_loss': klAv / ct,
                'train/alpha': alpha_schedule[epoch],
                'train/beta': beta_schedule[epoch],
                'train/Gamma1': opts.Gamma1,
                'train/Gamma2': opts.Gamma2,
            })
        
        if opts.mxBeta >= 1:
            if (mmdAv + reconAv + klAv)/ct < min_train_loss:
                min_train_loss = (mmdAv + reconAv + klAv)/ct 
                best_model = deepcopy(mvae)
                torch.save(best_model, os.path.join(savedir, 'best_model.pt'))
                if log:
                    epoch_metrics['train/min_loss'] = min_train_loss
                    epoch_metrics['train/min_epoch'] = epoch
        else:
            if (mmdAv + reconAv)/ct < min_train_loss:
                min_train_loss = (mmdAv + reconAv)/ct 
                best_model = deepcopy(mvae)
                torch.save(best_model, os.path.join(savedir, 'best_model.pt'))
                if log:
                    epoch_metrics['train/min_loss'] = min_train_loss
                    epoch_metrics['train/min_epoch'] = epoch
        
        # Validation loop (validation - for early stopping and save best model)
        mvae.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation
            val_lossAv = 0
            val_mmdAv = 0
            val_reconAv = 0
            val_klAv = 0
            val_ct = 0
            
            for (i, X) in enumerate(dataloader_val):
                x = X[0]  # control samples
                y = X[1]  # perturbation samples
                c_1 = X[2]  # perturbation labels (target 1)
                c_2 = X[3]  # perturbation labels (target 2)
                ptb_target = X[4] #name of perturbation target

                if 'moe' in model:
                    c_1_2 = X[5]
                    c_2_2 = X[6]
                    if '3expert' in model:
                        c_1_3 = X[7]
                        c_2_3 = X[8]

                if mvae.cuda:
                    x = x.to(device)
                    y = y.to(device)
                    c_1 = c_1.to(device)
                    c_2 = c_2.to(device)
                    if 'moe' in model:
                        c_1_2 = c_1_2.to(device)
                        c_2_2 = c_2_2.to(device)
                        if '3expert' in model:
                            c_1_3 = c_1_3.to(device)
                            c_2_3 = c_2_3.to(device)

                if 'moe' in model:
                    if '3expert' in model:
                        y_hat, x_recon, z_mu, z_logvar = mvae(x, c_1, c_2, c_1_2, c_2_2, c_1_3, c_2_3)
                    else:
                        y_hat, x_recon, z_mu, z_logvar = mvae(x, c_1, c_2, c_1_2, c_2_2)
                else:
                    y_hat, x_recon, z_mu, z_logvar = mvae(x, c_1, c_2)
                
                mmd_loss, recon_loss, kl_loss = loss_function(y_hat=y_hat, y=y, x_recon=x_recon, x=x, mu=z_mu, logvar=z_logvar,
                                                              MMD_sigma=opts.MMD_sigma, kernel_num=opts.kernel_num, 
                                                              gamma1=opts.Gamma1, gamma2=opts.Gamma2)
                if z_logvar is not None:
                    val_loss = mmd_loss + recon_loss + kl_loss
                else:
                    val_loss = mmd_loss + recon_loss

                val_ct += 1
                val_lossAv += val_loss.detach().cpu().numpy()
                val_mmdAv += mmd_loss.detach().cpu().numpy()
                val_reconAv += recon_loss.detach().cpu().numpy()
                if z_logvar is not None:
                    val_klAv += kl_loss.detach().cpu().numpy()
                else:
                    val_klAv = 0

            print('Validation - Epoch ' + str(epoch) + ': Loss=' + str(val_lossAv / val_ct) + ', MMD=' + str(val_mmdAv / val_ct) + ', MSE=' + str(val_reconAv / val_ct) + ', KL=' + str(val_klAv / val_ct))

            if log:
                epoch_metrics.update({
                    'val/loss': val_lossAv / val_ct,
                    'val/mmd_loss': val_mmdAv / val_ct,
                    'val/recon_loss': val_reconAv / val_ct,
                    'val/kl_loss': val_klAv / val_ct,
                })
            
            if opts.mxBeta >= 1: 
                if (val_mmdAv + val_reconAv + val_klAv)/val_ct < min_val_loss:
                    min_val_loss = (val_mmdAv + val_reconAv + val_klAv)/val_ct 
                    best_model_val = deepcopy(mvae)
                    torch.save(best_model_val, os.path.join(savedir, 'best_model_val.pt'))
                    patience = 0
                    if log:
                        epoch_metrics['val/min_loss'] = min_val_loss
                        epoch_metrics['val/min_epoch'] = epoch
                else:
                    patience += 1
            else:
                if (val_mmdAv + val_reconAv)/val_ct < min_val_loss:
                    min_val_loss = (val_mmdAv + val_reconAv)/val_ct 
                    best_model_val = deepcopy(mvae)
                    torch.save(best_model_val, os.path.join(savedir, 'best_model_val.pt'))
                    patience = 0
                    if log:
                        epoch_metrics['val/min_loss'] = min_val_loss
                        epoch_metrics['val/min_epoch'] = epoch
                else:
                    patience += 1
            
            if log:
                epoch_metrics['val/patience'] = patience
                wandb.log(epoch_metrics)

            if patience >= tolerance:
                print(f"Early stopping on epoch {epoch}. No improvement in validation loss for {tolerance} rounds.")
                break
        
        mvae.train()  # Set the model back to training mode
    
    # Calculate the total duration
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Training completed in {total_time} seconds.")
    if log:
        wandb.log({'train/total_time': total_time})
    last_model = deepcopy(mvae)
    torch.save(last_model, os.path.join(savedir, 'last_model.pt'))
    