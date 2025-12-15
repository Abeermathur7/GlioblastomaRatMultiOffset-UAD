
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models import *
import datetime
import os
import numpy as np
import tqdm
def test(model, test_loader):

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
        )
    model.to(device)

    model.eval()

    # all_outputs = np.zeros((len(test_loader.dataset), input_dims))
    # batch_counter = 0
    all_outputs = []

    with torch.no_grad():
        for data in test_loader:
            inputs = data[0].to(device)
            outputs = model(inputs)
            
            # all_outputs[batch_counter:(batch_counter + data.shape[0]),:] = np.squeeze(outputs.cpu())
            # batch_counter += data.shape[0]
            all_outputs.append(np.squeeze(outputs.cpu()))
            
    return np.concatenate(all_outputs, axis = 0)
        


def train(
    model,
    epochs,
    train_loader,
    val_loader,
    model_save_path,
    model_name,
    denoise=False,
    optimizer_name='sgd',
    lr=1e-1,
    momentum=0.9,
    weight_decay=0.0,
    scheduler_name='cosine',
    patience=5,
    criterion_name='mse',     # 'mse', 'mae', 'huber', 'smoothl1', 'kld'
    device=None,
    save_checkpoints=True,
):
    # device selection
    if device is None:
        device = 'cuda' if torch.cuda.is_available() \
                 else 'mps'  if torch.backends.mps.is_available() \
                 else 'cpu'
    model.to(device)

    # build optimizer
    if optimizer_name.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr,
                              momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer {optimizer_name}")

    # build scheduler
    if scheduler_name.lower() == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name.lower() == 'onecycle':
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr,
            steps_per_epoch=len(train_loader),
            epochs=epochs
        )
    else:
        scheduler = None

    # select criterion
    if criterion_name == 'mse':
        criterion = nn.MSELoss()
    elif criterion_name == 'mae':
        criterion = nn.L1Loss()
    elif criterion_name in ('huber', 'smoothl1'):
        criterion = nn.SmoothL1Loss()
    elif criterion_name == 'kld':
        criterion = nn.KLDivLoss(reduction='batchmean')
    else:
        raise ValueError(f"Unknown criterion {criterion_name}")

    history = {'train': [], 'val': []}
    best_vloss, best_epoch = float('inf'), 0

    for epoch in range(1, epochs + 1):
        # — training pass —
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            inputs = batch[0].to(device)
            labels = batch[1].to(device) if denoise else inputs

            optimizer.zero_grad()
            outputs = model(inputs)

            if criterion_name == 'kld':
                # apply along the sequence dimension (dim=2)
                log_p = F.log_softmax(outputs, dim=2)
                p_target = F.softmax(labels, dim=2)
                loss = criterion(log_p, p_target)
            else:
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # — validation pass —
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch[0].to(device)
                labels = batch[1].to(device) if denoise else inputs
                outputs = model(inputs)

                if criterion_name == 'kld':
                    log_p = F.log_softmax(outputs, dim=2)
                    p_target = F.softmax(labels, dim=2)
                    val_loss += criterion(log_p, p_target).item()
                else:
                    val_loss += criterion(outputs, labels).item()
        val_loss /= len(val_loader)

        # scheduler & history
        if scheduler is not None:
            scheduler.step()
        history['train'].append(train_loss)
        history['val'  ].append(val_loss)

        print(f"Epoch {epoch:02d} | train={train_loss:.4f} val={val_loss:.4f}")

        # checkpoint & early‐stop
        if save_checkpoints and val_loss < best_vloss:
            best_vloss, best_epoch = val_loss, epoch
            torch.save(model.state_dict(), f"{model_save_path}/{model_name}.pth")
        elif save_checkpoints and (epoch - best_epoch) >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    return history

def train_vae(model, epochs, train_loader, val_loader, model_save_path, model_name):

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
        )
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr = 0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 10)

    history = {'Training loss': [],
               'Reconstruction loss': [],
               'KL loss': [],
                'Validation loss': []}

    best_vloss = 1e6
    early_stop_thresh = 10
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.mkdir(os.path.join(model_save_path, str(timestamp)))
    
    chkpt_dir = os.path.join(model_save_path, str(timestamp))
    
    for epoch in range(epochs):
        
        print('EPOCH {}:'.format(epoch + 1))
        running_train_loss = 0
        running_recon_loss = 0
        running_kl_loss = 0
        running_val_loss = 0

        model.train()
        for _, data in tqdm.tqdm(enumerate(train_loader, 0), unit="batch", total=len(train_loader)):
            
                # inputs = data.to(device)
                inputs = data[0].to(device)      
                optimizer.zero_grad()
                outputs, mu_train, var_train = model(inputs)
                loss, recon_loss, kl_loss = vae_lossfn(outputs, inputs, mu_train, var_train)
                loss.backward()
                
                optimizer.step()
                
                running_train_loss += loss.item()  
                running_recon_loss += recon_loss.item()
                running_kl_loss += kl_loss.item()
        
        model.eval()
        with torch.no_grad():
            for _, vdata in enumerate(val_loader):
                v_inputs = vdata[0].to(device)
                
                outputs, mu_val, var_val = model(v_inputs)
                vloss, _, _ = vae_lossfn(outputs, v_inputs, mu_val, var_val)

                running_val_loss += vloss.item()
        
        avg_loss = running_train_loss / len(data)
        avg_recon_loss = running_recon_loss / len(data)
        avg_kl_loss = running_kl_loss / len(data)
        avg_vloss = running_val_loss / len(vdata)
        

        history['Training loss'].append(avg_loss)
        history['Reconstruction loss'].append(avg_recon_loss)
        history['KL loss'].append(avg_kl_loss)
        history['Validation loss'].append(avg_vloss)


        print('Loss: train {} valid {} recon {} KL {}'.format(avg_loss, avg_vloss, avg_recon_loss, avg_kl_loss))

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss  
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(chkpt_dir,model_name + '.pth'))
        elif epoch - best_epoch > early_stop_thresh:
            print('Early stopping stopped at epoch {}'.format(epoch))
            break
    
    scheduler.step()
    
    return history
def vae_lossfn(recon, inputs, mu, logvar, beta=1e-4):
    recon_loss = torch.nn.functional.mse_loss(recon, inputs, reduction='mean')
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld, recon_loss, kld

def test_vae(model, test_loader):

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
        )
    model.to(device)

    model.eval()

    # all_outputs = np.zeros((len(test_loader.dataset), input_dims))
    # batch_counter = 0
    all_outputs = []
    all_mu = []
    all_var = []

    with torch.no_grad():
        for data in test_loader:
            inputs = data.to(device)
            outputs, mu, var = model(inputs)
            
            # all_outputs[batch_counter:(batch_counter + data.shape[0]),:] = np.squeeze(outputs.cpu())
            # batch_counter += data.shape[0]
            all_outputs.append(np.squeeze(outputs.cpu()))
            all_mu.append(np.squeeze(mu.cpu()))
            all_var.append(np.squeeze(var.cpu()))
    
    all_outputs = np.concatenate(all_outputs, axis = 0)
    all_mu = np.concatenate(all_mu, axis = 0)
    all_var = np.concatenate(all_var, axis = 0)
     
    return all_outputs, all_mu, all_var 

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
