import matplotlib.pyplot as plt
import numpy as np

twelve_lead = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

def plot_12_lead_ecg(ecg, label=None, leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'], save=False, save_path=None):
    """ 
    Plot a 12 lead ECG in a single figure.
    
    Parameters:
        ecg (np.array): 12 lead ecg, numpy array of shape (5000, 12)
        
    Returns:
        None
    """
    no_leads = ecg.shape[1]
    samp_length = ecg.shape[0]
    
    assert(no_leads == len(leads))
    fig, axs = plt.subplots(nrows=no_leads, sharex=True, figsize=(samp_length//50, no_leads//2))
    
    if label:
        fig.suptitle(f'ECG: {label}')
    
    for i in range(no_leads):
        axs[i].plot(ecg[:, i])
        axs[i].set_ylabel(leads[i])
        
    plt.show()
    
    if save:
        plt.savefig(save_path)
    
def plot_12_lead_ecgs(ecgs, labels=None):
    """ 
    Plot a list of 12 lead ECGs side by side in a single figure.
    
    Parameters:
        ecgs (list): List of 12 lead ecgs, each numpy array of shape (5000, 12)
        labels (list): List of labels for each ECG
        
    Returns:
        None
    """
    fig, axs = plt.subplots(nrows=12, ncols=len(ecgs), sharex=True, figsize=(5*len(ecgs), 8))
    for i in range (len(ecgs)):
        for j in range(12):
            axs[j, i].plot(ecgs[i][:, j])
            axs[j, i].set_ylabel(twelve_lead[j])
        if labels:
            axs[0, i].set_title(f'ECG: {labels[i]}')
    plt.subplots_adjust(top=0.92)
    plt.show()
    
def plot_12_lead_ecgs_superimposed(ecgs):
    """ 
    Plot a list of 12 lead ECGs superimposed in a single figure.
    
    Parameters:
        ecgs (list): List of 12 lead ecgs, each numpy array of shape (12, 5000)
        
    Returns:
        None
    """
    fig, ax = plt.subplots(nrows=12, figsize=(12, 8))
    for i in range (len(ecgs)):
        for j in range(12):
            ax[j].plot(ecgs[i][j], label=f'ECG {i}')
            ax[j].set_ylabel(twelve_lead[j])

    plt.show()