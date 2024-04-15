import matplotlib.pyplot as plt
import numpy as np

twelve_lead = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

def plot_12_lead_ecg(ecg):
    """ 
    Plot a 12 lead ECG in a single figure.
    
    Parameters:
        ecg (np.array): 12 lead ecg, numpy array of shape (12, 5000)
        
    Returns:
        None
    """
    fig, axs = plt.subplots(nrows=12, sharex=True, figsize=(12, 8))
    for i in range(12):
        axs[i].plot(ecg[i])
        axs[i].set_ylabel(twelve_lead[i])
    plt.subplots_adjust(top=0.92)
    plt.show()
    
def plot_12_lead_ecgs(ecgs):
    """ 
    Plot a list of 12 lead ECGs side by side in a single figure.
    
    Parameters:
        ecgs (list): List of 12 lead ecgs, each numpy array of shape (12, 5000)
        
    Returns:
        None
    """
    fig, axs = plt.subplots(nrows=12, ncols=len(ecgs), sharex=True, figsize=(12, 8))
    for i in range (len(ecgs)):
        for j in range(12):
            axs[j, i].plot(ecgs[i][j])
            axs[j, i].set_ylabel(twelve_lead[j])
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