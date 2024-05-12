import numpy as np
import matplotlib.pyplot as plt
import data_visualisation as dv

def add_random_noise(ecg, noise_level_range=(0, 0.05)):
    """Adds random noise to an ECG signal.

    Args:
        ecg (np.array): 12 lead ecg, numpy array of shape (5000, 12)
        noise_level (float, optional): The strength of the noise. Defaults to 0.05.
    """
    ### generate high frequency noise
    
    # calc std of each lead
    stds = np.std(ecg, axis=0)
    
    # Generate noise for each lead using the standard deviations
    noise_level = np.random.uniform(*noise_level_range)
    noise = noise_level * stds * np.random.randn(*ecg.shape)
    ecg_noisy = ecg + noise
    return ecg_noisy

def generate_sin_drift(length, drift_wavelength_range):
    """
    Generate a sinusoidal curve to be used as baseline drift.
    
    Parameters:
        length (int): Length of the ECG signal.
        drift_wavelength_range (tuple): Range of wavelengths for the sinusoidal curve.
        
    Returns:
        array-like: Sinusoidal curve to be used as baseline drift.
    
    """
    time = np.arange(length)
    # random wavelength
    drift_wavelength = np.random.uniform(drift_wavelength_range[0], drift_wavelength_range[1])
    # random starting phase
    start_phase_pos = np.random.uniform(0, drift_wavelength)
    time = time + start_phase_pos

    # generate drift
    drift = np.sin(2 * np.pi * (1 / drift_wavelength) * time) #* drift_amplitude
    return drift

def add_random_baseline_drift(ecg, drift_wavelength_range=(1000, 4000), strength_range=(3,5)):
    """
    Add baseline drift to all leads of an ECG signal with a randomized sinusoidal curve.
    
    Parameters:
        ecg (np-array): 12-lead ECG signal, np array of shape (5000, 12).
        drift_wavelength_range (tuple): Range of wavelengths for the sinusoidal curve.
        strength_range (tuple): Range of strength of the baseline drift.
        
    Returns:
        array-like: 12-lead ECG signal with baseline drift added.
    """
    ecg_qranges = []

    # find IQR for each lead
    for lead in ecg.T:
        q75, q25 = np.percentile(lead, [75 ,25])
        ecg_qranges.append(q75 - q25)
        
    drift = generate_sin_drift(ecg.shape[0], drift_wavelength_range)
    strength = np.random.uniform(*strength_range)
    
    drifts = np.array([drift * ecg_qrange * strength for ecg_qrange in ecg_qranges])
    drifts = drifts.T
    
    ecg_with_drift = ecg + drifts
    return ecg_with_drift, drifts