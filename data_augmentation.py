import numpy as np
import matplotlib.pyplot as plt
import data_visualisation as dv

def add_random_noise(ecg, noise_level=0.05):
    """_summary_

    Args:
        ecg (np.array): 12 lead ecg, numpy array of shape (12, 5000)
        noise_level (float, optional): The strength of the noise. Defaults to 0.05.
    """
    ### generate high frequency noise
    std = np.std(ecg)
    level = std * noise_level
    noise = np.random.normal(0, level, ecg.shape)
    ecg_noisy = ecg + noise
    return ecg_noisy

def generate_sin_drift(length, frequency_range=(1 / 1000, 1 / 500), amplitude_range=(5, 10)):
    """
    Generate a sinusoidal curve to be used as baseline drift.
    
    Parameters:
        length (int): Length of the ECG signal.
        frequency_range (tuple): Range of frequencies for the sinusoidal curve.
        amplitude_range (tuple): Range of amplitudes for the sinusoidal curve.
        
    Returns:
        array-like: Sinusoidal curve to be used as baseline drift.
    
    """
    time = np.arange(length)
    drift_amplitude = np.random.uniform(amplitude_range[0], amplitude_range[1])
    drift_frequency = np.random.uniform(frequency_range[0], frequency_range[1])
    drift = np.sin(2 * np.pi * drift_frequency * time) * drift_amplitude
    return drift

def add_random_baseline_drift(ecg, frequency_range=(1 / 2000, 1 / 500), amplitude_range=(5, 20), strength=0.05):
    """
    Add baseline drift to all leads of an ECG signal with a randomized sinusoidal curve.
    
    Parameters:
        ecg (np-array): 12-lead ECG signal.
        frequency_range (tuple): Range of frequencies for the sinusoidal curve.
        amplitude_range (tuple): Range of amplitudes for the sinusoidal curve.
        strength (float): Strength of the baseline drift.
        
    Returns:
        array-like: 12-lead ECG signal with baseline drift added.
    """
    ecg_qranges = []
    for lead in ecg:
        # max - min
        q75, q25 = np.percentile(lead, [75 ,25])
        ecg_qranges.append(q75 - q25)
        
    drift = generate_sin_drift(ecg.shape[1], frequency_range, amplitude_range)
    drifts = np.array([drift * ecg_range * strength for ecg_range in ecg_qranges])
    ecg_with_drift = ecg + drifts
    
    return ecg_with_drift, drifts