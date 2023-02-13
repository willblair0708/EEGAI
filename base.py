import numpy as np
import matplotlib.pyplot as plt

# Load the EEG data
eeg_data = np.loadtxt('eeg_signal.txt')

# Plot the EEG data
plt.plot(eeg_data)
plt.xlabel('Sample Number')
plt.ylabel('Amplitude')
plt.title('EEG Signal')
plt.show()

# Filter the EEG data
low_pass = 0.1
high_pass = 1.0

b, a = scipy.signal.butter(5, [low_pass, high_pass], 'bandpass')
filtered_eeg = scipy.signal.filtfilt(b, a, eeg_data)

# Plot the filtered EEG data
plt.plot(filtered_eeg)
plt.xlabel('Sample Number')
plt.ylabel('Amplitude')
plt.title('Filtered EEG Signal')
plt.show()

def detect_features(psd, f, threshold=0.5):
    # Normalize the power spectral density
    psd = psd / np.sum(psd)
    
    # Find the peaks in the power spectral density
    peaks, _ = scipy.signal.find_peaks(psd, height=threshold)
    
    return peaks, f[peaks]

def process_eeg_signal(eeg_data):
    # Filter the EEG data
    low_pass = 0.1
    high_pass = 1.0
    b, a = scipy.signal.butter(5, [low_pass, high_pass], 'bandpass')
    filtered_eeg = scipy.signal.filtfilt(b, a, eeg_data)
    
    # Calculate the power spectral density of the filtered EEG data
    f, psd = scipy.signal.welch(filtered_eeg, fs=1000, nperseg=256)
    
    return f, psd

# Load the EEG data
eeg_data = np.loadtxt('eeg_signal.txt')

# Process the EEG data
f, psd = process_eeg_signal(eeg_data)

# Plot the power spectral density of the EEG data
plt.plot(f, psd)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density')
plt.title('EEG Power Spectral Density')
plt.show()

def process_eeg_signal(eeg_data):
    # Filter the EEG data
    low_pass = 0.1
    high_pass = 1.0
    b, a = scipy.signal.butter(5, [low_pass, high_pass], 'bandpass')
    filtered_eeg = scipy.signal.filtfilt(b, a, eeg_data)
    
    # Calculate the power spectral density of the filtered EEG data
    f, psd = scipy.signal.welch(filtered_eeg, fs=1000, nperseg=256)
    
    # Find the peak frequency
    peak_freq = f[np.argmax(psd)]
    
    return peak_freq

# Load the EEG data
eeg_data = np.loadtxt('eeg_signal.txt')

# Process the EEG data
peak_freq = process_eeg_signal(eeg_data)

# Print the peak frequency
print(peak_freq)

# Detect features in the power spectral density
peaks, peak_frequencies = detect_features(psd, f)

# Plot the power spectral density and the detected features
plt.plot(f, psd)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density')
plt.title('EEG Power Spectral Density')
plt.plot(peak_frequencies, psd[peaks], 'ro')
plt.show()
