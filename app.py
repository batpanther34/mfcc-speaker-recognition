import numpy as np
import scipy.fftpack
import scipy.signal
import matplotlib.pyplot as plt
import librosa
import librosa.display
import streamlit as st
from datetime import datetime
import os
import uuid

# Student information
STUDENT_NAME = "Sahil Kori"
STUDENT_ID = "2311401234"

def generate_timestamp():
    """Generate a timestamp for the current run"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def generate_uuid():
    """Generate a unique identifier for the current run"""
    return str(uuid.uuid4())

def frame_blocking(signal, frame_size=256, frame_step=100):
    """Divide the signal into overlapping frames"""
    signal_length = len(signal)
    frames = []
    
    for i in range(0, signal_length - frame_size, frame_step):
        frame = signal[i:i+frame_size]
        frames.append(frame)
    
    return np.array(frames)

def apply_window(frames, window_type='hamming'):
    """Apply window function to each frame"""
    if window_type == 'hamming':
        window = np.hamming(len(frames[0]))
    elif window_type == 'hanning':
        window = np.hanning(len(frames[0]))
    else:
        window = np.ones(len(frames[0]))
    
    return frames * window

def compute_power_spectrum(frames, n_fft=256):
    """Compute power spectrum for each frame"""
    mag_frames = np.absolute(np.fft.rfft(frames, n_fft))
    pow_frames = ((1.0 / n_fft) * (mag_frames ** 2))
    return pow_frames

def mel_filter_bank(sample_rate, n_fft=256, n_filters=26):
    """Create a Mel filter bank"""
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, n_filters + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    
    bin = np.floor((n_fft + 1) * hz_points / sample_rate)
    fbank = np.zeros((n_filters, int(np.floor(n_fft / 2 + 1))))
    
    for m in range(1, n_filters + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right
        
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    
    return fbank

def compute_mfcc(power_spectrum, mel_filter_bank, n_ceps=13):
    """Compute MFCC coefficients"""
    filter_banks = np.dot(power_spectrum, mel_filter_bank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical stability
    filter_banks = 20 * np.log10(filter_banks)  # dB
    
    mfcc = scipy.fftpack.dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (n_ceps + 1)]
    return mfcc, filter_banks

def plot_waveform(signal, sample_rate):
    """Plot the time-domain waveform"""
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.waveshow(signal, sr=sample_rate, ax=ax)
    ax.set_title('Time-domain Waveform')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    st.pyplot(fig)

def plot_power_spectrum(power_spectrum, sample_rate, frame_size=256):
    """Plot the power spectrum"""
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(10 * np.log10(power_spectrum.T), 
                  aspect='auto', origin='lower', 
                  extent=[0, len(power_spectrum), 0, sample_rate/2])
    ax.set_title('Power Spectrum (dB)')
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Frequency (Hz)')
    plt.colorbar(im, ax=ax, format='%+2.0f dB')
    st.pyplot(fig)

def plot_mel_filters(mel_filter_bank, sample_rate, frame_size=256):
    """Plot the mel filter bank"""
    fig, ax = plt.subplots(figsize=(10, 4))
    for i in range(mel_filter_bank.shape[0]):
        ax.plot(mel_filter_bank[i])
    ax.set_title('Mel Filter Bank')
    ax.set_xlabel('Frequency Bin')
    ax.set_ylabel('Amplitude')
    st.pyplot(fig)

def plot_mfcc(mfcc):
    """Plot MFCC coefficients as a heatmap"""
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(mfcc.T, aspect='auto', origin='lower')
    ax.set_title('MFCC Coefficients')
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('MFCC Coefficient Index')
    plt.colorbar(im, ax=ax)
    st.pyplot(fig)

def process_audio(file_path, frame_size=256, frame_step=100, n_filters=26, n_ceps=13):
    """Process audio file through the MFCC pipeline"""
    # Load audio file
    signal, sample_rate = librosa.load(file_path, sr=None)
    
    # Frame blocking
    frames = frame_blocking(signal, frame_size, frame_step)
    
    # Windowing
    windowed_frames = apply_window(frames)
    
    # Power spectrum
    power_spectrum = compute_power_spectrum(windowed_frames, frame_size)
    
    # Mel filter bank
    filter_bank = mel_filter_bank(sample_rate, frame_size, n_filters)
    
    # MFCC computation
    mfcc, filter_banks = compute_mfcc(power_spectrum, filter_bank, n_ceps)
    
    return {
        'signal': signal,
        'sample_rate': sample_rate,
        'frames': frames,
        'windowed_frames': windowed_frames,
        'power_spectrum': power_spectrum,
        'filter_bank': filter_bank,
        'filter_banks': filter_banks,
        'mfcc': mfcc,
        'frame_size': frame_size,
        'frame_step': frame_step,
        'n_filters': n_filters,
        'n_ceps': n_ceps
    }

def main():
    """Main Streamlit application"""
    st.title("Automatic Speaker Recognition using MFCC Feature Extraction")
    
    # Display student information
    st.sidebar.info(f"Submitted by: {STUDENT_NAME} | Roll No: {STUDENT_ID}")
    st.sidebar.info(f"Run ID: {generate_uuid()}")
    st.sidebar.info(f"Timestamp: {generate_timestamp()}")
    
    # Parameters
    st.sidebar.header("Processing Parameters")
    frame_size = st.sidebar.slider("Frame size (samples)", 128, 512, 256)
    frame_step = st.sidebar.slider("Frame step (samples)", 50, 200, 100)
    n_filters = st.sidebar.slider("Number of Mel filters", 20, 40, 26)
    n_ceps = st.sidebar.slider("Number of MFCC coefficients", 5, 20, 13)
    
    # File upload
    uploaded_file = st.file_uploader("Upload a WAV audio file", type=["wav"])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_file = "temp_audio.wav"
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Process the audio file
        with st.spinner("Processing audio..."):
            result = process_audio(temp_file, frame_size, frame_step, n_filters, n_ceps)
        
        # Remove temporary file
        os.remove(temp_file)
        
        # Display results
        st.header("Input Audio")
        plot_waveform(result['signal'], result['sample_rate'])
        
        st.header("Power Spectrum")
        plot_power_spectrum(result['power_spectrum'], result['sample_rate'], result['frame_size'])
        
        st.header("Mel Filter Bank")
        plot_mel_filters(result['filter_bank'], result['sample_rate'], result['frame_size'])
        
        st.header("MFCC Coefficients")
        plot_mfcc(result['mfcc'])
        
        # Show parameters used
        st.header("Processing Parameters Used")
        st.write(f"- Frame size: {result['frame_size']} samples")
        st.write(f"- Frame step: {result['frame_step']} samples")
        st.write(f"- Number of Mel filters: {result['n_filters']}")
        st.write(f"- Number of MFCC coefficients: {result['n_ceps']}")
        st.write(f"- Sample rate: {result['sample_rate']} Hz")
        
        # Interpretation
        st.header("Interpretation")
        st.write("""
        The MFCC (Mel-Frequency Cepstral Coefficients) represent the short-term power spectrum of sound. 
        They are derived from a type of cepstral representation of the audio clip and are commonly used 
        in speech and speaker recognition systems.
        
        - **Time-domain waveform**: Shows the amplitude of the audio signal over time.
        - **Power spectrum**: Shows the distribution of power into frequency components.
        - **Mel filter bank**: Triangular filters spaced according to the mel scale to mimic human auditory perception.
        - **MFCC coefficients**: The final feature vectors that capture the characteristics of the speaker's voice.
        """)

if __name__ == "__main__":
    main()