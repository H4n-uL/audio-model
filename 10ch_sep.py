import soundfile as sf
import numpy as np
from scipy import signal

def create_lr_bandpass(low_cutoff, high_cutoff, srate, order=4):
    nyq = srate * 0.5
    lowpass = None
    if high_cutoff is not None and high_cutoff < nyq:
        b_low, a_low = signal.butter(order//2, high_cutoff/nyq, btype='low')
        lowpass = (np.convolve(b_low, b_low), np.convolve(a_low, a_low))

    highpass = None
    if low_cutoff is not None and low_cutoff > 0:
        b_high, a_high = signal.butter(order//2, low_cutoff/nyq, btype='high')
        highpass = (np.convolve(b_high, b_high), np.convolve(a_high, a_high))

    return highpass, lowpass

def apply_phase_shift(data, degrees=90):
    analytic_signal = signal.hilbert(data)
    phase_shift = np.exp(1j * np.deg2rad(degrees))
    shifted = np.real(analytic_signal * phase_shift)
    return shifted * (np.max(np.abs(data)) / np.max(np.abs(shifted)))

def process_bandpass(input_file, lo, hi, order=4, phase_shift=None) -> np.ndarray:
    data, srate = sf.read(input_file)
    if len(data.shape) == 1: data = np.expand_dims(data, axis=1)

    high_filters, low_filters = create_lr_bandpass(lo, hi, srate, order)
    output = np.zeros_like(data)

    for channel in range(data.shape[1]):
        temp = data[:, channel]
        if high_filters is not None: temp = signal.filtfilt(high_filters[0], high_filters[1], temp)
        if low_filters is not None: temp = signal.filtfilt(low_filters[0], low_filters[1], temp)
        if phase_shift is not None: temp = apply_phase_shift(temp, phase_shift)
        output[:, channel] = temp

    return output

input_file = "input.wav"

_, srate = sf.read(input_file)
LR = process_bandpass(input_file, None, None)
C = np.mean(process_bandpass(input_file, 300, 3000), axis=1).reshape(-1, 1)
TL_TR = process_bandpass(input_file, 200, 20000)
SL_SR = process_bandpass(input_file, 500, 4000, phase_shift=90)
TBL_TBR = process_bandpass(input_file, 200, 20000, phase_shift=90)
LFE = np.mean(process_bandpass(input_file, 0, 120), axis=1).reshape(-1, 1)

def dB(db): return 10 ** (db / 20)

output_channels = [
    LR[:, 0],               # L
    LR[:, 1],               # R
    C[:, 0] * dB(-3),       # C
    LFE[:, 0] * dB(-10),    # LFE
    SL_SR[:, 0] * dB(-6),   # SL
    SL_SR[:, 1] * dB(-6),   # SR
    TBL_TBR[:, 0] * dB(-6), # TBL
    TBL_TBR[:, 1] * dB(-6), # TBR
    TL_TR[:, 0] * dB(-4),   # TL
    TL_TR[:, 1] * dB(-4),   # TR
]
sf.write("target.left.w64", output_channels[0], srate)
sf.write("target.right.w64", output_channels[1], srate)
sf.write("target.centre.w64", output_channels[2], srate)
sf.write("target.lfe.w64", output_channels[3], srate)
sf.write("target.surround.left.w64", output_channels[4], srate)
sf.write("target.surround.right.w64", output_channels[5], srate)
sf.write("target.top.back.left.w64", output_channels[6], srate)
sf.write("target.top.back.right.w64", output_channels[7], srate)
sf.write("target.top.left.w64", output_channels[8], srate)
sf.write("target.top.right.w64", output_channels[9], srate)

# sf.write("target.full.w64", np.column_stack(output_channels), srate)
