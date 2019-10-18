import numpy as np
from util import notes_bins


def top_k_frequencies(fs, data, k=20):
    """
    Compute the top k frequencies and associated amplitudes

    Keyword arguments:
    fs   -- the sampling frequency
    data -- the signal
    k    -- the number of frequencies and associated amplitudes desired (default 20)

    Return:
    a pair containing the top k frequencies and the top k associated amplitudes
    """

    num_samples = len(data)
    # Compute the DFT
    y = 2 * np.fft.fft(data)[:int(num_samples / 2)] / num_samples

    # Discard the phase and keep only the amplitude
    amplitudes = np.abs(y)
    # Create the frequencies associated with the amplitudes, up to half the sample rate
    frequencies = fs * np.arange((num_samples / 2)) / num_samples

    # Select top k amplitudes and associated frequencies
    top_k_idx  = np.argsort(-amplitudes)[:k]
    top_k_amp  = amplitudes[top_k_idx]
    top_k_freq = frequencies[top_k_idx]

    # Return the top k frequencies and associated amplitudes
    return top_k_freq, top_k_amp


def signature(top_k_frequencies, bins):
    """
    Compute the distribution of dominant notes from the top
    frequencies and associated amplitudes

    Keyword arguments:
    top_k_frequencies -- the dominant frequencies and associated amplitudes
    bins              -- the frequency bins separating notes

    Return:
    an array containing the distribution of dominant notes
    """

    hist = np.histogram(
        top_k_frequencies[0],
        bins    = bins,
        weights = top_k_frequencies[1]
    )[0]

    return hist / np.sum(hist)


def sign_track(track, fs):
    """
    Sign the track with the distribution of dominant
    notes at different times in the track

    Keyword arguments:
    track -- the signal
    fs    -- the sampling frequency

    Return:
    an array containing the distributions of dominant notes at different times in the track
    """

    bins = notes_bins(fs)
    # use smallest power of 2 that gives a window of more than one second
    window_width = 2 ** int(np.ceil(np.log2(fs)))
    step = window_width // 2 # To get a 50 % overlap
    signed_track = [
        signature(
            top_k_frequencies(
                fs,
                track[i:i + window_width],
                k=20
            ),
            bins
        )
        for i in range(0, track.size, step)
    ]
    return np.array(signed_track)
