import numpy as np


def trim(data):
    """
    Trim the signal to only contain relevant data and make sure the different versions are aligned.

    Keyword arguments:
    data -- the track signal

    Return:
    an array containing the trimmed signal
    """

    # We align the different versions by getting rid of the faint beginning before the music starts playing
    first_idx = (data > 250).argmax()
    last_idx = -(np.flip(data) != 0).argmax()
    if last_idx == 0:
        last_idx = data.size
    return data[first_idx:last_idx]


def notes_bins(fs):
    """
    Generates the frequency bins to isolate notes from A0 to C8 (the range of a modern piano)

    Keyword arguments:
    fs -- the sampling frequency

    Return:
    an array containing the bins' separations
    """

    ground_note = 440.0 # We choose A4 = 440 Hz
    base = 2 ** (1/12) # The constant ratio of the geometric series
    num_notes = 88 # Keys on a piano
    ground_note_offset = 48
    # Return an array of border frequencies between notes from A0 to C8 (the range of a modern piano)
    bins = [ground_note * (base ** (i - ground_note_offset - 0.5)) for i in range(num_notes + 1)]
    # Insert a frequency of 0 as the lower bound and the nyquist frequency as the upper bound
    bins.insert(0, 0.0)
    bins.append(fs / 2)
    return bins
