import numpy as np
import os
import sys
from scipy.io import wavfile
from util import trim
from signatures import sign_track
from similarity import signatures_similarity


def compute_similarity(path1, path2):
    """
    Compute the similarity between two wav audio files

    Keyword arguments:
    path1 -- the path of the first audio file
    path2 -- the path of the second audio file

    Return:
    a tuple containing the similarity score, the name of the first file and the name of the second file
    """

    # Read the audio files
    fs1, data1 = wavfile.read(path1)
    fs2, data2 = wavfile.read(path2)
    if fs1 != fs2:
        print('Files do not have the same sample frequency')
        return 0.0

    # Align and sign the tracks
    track1 = trim(data1)
    track2 = trim(data2)
    signed1 = sign_track(track1, fs1)
    signed2 = sign_track(track2, fs2)

    # Sanitize the filenames
    filename1 = path1.split('/')[-1].split('.')[0]
    filename2 = path2.split('/')[-1].split('.')[0]
    return (signatures_similarity(signed1, signed2), filename1, filename2)


def compute_all_similarities(folder):
    """
    Compute the similarity between all pairs of wav audio files in the folder

    Keyword arguments:
    folder -- the folder where the audio files are located

    Return:
    a sorted list of tuples containing the similiarity score,
    the name of the first file and the name of the second file
    """

    signatures = {}
    similarities = []

    reference_fs, _ = wavfile.read(folder + '/' + os.listdir(folder)[0])

    # Sign all the audio files in the folder
    for filename in os.listdir(folder):
        fs, data = wavfile.read(folder + '/' + filename)
        if fs != reference_fs:
            print(filename, 'does not have the same sample frequency')
        else:
            track = trim(data)
            signature = sign_track(data, fs)
            signatures[filename] = signature

    # Compute the similarity score between each pair of signed tracks
    for idx, key1 in enumerate(list(signatures.keys())):
        signed1 = signatures[key1]
        for key2 in list(signatures.keys())[idx+1:]:
            signed2 = signatures[key2]
            similarity = signatures_similarity(signed1, signed2)
            similarities.append((similarity, key1.split('.')[0], key2.split('.')[0]))

    similarities.sort(reverse=True)
    return similarities


def print_usage():
    print("""
Usage:
    python3 MusicVersionSimilarity.py [Options] path_to_first_audio_file path_to_second_audio_file

Options:
    -h|--help       : Display this page
    -d|--directory  : Compute the similarity between each pair of audio files in the specified directory
                      Provide only the path to the directory with this option
    """)


def main():
    num_args = len(sys.argv)
    if num_args <= 1:
        print_usage()
    elif sys.argv[1] == '-h' or sys.argv[1] == '--help':
        print_usage()
    elif sys.argv[1] == '-d' or sys.argv[1] == '--directory':
        if num_args != 3:
            print_usage()
        else:
            folder = sys.argv[2]
            similarities = compute_all_similarities(folder)
            for similarity_score, filename1, filename2 in similarities:
                print(filename1, 'and', filename2, 'have a similarity score of', similarity_score)
    else:
        if num_args != 3:
            print_usage()
        else:
            path1 = sys.argv[1]
            path2 = sys.argv[2]
            similarity_score, filename1, filename2 = compute_similarity(path1, path2)
            print(filename1, 'and', filename2, 'have a similarity score of', similarity_score)


if __name__ == "__main__":
    main()
