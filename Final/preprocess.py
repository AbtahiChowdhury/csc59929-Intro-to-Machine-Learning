import os
import math
import json
import librosa

DATASET_PATH = 'genres'
JSON_PATH = 'data.json'

SAMPLE_RATE = 22050
DURATION = 30 # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def extract_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    
    # dictionary to store data
    data = {
        'mapping': [],
        'mfcc': [],
        'labels': []
    }

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)

    # loop through all the genres
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        
        # check if not at root level
        if dirpath is not dataset_path:

            # save the semantic label
            dirpath_components = dirpath.split('/')
            semantic_label = (dirpath_components[-1])[len(dataset_path)+1:]
            data['mapping'].append(semantic_label)
            print(f'\nProcessing {semantic_label}')

            # process files for a specific genre
            for f in filenames:

                # load audio file
                file_path = os.path.join(dirpath,f)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                # process segments: extract mfcc and store data
                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s
                    finish_sample = start_sample + num_samples_per_segment

                    mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample], sr=sr, n_fft=n_fft, n_mfcc=n_mfcc, hop_length=hop_length)
                    mfcc = mfcc.T
                    
                    # store mfcc for segment if it has the expected length
                    if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                        data['mfcc'].append(mfcc.tolist())
                        data['labels'].append(i-1)
                        print(f'{file_path}, segment: {s+1}')
    
    with open(json_path, 'w') as fp:
        json.dump(data, fp, indent=4)

if __name__ == '__main__':
    extract_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)