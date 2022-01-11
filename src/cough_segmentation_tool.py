
# Copyright Alice Ashby 2022
# Special thanks to Julia Meister
# Version 0.0.1
# MIT license

# TODO
# - avoid any interpolation with sample rate
# - max normalize rmse to 0,1 before onset detection
# - avoid detection of onsets in last 2 frames
# - implement rmse threshold diagnostics function
# - implement onset offset diagnostics function
# - add docstrings to class and class methods
# - test on coughvid and coswara audio datasets

# import required libraries

import librosa                                     # librosa music package
import soundfile as sf                             # for audio exportation
import numpy as np                                 # for handling large arrays
from sklearn.preprocessing import MinMaxScaler     # for audio normalization
from tqdm.notebook import tqdm                     # for progress bars

class CoughSegmentationTool:
    ''''''

    # declare constants
    SAMPLE_RATE = 16000
    FRAME_LENGTH = 512
    HOP_LENGTH = 256
    N_FFT = 2048

    def __init__(self, sample_filenames, new_dir_path, threshold=0.5, 
                minimum_distance=20, backtrack=0.01):
        self.sample_filenames = sample_filenames
        self.new_dir_path = new_dir_path
        self.threshold = threshold
        self.minimum_distance = minimum_distance
        self.backtrack = backtrack

    def get_max_normed_sample(self, sample):
        ''''''
        
        # we want to normalize the samples not the features
        sample_scaled = sample/sample.max()

        return sample_scaled

    def get_min_max_normed_sample(self, sample):
        ''''''
        
        # we want to normalize the samples not the features
        scaler = MinMaxScaler(feature_range=(-1, 1))
        sample_scaled = scaler.fit_transform(sample.reshape(-1, 1)).flatten()
        
        return sample_scaled

    def get_sample_duration(self, sample_processed):
        ''''''
        
        # get the duration of the sample in seconds
        duration = librosa.get_duration(sample_processed, sr=self.SAMPLE_RATE)
        
        return duration

    def get_onset_frames_from_rmse(self, sample_processed, threshold, minimum_distance):
        ''''''

        # compute root mean squared energy over frames
        rmse = librosa.feature.rms(sample_processed, frame_length=self.FRAME_LENGTH, 
                                   hop_length=self.HOP_LENGTH, center=True)
        rmse_array = rmse[0]
        
        # normalize the root mean squared energy
        rmse_array_scaled = rmse_array/rmse_array.max()
        
        high_rmse_frames = list()
        for i, rmse_value in enumerate(rmse_array_scaled):
            # compare RMSE values with threshold
            if rmse_value > threshold:
                # if RMSE value exceeds threshold then
                # obtain the frame associated with that RMSE value
                high_rmse_frames.append(i)
        
        cough_frame_index = np.ones_like(high_rmse_frames, dtype=bool)
        
        # we dont want a cough to be split into multiple very small chunks
        # so check if distance between the high RMSE frames is less than 
        # or equal to x frames / our minimum distance and only select first occurrence
        
        for i in range(len(high_rmse_frames)):
            # keep first frame occurrence
            if i == 0: continue
            # check distance between adjacent frames
            j = i-1
            while not cough_frame_index[j]:
                j -= 1
            # if distance between adjacent frames is less than or equal to the minimum distance
            # then ignore these frames
            if high_rmse_frames[i] - high_rmse_frames[j] <= minimum_distance:
                cough_frame_index[i] = False
        
        # return array of onset frames with a minimum distance between them
        return np.array(high_rmse_frames)[cough_frame_index]

    def get_onset_times_backtracked(self, onset_frames, backtrack):
        ''''''

        # get the onset times from the onset frames
        onset_times = librosa.frames_to_time(onset_frames, hop_length=self.HOP_LENGTH, 
                                            sr=self.SAMPLE_RATE)
        # get the backtracked onset times
        onset_times_backtracked = list()
        for time in onset_times:
            # calculate the backtracked times
            time_backtracked = time - backtrack
            onset_times_backtracked.append(time_backtracked)
        
        return onset_times_backtracked

    def process_and_export_cough(self, sample_processed, onset_time, offset_time, new_filename):
        ''''''
        
        # get onset and offset samples from times
        onset_sample = librosa.time_to_samples(onset_time)
        offset_sample = librosa.time_to_samples(offset_time)
        
        # create sample from onset and offset
        new_sample = sample_processed[onset_sample:offset_sample]
         
        # trim leading and trailing silence under 10dB
        new_sample_trim = librosa.effects.trim(new_sample, top_db=10, frame_length=self.FRAME_LENGTH, 
                                                hop_length=self.HOP_LENGTH)[0]
        # normalize the sample
        new_sample_scaled = self.get_min_max_normed_sample(new_sample_trim)
        
        # export cough sample to audio file
        sf.write(new_filename, new_sample_scaled, self.SAMPLE_RATE)

    def run_rmse_threshold_diagnostics(self):
        ''''''

        raise NotImplementedError

    def run_onset_offset_diagnostics(self):
        ''''''

        raise NotImplementedError
    
    def run_segmentation(self):
        ''''''

        # initialize new filename i.e. ID_X where X is the cough index
        filename_split = self.new_dir_path + '/{}_{}.wav'

        for filename in tqdm(self.sample_filenames, desc='Processing samples'):
            try:
                # remove file path and filetype
                sample_name_split = filename.split('/')[-1].split('.')[0]
                sample_name = sample_name_split.split('-')[0]

                # load and process audio samples
                # [0] because sample is loaded as a tuple
                sample = librosa.load(filename, self.SAMPLE_RATE, mono=True)[0]
                sample_processed = self.get_min_max_normed_sample(sample)
                sample_duration = self.get_sample_duration(sample_processed)

                # skip samples smaller than 0.5 seconds
                if sample_duration > 0.5:
                    
                    onset_frames = self.get_onset_frames_from_rmse(sample_processed, self.threshold, 
                                                                   self.minimum_distance)
                    onset_times = self.get_onset_times_backtracked(onset_frames, self.backtrack)
                    onset_count = len(onset_times)
                    
                    # segment individual coughs
                    for i in range(onset_count):
                        try:
                            offset_time = onset_times[i+1]
                        # special case for last onset
                        except IndexError:
                            # sample duration is also sample end time
                            offset_time = sample_duration

                        # trim and normalize each individual cough, then export to audio file
                        self.process_and_export_cough(sample_processed, onset_times[i], offset_time, 
                                                      filename_split.format(sample_name, i))
                else:
                    print(f'Sample {sample_name} is too small. Skipped.')
                    continue
            except ValueError as err:
                print(f'Sample {sample_name} had an error: "{err}". Skipped.')
                continue
