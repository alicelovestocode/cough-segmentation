
# Copyright Alice Ashby 2022
# Special thanks to Julia Meister
# Version 0.0.2
# MIT license

# TODO
# - add docstrings to class and class methods
# - test on coughvid, coswara, compare audio datasets
# - fine-tune parameters RMSE, min_dist, backtrack
# - use logger module for debugging?

# import required libraries

import librosa                                     # librosa music package
import soundfile as sf                             # for audio exportation
import numpy as np                                 # for handling large arrays

import librosa.display                             # librosa plot functions
import matplotlib.pyplot as plt                    # plotting with Matlab functionality

from sklearn.preprocessing import MinMaxScaler     # for audio normalization
from tqdm.notebook import tqdm                     # for progress bars

class CoughSegmentationTool:
    ''''''

    # declare constants
    SAMPLE_RATE = 16000
    FRAME_LENGTH = 512
    HOP_LENGTH = 256
    N_FFT = 2048

    def __init__(self, threshold=0.5, minimum_distance=20, backtrack=0.01, end_frames=2):
        ''''''
        self.threshold = threshold
        self.minimum_distance = minimum_distance
        self.backtrack = backtrack
        self.end_frames = end_frames

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

    def get_onset_frames_from_rmse(self, sample_processed, threshold, minimum_distance,
                                   end_frames):
        ''''''

        # compute root mean squared energy (RMSE)
        rmse = librosa.feature.rms(sample_processed, frame_length=self.FRAME_LENGTH, 
                                   hop_length=self.HOP_LENGTH, center=True)[0]
        
        # normalize the RMSE with min max normalization
        scaler = MinMaxScaler(feature_range=(0, 1))
        rmse_scaled = scaler.fit_transform(rmse.reshape(-1, 1)).flatten()
        
        high_rmse_frames = list()
        for i, rmse_value in enumerate(rmse_scaled):
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

        # do not allow onsets in the last 2 frames if there are >6 frames in total
        if len(high_rmse_frames) > 5:
            cough_frame_index[-end_frames:] = False
        
        # return array of onset frames with a minimum distance between them
        return (high_rmse_frames, np.array(high_rmse_frames)[cough_frame_index])

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
        
        return (onset_times, onset_times_backtracked)

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

    def run_rmse_normalization_diagnostics(self, sample_filenames):
        ''''''

        for filename in tqdm(sample_filenames, desc='Generating graphs'):
            try:   
                # [0] because sample is loaded as a tuple
                sample = librosa.load(filename, self.SAMPLE_RATE, mono=True)[0]

                # remove file path and filetype
                sample_name_split = filename.split('/')[-1].split('.')[0]
                sample_name = 'Sample ' + sample_name_split.split('-')[0]
                
                # compute short-time energy (STE)
                energy = np.array([ 
                    sum(abs(sample[i:i+self.FRAME_LENGTH]**2))
                    for i in range(0, len(sample), self.HOP_LENGTH)
                ])
                
                # compute root mean squared energy (RMSE)
                rmse = librosa.feature.rms(sample, frame_length=self.FRAME_LENGTH, 
                                           hop_length=self.HOP_LENGTH, center=True)[0]
                
                # normalize STE and RMSE with max normalization
                energy_max_scaled = self.get_max_normed_sample(energy)
                rmse_max_scaled = self.get_max_normed_sample(rmse)
                
                # normalize STE and RMSE with min max normalization
                energy_min_max_scaled = self.get_min_max_normed_sample(energy)
                rmse_min_max_scaled = self.get_min_max_normed_sample(rmse)

                plt.figure(figsize=(15, 5))

                # calculate onset frames from RMSE, then convert frames to onset times
                # plot RMSE with max normalization across sample waveform
                frames = range(len(rmse))
                t = librosa.frames_to_time(frames, sr=self.SAMPLE_RATE, hop_length=self.HOP_LENGTH)
                
                plt.plot(t[:len(rmse)], rmse_max_scaled, color='b')

                # calculate onset frames from STE, then convert frames to onset times
                # plot STE with max normalization across sample waveform
                frames = range(len(energy))
                t = librosa.frames_to_time(frames, sr=self.SAMPLE_RATE, hop_length=self.HOP_LENGTH)
                plt.plot(t[:len(energy)], energy_max_scaled, 'r--')

                librosa.display.waveplot(sample, sr=self.SAMPLE_RATE, alpha=0.4)
                plt.legend(('RMSE', 'STE'))
                plt.title(f'{sample_name} (max normalization)')

                plt.figure(figsize=(15, 5))

                # calculate onset frames from RMSE, then convert frames to onset times
                # plot RMSE with min max normalization across sample waveform
                frames = range(len(rmse))
                t = librosa.frames_to_time(frames, sr=self.SAMPLE_RATE, hop_length=self.HOP_LENGTH)
                plt.plot(t[:len(rmse)], rmse_min_max_scaled, color='b')

                # calculate onset frames from STE, then convert frames to onset times
                # plot STE with min max normalization across sample waveform
                frames = range(len(energy))
                t = librosa.frames_to_time(frames, sr=self.SAMPLE_RATE, hop_length=self.HOP_LENGTH)
                plt.plot(t[:len(energy)], energy_min_max_scaled, 'r--')

                librosa.display.waveplot(sample, sr=self.SAMPLE_RATE, alpha=0.4)
                plt.legend(('RMSE', 'STE'))
                plt.title(f'{sample_name} (min max normalization)')
            
            except ValueError as err:
                print(f'{sample_name} had an error: "{err}". Skipped.')
                continue

    def run_onset_offset_detection_diagnostics(self, sample_filenames, threshold=0.5, minimum_distance=20,
                                     backtrack=0.01, end_frames=2, debug=False, max_normalization=False):
        ''''''

        for filename in tqdm(sample_filenames, desc='Processing samples'):
            
            # remove file path and filetype
            sample_name_split = filename.split('/')[-1].split('.')[0]
            sample_name = 'Sample ' + sample_name_split.split('-')[0]

            # load and process audio samples
            # [0] because sample is loaded as a tuple
            sample = librosa.load(filename, self.SAMPLE_RATE, mono=True)[0]

            # select max or min max normalization
            sample_processed = self.get_max_normed_sample(sample) if max_normalization \
                else self.get_min_max_normed_sample(sample)

            # calculate cough onsets and offsets
            high_rmse_frames = self.get_onset_frames_from_rmse(sample_processed, threshold, 
                                                               minimum_distance, end_frames)[0]
            onset_frames = self.get_onset_frames_from_rmse(sample_processed, threshold, 
                                                           minimum_distance, end_frames)[1]
            onset_times = self.get_onset_times_backtracked(onset_frames, backtrack)[0]
            onset_times_backtracked = self.get_onset_times_backtracked(onset_frames, backtrack)[1]
            
            # plot the backtracked onset times
            plt.figure(figsize=(15, 5))
            librosa.display.waveplot(sample_processed, sr=self.SAMPLE_RATE, alpha=0.4)
            plt.vlines(onset_times, ymin=-1, ymax=1, color='r', alpha=0.8)
            plt.title(sample_name)
            title = f'{sample_name} (max normalization)' if max_normalization \
                else f'{sample_name} (min max normalization)'
            plt.title(title)
            
            if debug:
                print(f'{sample_name}: onset frames are {high_rmse_frames}')
                print(f'{sample_name}: selected onset frames are {onset_frames}')
                print(f'{sample_name}: onset times are {onset_times}')
                print(f'{sample_name}: backtracked onset times are {onset_times_backtracked}\n')
    
    def run_segmentation(self, sample_filenames, new_dir_path):
        ''''''

        # initialize new filename i.e. ID_X where X is the cough index
        filename_split = new_dir_path + '/{}_{}.wav'

        for filename in tqdm(sample_filenames, desc='Processing samples'):
            try:
                # remove file path and filetype
                sample_name_split = filename.split('/')[-1].split('.')[0]
                sample_name = sample_name_split.split('-')[0]

                # load and process audio samples
                # [0] because sample is loaded as a tuple
                sample = librosa.load(filename, self.SAMPLE_RATE, mono=True)[0]
                # select max or min max normalization
                sample_processed = self.get_max_normed_sample(sample) if self.max_normalization \
                    else self.get_min_max_normed_sample(sample)
                sample_duration = self.get_sample_duration(sample_processed)

                # skip samples smaller than 0.5 seconds
                if sample_duration > 0.5:
                    
                    # calculate cough onsets and offsets
                    onset_frames = self.get_onset_frames_from_rmse(sample_processed, self.threshold, 
                                                                   self.end_frames, self.minimum_distance)[1]
                    onset_times = self.get_onset_times_backtracked(onset_frames, self.backtrack)[1]
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
