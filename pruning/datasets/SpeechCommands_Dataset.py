import os
import sys
import librosa
import warnings
import pathlib
import json
import subprocess
warnings.filterwarnings("ignore", category=FutureWarning)

from torch.utils.data.dataset import Dataset
from typing import Callable, Dict, Iterable, List, NamedTuple, Optional
from torch.utils.data.dataset import Dataset
from .input_data import *

def prepare_model_settings(label_count, sample_rate, clip_duration_ms,
                           window_size_ms, window_stride_ms,
                           dct_coefficient_count):
    """Calculates common settings needed for all models.

    Args:
    label_count: How many classes are to be recognized.
    sample_rate: Number of audio samples per second.
    clip_duration_ms: Length of each audio clip to be analyzed.
    window_size_ms: Duration of frequency analysis window.
    window_stride_ms: How far to move in time between frequency windows.
    dct_coefficient_count: Number of frequency bins to use for analysis.

    Returns:
    Dictionary containing common settings.
    """
    desired_samples = int(sample_rate * clip_duration_ms / 1000)
    window_size_samples = int(sample_rate * window_size_ms / 1000)
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)
    length_minus_window = (desired_samples - window_size_samples)
    if length_minus_window < 0:
        spectrogram_length = 0
    else:
        spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
        fingerprint_size = dct_coefficient_count * spectrogram_length
    return {
        'desired_samples': desired_samples,
        'window_size_samples': window_size_samples,
        'window_stride_samples': window_stride_samples,
        'spectrogram_length': spectrogram_length,
        'dct_coefficient_count': dct_coefficient_count,
        'fingerprint_size': fingerprint_size,
        'label_count': label_count,
        'isample_rate': sample_rate,
    }

CLASSES = 'yes,no,up,down,left,right,on,off,stop,go'.split(',')
GOOGLE_SPEECH_SAMPLE_RATE = 16000

class SpeechCommandsDataset(Dataset):
    def __init__(self, arch, split, transform = None, sample_rate=16000, clip_duration_ms=1000, window_size_ms=40, window_stride_ms=40, dct_coefficient_count=10, silence_percentage=10, unknown_percentage=10, validation_percentage=10, testing_percentage=10, background_volume_range=0.1, background_frequency=0.8, time_shift_ms=100):

        classes = prepare_words_list(CLASSES)
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        subprocess.call('mkdir -p ' + ROOT_DIR + '/../datasets/', shell=True)
        subprocess.call('mkdir -p ' + ROOT_DIR + '/../datasets/KWS_CNN_S/', shell=True)
        path = ROOT_DIR + '/../datasets/KWS_CNN_S/' + split
        #subprocess.call('mkdir -p ~/.cache/KWS_CNN_S/', shell=True)
        #path = '~/.cache/KWS_CNN_S/' + split
        if os.path.isfile(path + "_data.json") and os.path.isfile(path + "_label.json"):
            print("Load cached data ...")
            with open(path + "_data.json", "r") as fd:
                data = np.array(json.load(fd))
            with open(path + "_label.json", "r") as fl:
                label = np.array(json.load(fl))
        else:
            model_settings = prepare_model_settings(
                label_count=len(classes), \
                sample_rate=sample_rate, \
                clip_duration_ms=clip_duration_ms, \
                window_size_ms=window_size_ms, \
                window_stride_ms=window_stride_ms, \
                dct_coefficient_count=dct_coefficient_count)

            DOWNLOAD_URL = 'http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
            subprocess.call('mkdir -p ~/.cache/speech_commands_v2', shell=True)
            FOLDER = pathlib.Path('~/.cache/speech_commands_v2').expanduser()

            audio_processor = AudioProcessor(
                data_url=DOWNLOAD_URL, \
                data_dir=FOLDER, \
                silence_percentage=silence_percentage, \
                unknown_percentage=unknown_percentage, \
                wanted_words=CLASSES, \
                validation_percentage=validation_percentage, \
                testing_percentage=testing_percentage, \
                model_settings=model_settings)

            sess = tf.compat.v1.InteractiveSession()

            if split == 'test' or split == 'validation':
                time_shift = 0
            else:
                time_shift = int((time_shift_ms * sample_rate) / 1000)

            data, label = audio_processor.get_data(
                how_many=-1, \
                offset=0, \
                model_settings=model_settings, \
                background_frequency=background_frequency, \
                background_volume_range=background_volume_range, \
                time_shift=time_shift, \
                mode=split, \
                sess=sess)
            sess.close()
            if arch == 'KWS_CNN_S':
                data = np.reshape(data, (-1, 1, 49, 10))
                data = data.tolist()
            with open(path + "_data.json", "w") as fd:
                json.dump(data, fd, indent=2)
            with open(path + "_label.json", "w") as fl:
                json.dump(label.tolist(), fl, indent=2)
            data = np.array(data)

        self.classes = classes
        self.data = data
        self.transform = transform
        self.label = label
        return

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)
