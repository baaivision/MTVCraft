import os
import pandas as pd
import numpy as np
import torch
import argparse
import soundfile as sf
from demucs.states import load_model
from demucs.apply import apply_model
from time import time
import librosa
from tqdm import tqdm


class Demucs4_SeparationModel:

    def __init__(self, options):
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        if 'cpu' in options:
            if options['cpu']:
                device = 'cpu'
        self.device = device

        self.model_list = [
            '97d170e1-dbb4db15.th',
        ]
        if 'high_quality' in options:
            if options['high_quality']:
                print('Use 3 checkpoints!')
                self.model_list = [
                    '97d170e1-a778de4a.th',
                    '97d170e1-dbb4db15.th',
                    '97d170e1-e41a5468.th'
                ]

        self.models = []
        models_folder = os.path.dirname(os.path.abspath(__file__)) + '/models/'
        if not os.path.isdir(models_folder):
            os.mkdir(models_folder)
        for model_name in self.model_list:
            model_path = models_folder + model_name
            if not os.path.isfile(model_path):
                remote_url = 'https://github.com/ZFTurbo/MVSEP-CDX23-Cinematic-Sound-Demixing/releases/download/v.1.0.0/' + model_name
                torch.hub.download_url_to_file(remote_url, model_path)
            model = load_model(model_path)
            model.to(device)
            self.models.append(model)

        self.device = device
        pass

    @property
    def instruments(self):
        return ['dialog', 'effect', 'music']

    def raise_aicrowd_error(self, msg):
        raise NameError(msg)

    def separate_music_file(
            self,
            mixed_sound_array,
            sample_rate,
            update_percent_func=None,
            current_file_number=0,
            total_files=0,
    ):
        separated_music_arrays = {}
        output_sample_rates = {}

        audio = np.expand_dims(mixed_sound_array.T, axis=0)
        audio = torch.from_numpy(audio).type('torch.FloatTensor').to(self.device)

        all_out = []
        for model in self.models:
            out = apply_model(model, audio, shifts=1, overlap=0.8)[0].cpu().numpy()
            all_out.append(out)
        dnr_demucs = np.array(all_out).mean(axis=0)

        # dialog
        separated_music_arrays['dialog'] = dnr_demucs[2].T
        output_sample_rates['dialog'] = sample_rate

        # music
        separated_music_arrays['music'] = dnr_demucs[0].T
        output_sample_rates['music'] = sample_rate

        # effect
        separated_music_arrays['effect'] = dnr_demucs[1].T
        output_sample_rates['effect'] = sample_rate

        return separated_music_arrays, output_sample_rates


def predict_with_model(options):
    # 从 CSV 文件中读取音频文件路径
    csv_file = options['csv_file']
    if not os.path.isfile(csv_file):
        print(f"Error. No such file: {csv_file}. Please check path!")
        return
    
    # 读取 CSV 文件
    df = pd.read_csv(csv_file)
    
    # 获取路径列（假设列名为 "path"）
    input_audio_paths = df['path'].tolist()


    # 创建 Demucs4 分离模型
    model = Demucs4_SeparationModel(options)

    update_percent_func = None
    if 'update_percent_func' in options:
        update_percent_func = options['update_percent_func']

    for i, input_audio in tqdm(enumerate(input_audio_paths), total=len(input_audio_paths), desc="Processing audio files"):
        # import pdb; pdb.set_trace()
        # print(f'Processing: {input_audio}')
        input_audio = input_audio.replace('.mp4', '.wav')
        if not os.path.isfile(input_audio):
            print(f'Error. No such file: {input_audio}. Please check path!')
            continue

        output_name = os.path.splitext(os.path.basename(input_audio))[0] + '_instrum.wav'
        output_path = os.path.join(os.path.dirname(input_audio), output_name)
        if os.path.isfile(output_path):
            # print(f'Error. No such file: {input_audio}. Please check path!')
            continue
        
        audio, sr = librosa.load(input_audio, mono=False, sr=44100)
        if len(audio.shape) == 1:
            audio = np.stack([audio, audio], axis=0)
        # print(f"Input audio: {audio.shape} Sample rate: {sr}")
        
        result, sample_rates = model.separate_music_file(audio.T, sr, update_percent_func, i, len(input_audio_paths))
        
        for instrum in model.instruments:
            output_name = os.path.splitext(os.path.basename(input_audio))[0] + f'_{instrum}.wav'
            output_path = os.path.join(os.path.dirname(input_audio), output_name)
            sf.write(output_path, result[instrum], sample_rates[instrum], subtype='FLOAT')
            # print(f'File created: {output_path}')
        # print(f'File created: {output_path}')
        # 清空显存
        torch.cuda.empty_cache()

    if update_percent_func is not None:
        val = 100
        update_percent_func(int(val))


if __name__ == '__main__':
    start_time = time()

    m = argparse.ArgumentParser()
    m.add_argument("--csv_file", "-c", type=str, help="CSV file containing paths to input audio files", required=True)
    m.add_argument("--cpu", action='store_true', help="Choose CPU instead of GPU for processing. Can be very slow.")
    m.add_argument("--high_quality", action='store_true', help="Use 3 checkpoints. Will be 3 times slower.")

    options = m.parse_args().__dict__
    
    predict_with_model(options)

