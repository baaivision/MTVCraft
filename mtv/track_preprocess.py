import argparse
import logging
import os
from pathlib import Path
from typing import List
import pandas as pd
import torch
from tqdm import tqdm
from sgm.utils.audio_processor import AudioProcessor
from sgm.utils.util import get_fps

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
    


def process_all_videos(input_video_list: List[Path]) -> None:
    """
    Process all videos in the input list.

    Args:
        input_video_list (List[Path]): List of video paths to process.
        output_dir (Path): Directory to save the output.
        gpu_status (bool): Whether to use GPU for processing.
    """
    wav2vec_model_path = './pretrained_models/wav2vec2-base-960h'
    audio_processor = AudioProcessor(
        16000,
        wav2vec_model_path,
        False,
    )

    for video_path in tqdm(input_video_list, desc="Processing videos", unit="video"):
        try:
            # import pdb; pdb.set_trace()
            audio_path = video_path.with_name(video_path.stem + "_effect.wav")
            if (not audio_path.exists()):
                continue
            if (audio_path.with_suffix('.pt').exists()):
                continue
            fps = get_fps(video_path)
            
            audio_emb, _ = audio_processor.preprocess(audio_path, fps=fps)
            torch.save(audio_emb, str(audio_path.with_suffix('.pt')))
            torch.cuda.empty_cache()
            
            audio_path = video_path.with_name(video_path.stem + "_music.wav")
            if (not audio_path.exists()):
                continue
            if (audio_path.with_suffix('.pt').exists()):
                continue
            fps = get_fps(video_path)
            
            audio_emb, _ = audio_processor.preprocess(audio_path, fps=fps)
            torch.save(audio_emb, str(audio_path.with_suffix('.pt')))
            torch.cuda.empty_cache()
            
            audio_path = video_path.with_name(video_path.stem + "_dialog.wav")
            if (not audio_path.exists()):
                continue
            if (audio_path.with_suffix('.pt').exists()):
                continue
            fps = get_fps(video_path)
            
            audio_emb, _ = audio_processor.preprocess(audio_path, fps=fps)
            torch.save(audio_emb, str(audio_path.with_suffix('.pt')))
            torch.cuda.empty_cache()
            
        except Exception as e:
            logging.error(f"Failed to process video {video_path}: {e}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process a CSV file of video paths.")
    parser.add_argument("--input_csv", type=str, help="Path to the input CSV file")
    args = parser.parse_args()
    
    df = pd.read_csv(args.input_csv)
    video_path_list = [Path(p) for p in df['path'].tolist()]

    if not video_path_list:
        logging.warning("No videos to process.")
    else:
        process_all_videos(video_path_list)