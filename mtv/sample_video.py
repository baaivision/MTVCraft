import os
import math
import argparse
from typing import List, Union
from tqdm import tqdm
from omegaconf import ListConfig
import imageio
import gc
import torch
import numpy as np
from einops import rearrange


from sat.model.base_model import get_model
from sat.training.model_io import load_checkpoint
from sat import mpu

from diffusion_video import SATVideoDiffusionEngine
from arguments import get_args

import moviepy.editor as mp

from sgm.utils.audio_processor import AudioProcessor
from icecream import ic

from utils import enhance_prompt_one, enhance_prompt_two, enhance_prompt_effect, generate_audio_prompt_one, generate_audio_prompt_two, generate_audio_prompt_effect, label_prompt


def read_from_cli():
    cnt = 0
    try:
        while True:
            x = input("Please input English text (Ctrl-D quit): ")
            yield x.strip(), cnt
            cnt += 1
    except EOFError as e:
        pass


def read_from_file(p, rank=0, world_size=1):
    with open(p, "r") as fin:
        cnt = -1
        for l in fin:
            cnt += 1
            if cnt % world_size != rank:
                continue
            yield l.strip(), cnt


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def get_batch(keys, value_dict, N: Union[List, ListConfig], T=None, device="cuda"):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "txt":
            batch["txt"] = np.repeat([value_dict["prompt"]], repeats=math.prod(N)).reshape(N).tolist()
            batch_uc["txt"] = np.repeat([value_dict["negative_prompt"]], repeats=math.prod(N)).reshape(N).tolist()
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc



def save_video_as_grid_and_mp4_with_audio(video_batch: torch.Tensor, save_path: str, audio_path: str, fps: int = 5, name = None):
    os.makedirs(save_path, exist_ok=True)

    for i, vid in enumerate(video_batch):
        gif_frames = []
        for frame in vid:
            frame = rearrange(frame, "c h w -> h w c")
            frame = (255.0 * frame).cpu().numpy().astype(np.uint8)

            gif_frames.append(frame)

        now_save_path = os.path.join(save_path, f"{name}.mp4")
        with imageio.get_writer(now_save_path, fps=fps) as writer:
            for frame in gif_frames:
                writer.append_data(frame)

        video_clip = mp.VideoFileClip(now_save_path)
        audio_clip = mp.AudioFileClip(audio_path)

        if audio_clip.duration > video_clip.duration:
            audio_clip = audio_clip.subclip(0, video_clip.duration)

        video_with_audio = video_clip.set_audio(audio_clip)
 
        final_save_path = os.path.join(save_path, f"{name}_with_audio.mp4")
        video_with_audio.write_videofile(final_save_path, fps=fps)
        
        os.remove(now_save_path)

        video_clip.close()
        audio_clip.close()


def process_audio_emb(audio_emb):
    concatenated_tensors = []

    for i in range(audio_emb.shape[0]):
        vectors_to_concat = [
            audio_emb[max(min(i + j, audio_emb.shape[0]-1), 0)]for j in range(-2, 3)]
        concatenated_tensors.append(torch.stack(vectors_to_concat, dim=0))

    audio_emb = torch.stack(concatenated_tensors, dim=0)

    return audio_emb



def sampling_main(args, model_cls):

    if isinstance(model_cls, type):
        model = get_model(args, model_cls)
    else:
        model = model_cls

    step = None
    load_checkpoint(model, args, specific_iteration=step)
    model.eval()
                    
    if args.input_type == "cli":
        data_iter = read_from_cli()
    elif args.input_type == "txt":
        rank, world_size = mpu.get_data_parallel_rank(), mpu.get_data_parallel_world_size()
        print("*********************rank and world_size", rank, world_size)
        print(args.input_file)
        data_iter = read_from_file(args.input_file, rank=rank, world_size=world_size)
    else:
        raise NotImplementedError

    image_size = [480, 720]

    wav2vec_model_path = args.wav2vec_model_path

    audio_processor = AudioProcessor(
                    args.sample_rate,
                    wav2vec_model_path,
                    False,
    )

    T, H, W, C, F = args.sampling_num_frames, image_size[0], image_size[1], args.latent_channels, 8
    L = (T-1)*4 + 1


    num_samples = [1]
    force_uc_zero_embeddings = ["txt"]
    device = model.device
    ic(device)
    model = model.to("cuda")
    
    pre_label = "one_person_conversation"

    with torch.no_grad():
        for prompt, cnt in tqdm(data_iter):
            if (cnt < args.start_index):
                continue

            # import pdb; pdb.set_trace()
            
            labels = label_prompt(prompt)
            
            if (labels != pre_label):
                model=model.to('cpu')
                torch.cuda.empty_cache()
                gc.collect()
                del model
                
                
                if (labels == "no_conversation"):
                    args.base[0] = './configs/cogvideox_5b.yaml'
                    args.base[1] = './configs/inference_accm.yaml'
                    args.load = './pretrained_models/mtv/accm'
                    args.model_config['network_config']['params']['transformer_args']['is_multi_person'] = False
                    if isinstance(model_cls, type):
                        model = get_model(args, model_cls)
                    else:
                        model = model_cls

                    step = None
                    load_checkpoint(model, args, specific_iteration=step)
                    model.eval()
                elif (labels == "two_person_conversation"):
                    args.base[0] = './configs/cogvideox_5b_multi.yaml'
                    args.base[1] = './configs/inference_multi.yaml'
                    args.load = './pretrained_models/mtv/multi'
                    args.model_config['network_config']['params']['transformer_args']['is_multi_person'] = True
                    if isinstance(model_cls, type):
                        model = get_model(args, model_cls)
                    else:
                        model = model_cls

                    step = None
                    load_checkpoint(model, args, specific_iteration=step)
                    model.eval()
                    
                else:
                    args.base[0] = './configs/cogvideox_5b.yaml'
                    args.base[1] = './configs/inference.yaml'
                    args.load = './pretrained_models/mtv/single'
                    args.model_config['network_config']['params']['transformer_args']['is_multi_person'] = False
                    if isinstance(model_cls, type):
                        model = get_model(args, model_cls)
                    else:
                        model = model_cls

                    step = None
                    load_checkpoint(model, args, specific_iteration=step)
                    model.eval()
                    
                model = model.to("cuda") 
                device = model.device
                ic(device)
                
                pre_label = labels

            
            if (labels == 'no_conversation'):
                enhanced_prompt, enhanced_prompt1 = enhance_prompt_effect(prompt)
            elif (labels == 'two_person_conversation'):
                enhanced_prompt, enhanced_prompt1 = enhance_prompt_two(prompt)
            else:
                enhanced_prompt, enhanced_prompt1 = enhance_prompt_one(prompt)
                
            if (labels == 'two_person_conversation'):
                text, vocal_audio_path, vocal_1_audio_path, accm_audio_path, music_audio_path, combine_audio_path = generate_audio_prompt_two(
                    prompt, enhanced_prompt
                )
            elif (labels == 'no_conversation'):
                text, vocal_audio_path, accm_audio_path, music_audio_path, combine_audio_path = generate_audio_prompt_effect(
                    prompt, enhanced_prompt
                )
            else:
                text, vocal_audio_path, accm_audio_path, music_audio_path, combine_audio_path = generate_audio_prompt_one(
                    prompt, enhanced_prompt
                )
            
            
            
            model.conditioner = model.conditioner.to('cuda')

            name = str(cnt).zfill(2) + f"-seed_{args.seed}"
            save_path = args.output_dir
            os.makedirs(save_path, exist_ok=True)

            vocal_audio_emb, vocal_length = audio_processor.preprocess(vocal_audio_path, L, fps = 24)
            vocal_audio_emb = process_audio_emb(vocal_audio_emb)
            
            if (labels == 'two_person_conversation'):
                vocal_audio_emb_1, vocal_length_1 = audio_processor.preprocess(vocal_1_audio_path, L, fps = 24)
                vocal_audio_emb_1 = process_audio_emb(vocal_audio_emb_1)

            accm_audio_emb, accm_length = audio_processor.preprocess(accm_audio_path, L, fps = 24)
            accm_audio_emb = process_audio_emb(accm_audio_emb)
            
            music_audio_emb, music_length = audio_processor.preprocess(music_audio_path, L, fps = 24)
            music_audio_emb = process_audio_emb(music_audio_emb)

            model.first_stage_model = model.first_stage_model.to('cpu')
            torch.cuda.empty_cache()


            pad_shape = (args.batch_size, T, C, H // F, W // F)
            mask_image = torch.zeros(pad_shape).to(model.device).to(torch.bfloat16)

            value_dict = {
                "prompt": text,
                "negative_prompt": "",
                "num_frames": torch.tensor(T).unsqueeze(0),
            }

            batch, batch_uc = get_batch(
                get_unique_embedder_keys_from_conditioner(model.conditioner), value_dict, num_samples
            )
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    print(key, batch[key].shape)
                elif isinstance(batch[key], list):
                    print(key, [len(l) for l in batch[key]])
                else:
                    print(key, batch[key])
            with torch.no_grad():
                c, uc = model.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=force_uc_zero_embeddings,
                )

            for k in c:
                if not k == "crossattn":
                    c[k], uc[k] = map(lambda y: y[k][: math.prod(num_samples)].to("cuda"), (c, uc))

            times = max((vocal_audio_emb.shape[0] - 8) // (L-5), 1)

            video = []

            first_latent = None
            model.conditioner = model.conditioner.to('cpu')
            torch.cuda.empty_cache()
            for t in range(times):
                print(f"[{t+1}/{times}]")

                if args.image2video and mask_image is not None:
                    c["concat"] = mask_image
                    uc["concat"] = mask_image
                assert args.batch_size == 1
                if (t == 0):
                    vocal_audio_tensor = vocal_audio_emb[
                         : L
                    ]
                    if (labels == 'two_person_conversation'):
                        vocal_audio_tensor_1 = vocal_audio_emb_1[
                            : L
                        ]
                    accm_audio_tensor = accm_audio_emb[
                         : L
                    ]
                    music_audio_tensor = music_audio_emb[
                        : L
                    ]
                else:
                    vocal_audio_tensor = vocal_audio_emb[
                        t * (L - 5) : t * (L - 5) + L
                    ]
                    if (labels == 'two_person_conversation'):
                        vocal_audio_tensor_1 = vocal_audio_emb_1[
                            t * (L - 5) : t * (L - 5) + L
                        ]
                    accm_audio_tensor = accm_audio_emb[
                        t * (L - 5) : t * (L - 5) + L
                    ]
                    music_audio_tensor = music_audio_emb[
                        t * (L - 5) : t * (L - 5) + L
                    ]

                pre_fix = torch.zeros_like(vocal_audio_emb)
                if vocal_audio_tensor.shape[0]!=L:
                    pad = L - vocal_audio_tensor.shape[0]
                    assert pad > 0
                    padding = pre_fix[-1:].repeat(pad, *([1] * (pre_fix.dim() - 1)))
                    vocal_audio_tensor = torch.cat([vocal_audio_tensor, padding], dim=0)
                
                vocal_audio_tensor = vocal_audio_tensor.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
                if (labels == 'two_person_conversation'):
                    if vocal_audio_tensor_1.shape[0]!=L:
                        pad = L - vocal_audio_tensor_1.shape[0]
                        assert pad > 0
                        padding = pre_fix[-1:].repeat(pad, *([1] * (pre_fix.dim() - 1)))
                        vocal_audio_tensor_1 = torch.cat([vocal_audio_tensor_1, padding], dim=0)
                    
                    vocal_audio_tensor_1 = vocal_audio_tensor_1.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
                    
                if accm_audio_tensor.shape[0]!=L:
                    pad = L - accm_audio_tensor.shape[0]
                    assert pad > 0
                    padding = pre_fix[-1:].repeat(pad, *([1] * (pre_fix.dim() - 1)))
                    accm_audio_tensor = torch.cat([accm_audio_tensor, padding], dim=0)
                
                accm_audio_tensor = accm_audio_tensor.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
                
                if music_audio_tensor.shape[0]!=L:
                    pad = L - music_audio_tensor.shape[0]
                    assert pad > 0
                    padding = pre_fix[-1:].repeat(pad, *([1] * (pre_fix.dim() - 1)))
                    music_audio_tensor = torch.cat([music_audio_tensor, padding], dim=0)
                    
                music_audio_tensor = music_audio_tensor.unsqueeze(0).to(device=device, dtype=torch.bfloat16)


                print(f'Processing : {cnt}')

                latent_frame_mask = torch.ones(T, dtype=torch.bool)
                if (t > 0):
                    latent_frame_mask[:2] = 0
                latent_frame_mask = latent_frame_mask.unsqueeze(0).to(device=device)
                if (labels != 'two_person_conversation'):
                    vocal_audio_tensor_1 = None
                    
                samples_z = model.sample(
                    c,
                    uc=uc,
                    batch_size=1,
                    shape=(T, C, H // F, W // F),
                    audio_emb_vocal = vocal_audio_tensor,
                    audio_emb_vocal_1 = vocal_audio_tensor_1,
                    audio_emb_accm = accm_audio_tensor,
                    audio_emb_music = music_audio_tensor,
                    latent_frame_mask = latent_frame_mask,
                    first_latent = first_latent,
                )

                samples_z = samples_z.permute(0, 2, 1, 3, 4).contiguous()
                torch.cuda.empty_cache()
                latent = 1.0 / model.scale_factor * samples_z
                

                # Decode latent serial to save GPU memory
                recons = []
                loop_num = (T - 1) // 2
                model.conditioner = model.conditioner.to('cpu')
                model.first_stage_model = model.first_stage_model.to('cuda')
                torch.cuda.empty_cache()
                for i in range(loop_num):
                    if i == 0:
                        start_frame, end_frame = 0, 3
                    else:
                        start_frame, end_frame = i * 2 + 1, i * 2 + 3
                    if i == loop_num - 1:
                        clear_fake_cp_cache = True
                    else:
                        clear_fake_cp_cache = False
                    # model.conditioner
                    
                    torch.cuda.empty_cache()
                    with torch.no_grad():
                        recon = model.first_stage_model.decode(
                            latent[:, :, start_frame:end_frame].contiguous(), clear_fake_cp_cache=clear_fake_cp_cache
                        )



                    recons.append(recon)
                recon = torch.cat(recons, dim=2).to(torch.float32)
                samples_x = recon.permute(0, 2, 1, 3, 4).contiguous()
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0).cpu()
                
                last_5_img = (samples[:, -5:] * 2.0) - 1.0
                last_5_img = last_5_img.permute(0,2,1,3,4).cuda().to(torch.bfloat16).contiguous()
                first_latent = model.encode_first_stage(last_5_img, None).permute(0,2,1,3,4)
                
                if (t == 0):
                    video.append(samples)
                else:
                    video.append(samples[:, 5 : ])
                
            torch.cuda.empty_cache()
            video = torch.cat(video, dim=1)
            video = video[:, :vocal_length]
            
            if mpu.get_model_parallel_rank() == 0:
                save_video_as_grid_and_mp4_with_audio(video, save_path, combine_audio_path, fps=args.sampling_fps, name = name)
                print("saving in: ", save_path)


if __name__ == "__main__":
    if "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
        os.environ["LOCAL_RANK"] = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]
        os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
        os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
    py_parser = argparse.ArgumentParser(add_help=False)
    known, args_list = py_parser.parse_known_args()

    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    del args.deepspeed_config
    args.model_config.first_stage_config.params.cp_size = 1
    args.model_config.network_config.params.transformer_args.model_parallel_size = 1
    args.model_config.network_config.params.transformer_args.checkpoint_activations = False
    args.model_config.loss_fn_config.params.sigma_sampler_config.params.uniform_sampling = False
    sampling_main(args, model_cls=SATVideoDiffusionEngine)
