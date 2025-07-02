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

import gradio as gr
from sgm.util import instantiate_from_config
from utils import enhance_prompt_one, enhance_prompt_two, enhance_prompt_effect, generate_audio_prompt_one, generate_audio_prompt_two, generate_audio_prompt_effect, label_prompt



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
model_cls=SATVideoDiffusionEngine
device = "cuda" if torch.cuda.is_available() else "cpu"
cnt_video = 0
# import pdb; pdb.set_trace()
if isinstance(model_cls, type):
    model = get_model(args, model_cls)
else:
    model = model_cls

load_checkpoint(model, args, specific_iteration=None)
model.eval()

model = model.to(device)
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
pre_label = "one_person_conversation"



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
        return final_save_path


def process_audio_emb(audio_emb):
    concatenated_tensors = []

    for i in range(audio_emb.shape[0]):
        vectors_to_concat = [
            audio_emb[max(min(i + j, audio_emb.shape[0]-1), 0)]for j in range(-2, 3)]
        concatenated_tensors.append(torch.stack(vectors_to_concat, dim=0))

    audio_emb = torch.stack(concatenated_tensors, dim=0)

    return audio_emb



def sampling_main(state, num_inference_steps, guidance_scale):
    # import pdb; pdb.set_trace()
    global audio_processor
    global model
    global pre_label
    global device
    global args
    global cnt_video
    
    original_prompt, enhanced_prompt, enhanced_prompt1, labels = state
    
    sampler_config = args.model_config.get("sampler_config", None)
    sampler_config['params']['num_steps'] = num_inference_steps
    sampler_config['params']['guider_config']['params']['num_steps'] = num_inference_steps
    sampler_config['params']['guider_config']['params']['scale'] = guidance_scale
    model.sampler = instantiate_from_config(sampler_config) if sampler_config is not None else None
    

    with torch.no_grad():
            
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
                
            model = model.to(device) 
            
            pre_label = labels

            
        if (labels == 'two_person_conversation'):
            text, vocal_audio_path, vocal_1_audio_path, accm_audio_path, music_audio_path, combine_audio_path = generate_audio_prompt_two(
                original_prompt, enhanced_prompt
            )
        elif (labels == 'no_conversation'):
            text, vocal_audio_path, accm_audio_path, music_audio_path, combine_audio_path = generate_audio_prompt_effect(
                original_prompt, enhanced_prompt
            )
        else:
            text, vocal_audio_path, accm_audio_path, music_audio_path, combine_audio_path = generate_audio_prompt_one(
                original_prompt, enhanced_prompt
            )
        
        
        
        model.conditioner = model.conditioner.to('cuda')

        name = str(cnt_video).zfill(2) + f"-seed_{args.seed}"
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


            print(f'Processing : {cnt_video}')
            cnt_video += 1
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
            final_save_path = save_video_as_grid_and_mp4_with_audio(video, save_path, combine_audio_path, fps=args.sampling_fps, name = name)
            print("saving in: ", final_save_path)
    return final_save_path




def enhance_and_show_prompt(prompt, label):
    # import pdb; pdb.set_trace()
    try:
        if (label == 'no_conversation'):
            enhanced_prompt, enhanced_prompt1 = enhance_prompt_effect(prompt)
        elif (label == 'two_person_conversation'):
            enhanced_prompt, enhanced_prompt1 = enhance_prompt_two(prompt)
        else:
            enhanced_prompt, enhanced_prompt1 = enhance_prompt_one(prompt)
        
        return enhanced_prompt1, [prompt, enhanced_prompt, enhanced_prompt1, label]  # Return enhanced_prompt1 and tuple for next step
    except Exception as e:
        return f"增强提示时出错: {str(e)}", None
def label_and_show_prompt(prompt):
    """Step 1: Classify the input prompt and display labels"""
    try:
        labels = label_prompt(prompt)
        return labels  # Return labels and pass prompt to next step
    except Exception as e:
        return f"提示分类时出错: {str(e)}", None




with gr.Blocks() as demo:
    gr.Markdown("# MTVCraft")
    
    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(
                label="Input Prompt",
                # placeholder="例如：一个年轻的亚洲女性站在柔和的阳光下，微笑着唱歌。她的面部表情和嘴部动作与旋律自然同步，眼中带有情感细微变化。",
                lines=5
            )
            num_inference_steps = gr.Slider(label="Sample Steps", minimum=20, maximum=50, step=1, value=30, interactive=True)
            guidance_scale = gr.Slider(label="CFG Guidance", minimum=1, maximum=10, step=0.1, value=6.0, interactive=True)
            enhance_button = gr.Button("Generate")
            
        with gr.Column():
            label_output = gr.Textbox(label="Label", lines=1)
            enhanced_prompt_output = gr.Textbox(label="Enhanced Prompt", lines=5)
            video_output = gr.Video(label="Generated Video")
    text_tmp = gr.State()
    enhance_button.click(
        fn=label_and_show_prompt,
        inputs=prompt_input,
        outputs=[label_output],
        queue=True
    ).then(
        fn=enhance_and_show_prompt,
        inputs=[prompt_input, label_output],
        outputs=[enhanced_prompt_output, text_tmp],
        queue=True
    ).then(
        fn=sampling_main,
        inputs=[text_tmp, num_inference_steps, guidance_scale],
        outputs=[video_output],
        concurrency_limit=1,
        queue=True
    )


demo.launch(share=True, server_port=8080)