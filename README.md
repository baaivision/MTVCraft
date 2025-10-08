<h1 align='center'>MTVCraft</h1>
<h2 align='center'>An Open Veo3-style Audio-Video Generation Demo</h2>

<table align='center' border="0" style="width: 100%; text-align: center; margin-top: 80px;">
  <tr>
    <td>
      <video align='center' src="https://github.com/user-attachments/assets/b63d1f73-04a6-42fc-abd2-c5ebe0e76d46" autoplay loop></video>
    </td>
  </tr>
    <tr align="center">
    <td>
      <em>For the best experience, please enable audio.</em>
    </td>
  </tr>
</table>

## News: Our paper has been accepted by NeurIPS 2025! ##

## 🎬 Pipeline

MTVCraft is a framework for generating videos with synchronized audio from a single text prompt, exploring a potential pipeline for creating general audio-visual content.

Specifically, the framework consists of a multi-stage pipeline. First, MTVCraft employs the [Qwen3](https://bailian.console.aliyun.com/?tab=model#/model-market/detail/qwen3?modelGroup=qwen3) to interpret the user's initial prompt, deconstructing it into separate descriptions for three audio categories: human speech, sound effects, and background music. Subsequently, these descriptions are fed into [ElevenLabs](https://elevenlabs.io/) to synthesize the corresponding audio tracks. Finally, these generated audio tracks serve as conditions to guide the [MTV framework](https://arxiv.org/pdf/2506.08003) in generating a video that is temporally synchronized with the sound.

Notably, both Qwen3 and ElevenLabs can be replaced by available alternatives with similar capabilities.

An online demo is available [here](https://huggingface.co/spaces/BAAI/MTVCraft).

<div align="center">
  
  ![pipeline](https://github.com/baaivision/MTVCraft/blob/main/assets/pipeline.png)
  
</div>

## ⚙️ Installation

For CUDA 12.1, you can install the dependencies with the following commands. Otherwise, you need to manually install `torch`, `torchvision` , `torchaudio` and `xformers`.

Download the codes:

```bash
git clone https://github.com/baaivision/MTVCraft
cd MTVCraft
```

Create conda environment:

```bash
conda create -n mtv python=3.10
conda activate mtv
```

Install packages with `pip`

```bash
pip install -r requirements.txt
```

Besides, ffmpeg is also needed:

```bash
apt-get install ffmpeg
```

## 📥 Download Pretrained Models

You can easily get all pretrained models required by inference from our [HuggingFace repo](https://huggingface.co/BAAI/MTVCraft).

Using `huggingface-cli` to download the models:

```shell
cd $ProjectRootDir
pip install "huggingface_hub[cli]"
huggingface-cli download BAAI/MTVCraft --local-dir ./pretrained_models
```

Or you can download them separately from their source repo:

- [mtv](https://huggingface.co/BAAI/MTVCraft/tree/main/mtv): Our checkpoints
- [t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl): text encoder, you can download from [text_encoder](https://huggingface.co/THUDM/CogVideoX-2b/tree/main/text_encoder) and [tokenizer](https://huggingface.co/THUDM/CogVideoX-2b/tree/main/tokenizer)
- [vae](https://huggingface.co/THUDM/CogVideoX1.5-5B-SAT/tree/main/vae): Cogvideox-5b pretrained 3d vae
- [wav2vec](https://huggingface.co/facebook/wav2vec2-base-960h): wav audio to vector model from [Facebook](https://huggingface.co/facebook/wav2vec2-base-960h)

Finally, these pretrained models should be organized as follows:

```text
./pretrained_models/
|-- mtv
|   |--single/
|   |   |-- 1/
|   |     |-- mp_rank_00_model_states.pt
|   |   `--latest
|   |
|   |--multi/
|   |   |-- 1/
|   |	  |-- mp_rank_00_model_states.pt
|   |   `-- latest
|   |
|   `--accm/
|       |-- 1/
|         |-- mp_rank_00_model_states.pt
|       `-- latest
|
|-- t5-v1_1-xxl/
|   |-- config.json
|   |-- model-00001-of-00002.safetensors
|   |-- model-00002-of-00002.safetensors
|   |-- model.safetensors.index.json
|   |-- special_tokens_map.json
|   |-- spiece.model
|   `-- tokenizer_config.json
|
|-- vae/
|   |--3d-vae.pt
|
`-- wav2vec2-base-960h/
    |-- config.json
    |-- feature_extractor_config.json
    |-- model.safetensors
    |-- preprocessor_config.json
    |-- special_tokens_map.json
    |-- tokenizer_config.json
    `-- vocab.json
```

## 🎮 Run Inference

#### API Setup (Required)
Before running the inference script, make sure to configure your API keys in the file `mtv/utils.py`. Edit the following section:
```python
# mtv/utils.py

qwen_model_name = "qwen-plus"  # or another model name you prefer
qwen_api_key = "YOUR_QWEN_API_KEY"  # replace with your actual Qwen API key

client = OpenAI(
    api_key=qwen_api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

elevenlabs = ElevenLabs(
    api_key="YOUR_ELEVENLABS_API_KEY",  # replace with your actual ElevenLabs API key
)
```

#### Batch

Once the API keys are set, you can run inference using the provided script:

```bash
bash scripts/inference_long.sh ./examples/samples.txt ouput_dir
```
This will read the input prompts from `./examples/samples.txt` and the results will be saved at `./output`.

#### Gradio UI
To run the Gradio UI simply run:
```bash
bash scripts/app.sh ouput_dir
```


## 📝 Citation

If you find our work useful for your research, please consider citing the paper:

```
@article{MTV,
      title={Audio-Sync Video Generation with Multi-Stream Temporal Control},
      author={Weng, Shuchen and Zheng, Haojie and Chang, Zheng and Li, Si and Shi, Boxin and Wang, Xinlong},
      journal={NeurIPS},
      year={2025}
}
```
