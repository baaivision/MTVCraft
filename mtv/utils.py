#pip install openai elevenlabs python-dotenv pudub demucs soundfile librosa gradio

import os
import sys
from openai import OpenAI
from elevenlabs.client import ElevenLabs
from elevenlabs import play
from elevenlabs import save
from elevenlabs import VoiceSettings
import json
import base64
import requests
from pydub import AudioSegment
from pydub.playback import play
from pydub.effects import speedup
import subprocess
import re
import glob
from datetime import datetime
import threading


def normalize_effect(audio):
    change_in_dBFS = -15.0 - audio.max_dBFS
    return audio.apply_gain(change_in_dBFS)

def normalize_music(audio):
    change_in_dBFS = -20.0 - audio.max_dBFS
    return audio.apply_gain(change_in_dBFS)

def normalize_speech(audio):
    change_in_dBFS = -10.0 - audio.max_dBFS
    return audio.apply_gain(change_in_dBFS)

def extract_times_from_filename(filename):
    base_name, _ = os.path.splitext(filename)
    parts = base_name.split('_')

    if len(parts) >= 3:
        try:
            start_time = float(parts[-2]) # 倒数第二个就是起始时间
            end_time = float(parts[-1])   # 最后一个就是结束时间
            return start_time, end_time
        except ValueError:
            print(f"警告: 无法从文件名 '{filename}' 中解析浮点数。")
            return None, None
    else:
        print(f"警告: 文件名 '{filename}' 格式不符合预期。")
        return None, None


LLM_lock = threading.Lock()

elevenlab_lock = threading.Lock()

qwen_model_name = "qwen-plus"  # 你可以根据需要更改为其他模型名称
qwen_api_key = "xxxxxx"

client = OpenAI(
    api_key=qwen_api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

elevenlabs = ElevenLabs(
    api_key="xxxx",
)

def label_prompt(input_prompt):

    input='''Classify the following input as either one_person_conversation or two_person_conversation or no_conversation.
        Here is some examples:
        Input1:
        At a red carpet event, a man in a black suit and a woman in a shiny red dress are having a conversation, both holding microphones.
        Output1:
        two_person_conversation

        Input2:
        In a cozy, book-filled room, a woman sits comfortably, speaking with expressive hand gestures. Warm light filters through a window behind her, casting a soft glow on the shelves. The atmosphere is relaxed and intimate, focusing on her and the conversation.
        Output2:
        one_person_conversation
        
        Input3:
        A close-up shot of a clear glass with a textured surface.
        Output3:
        no_conversation

        Now here's a new input text prompt, please output only the most appropriate label with no additional text:
    '''
    with LLM_lock:
        completion = client.chat.completions.create(
            model=qwen_model_name,
            messages=[
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": input+input_prompt},
                    ],
                },
            ],
        )
        label = completion.choices[0].message.content
    return label


def enhance_prompt_effect(input_prompt):

    input='''Please expand and enhance the following prompt to make it more descriptive, expressive, and detailed while preserving its original intent. The goal is to improve its effectiveness as an input for a language model by adding relevant context, clarifying the task, and enriching the language.
        Here are some good examples:
        1.No conversation. Scene Description: In a dimly lit, narrow corridor flanked by large, cylindrical storage drums, a figure slowly emerges from the shadows. The drums are stacked in an orderly fashion on either side of the corridor, and some bear hazardous material labels. The atmosphere is eerie and suspenseful, with a single light source hanging from the ceiling, casting a faint glow that barely illuminates the scene. As the figure moves forward, it becomes clear that it is a person wearing a hooded garment, which obscures their face and adds to the mysterious and possibly ominous nature of the setting. The person's pace is deliberate and cautious, suggesting a sense of trepidation or the need for stealth. The camera follows the figure from behind at a steady pace, maintaining a consistent distance and keeping the person centered in the frame as they navigate the confined space. The sequence conveys a narrative of tension and uncertainty, leaving the viewer curious about the person's intentions and the context of the setting.
        2.No conversation. Scene Description: A woman in traditional clothing, including a headscarf and a long dress, is engaged in an activity outdoors. She is focused on handling a piece of cloth or garment, which she appears to be washing or rinsing. The setting includes a simple, unadorned wall and a ground that looks dry and dusty, suggesting an arid or semi-arid environment. The woman's actions are deliberate as she manipulates the fabric, possibly wringing it out or shaking it to remove water. Her facial expressions and body language convey a sense of concentration and perhaps routine, as she performs this task. The sequence of shots suggests a continuous motion, with the woman moving slightly as she works with the cloth.

        Here is an example of how to generate a prompt from a user input:
        Input:
        Water is poured into a textured glass, showing clear ripples and bubbles, with a blurred background.
        Output:
        No conversation. Scene Description: A close-up shot of a clear glass with a textured surface, placed on a wooden surface. The glass is partially filled with water, and the focus is on the water's movement as it is poured into the glass from above. The pouring action creates ripples and bubbles in the water, which are captured in detail. The background is blurred, emphasizing the glass and the water.

        Now here's a new input text prompt, please generate detailed prompt in similar format, and do not output any other words:
    '''
    with LLM_lock:
        completion = client.chat.completions.create(
            model=qwen_model_name,
            messages=[
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": input+input_prompt},
                    ],
                },
            ],
        )
        enhance_prompt = completion.choices[0].message.content
    
    return enhance_prompt, enhance_prompt


def enhance_prompt_one(input_prompt=None):
    
    input='''If the input contains direct speech or quoted dialogue (i.e., text enclosed in quotation marks), omit the quoted content. Instead, summarize or reinterpret the speech content in a concise and context-appropriate manner, while retaining the speaking action or attribution verb. Do not alter any other part of the input.
        Here are some examples:
        Input:
        an angry and mad man  speaks "dont prompt me anymore because  I know I am not human"
        Output:
        an angry and mad man speaks in frustration about being aware of his non-human identity.
        
        Input:
        A confident young woman stands on a brightly lit TED-style stage, speaking "everybody, good night"
        Output:
        A confident young woman stands on a brightly lit TED-style stage, speaking a closing remark to the audience.
        
        Now here's a new input text prompt, please generate the result in similar format, and do not output any other words:
    '''
    with LLM_lock:
        completion = client.chat.completions.create(
            model=qwen_model_name,
            messages=[
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": input+input_prompt},
                    ],
                },
            ],
        )
        input_prompt = completion.choices[0].message.content
    
    input='''Please help me transfer a user input to a prompt for video generation. The input is a text prompt to describe a video with sound, and the duration of the video cannot be more than 5 seconds. You need to analyze the prompt and generate detailed prompts so that they can be used to generate the video. 
        Here are some good examples:
        1.One person conversation. Persons: Person1: a woman on the left side, wearing a patterned blouse with a geometric design, and she has short black hair. Person1 is speaking. Scene Description: The video captures a scene where Person1, a woman with short black hair, is engaged in a conversation with another individual whose back is turned to the camera. They stand outdoors, likely in a parking lot, given the blurred cars visible in the background. The atmosphere suggests a casual yet focused interaction, with Person1 appearing attentive and expressive as they speak. The lighting indicates it might be late afternoon or early evening, casting a soft glow that enhances the natural ambiance. The background activity, including a person walking by, adds a subtle layer of movement without detracting from the central focus on Person1 and their dialogue.
        2.One person conversation. Persons: Person1: a man wearing glasses and a brown suit jacket over a blue shirt, positioned centrally in the frame, gesturing with his hands while speaking. Person1 is speaking. Scene Description: In a warmly lit room, Person1 stands centrally, engaging in a conversation that seems both earnest and animated. The setting exudes a cozy ambiance, with a lamp casting a soft glow and bookshelves filled with books forming the backdrop. Person1 gestures expressively with their hands, emphasizing points as they speak, suggesting a discussion of some importance or interest. The atmosphere is intimate, hinting at a personal or intellectual exchange, perhaps a reflection or a debate. The overall mood is one of focused engagement, with Person1 appearing deeply involved in the dialogue.
        3.One person conversation. Persons: Person1: a man with short dark hair and a beard, wearing a maroon polo shirt, positioned centrally in the frame. Person1 is speaking. Scene Description: The video captures Person1 standing indoors, likely in a hallway or corridor, with a door visible behind them. The lighting is soft and natural, suggesting daytime. Person1 appears to be engaged in a conversation, as indicated by their changing facial expressions—smiling, nodding, and occasionally looking slightly to the side, possibly addressing someone off-camera. The background is minimalistic, with neutral-colored walls and a simple door, which keeps the focus squarely on Person1. The overall atmosphere feels casual and intimate, hinting at a personal interaction rather than a public setting
        4.One person conversation. Persons: Person1: a man sitting in the center, wearing a dark suit, a red knitted scarf, and a green tie, with a serious expression. Person1 is speaking. Scene Description: The video features Person1 seated in a richly carved wooden chair within what appears to be a grand, possibly historical, interior. The ornate details of the chair and the backdrop suggest a setting of significance, perhaps a study or a formal room. Person1 is dressed in a dark suit, complemented by a vibrant red knitted scarf and a green tie, adding a touch of color to the otherwise muted tones of the surroundings. The lighting is soft and warm, casting gentle shadows that enhance the texture of the wood and the intricate carvings behind Person1. The atmosphere is one of quiet contemplation, with Person1 maintaining a steady posture, occasionally shifting slightly but mostly remaining composed and engaged in whatever conversation or monologue is taking place. The overall ambiance is one of formality and introspection, suggesting a moment of reflection or discussion in a setting of considerable depth and history.

        Here is an example of how to generate a prompt from a user input:
        Input:
        In a cozy, book-filled room, a woman sits comfortably, speaking with expressive hand gestures. Warm light filters through a window behind her, casting a soft glow on the shelves. The atmosphere is relaxed and intimate, focusing on her and the conversation.
        Output:
        One person conversation. Persons: Person1: a woman on the right side, wearing a dark blue shirt and a necklace, with long dark hair. Person1 is speaking. Scene Description: In a cozy room filled with shelves brimming with books, Person1 sits comfortably, engaged in a conversation. The natural light streaming through the window behind them casts a warm glow across the scene, highlighting the array of books that line the shelves. Person1 gestures occasionally with their hands, indicating an animated discussion. The atmosphere is relaxed and intimate, suggesting a casual yet meaningful exchange, perhaps a reflection on a personal story or a shared interest. The background, though slightly blurred, emphasizes the quiet charm of the room, where the focus remains firmly on Person1 and their dialogue.

        Now here's a new input text prompt, please generate detailed prompt in similar format, and do not output any other words:
    '''
    with LLM_lock:
        completion = client.chat.completions.create(
            model=qwen_model_name,
            messages=[
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": input+input_prompt},
                    ],
                },
            ],
        )
        enhance_prompt = completion.choices[0].message.content
    
    input='''If the input contains direct speech or quoted dialogue (i.e., text enclosed in quotation marks), remove only the quoted content, including the quotation marks. Do not change anything else in the input:
        Here is an example:
        Input:
        One person conversation. Persons: Person1: a man on the left side, wearing a dark gray jacket over a white shirt, with short brown hair and a stern expression. Person1 is speaking. Scene Description: The video captures Person1 standing in a slightly dimly lit room, his posture tense as he speaks with clear agitation. His facial expression conveys frustration, eyebrows furrowed, and his tone matches the intensity of his words: "Don't prompt me any more." The background includes a plain wall with minimal distractions, keeping the focus squarely on Person1's emotions and delivery. A faint shadow falls across his face from a single overhead light source, enhancing the dramatic effect of his irritation. Despite the brevity of the statement, the scene effectively communicates his annoyance, leaving no doubt about the strength of his feelings.
        Output:
        One person conversation. Persons: Person1: a man on the left side, wearing a gray shirt and dark pants, with short brown hair. Person1 is speaking. Scene Description: The video captures Person1 standing in a slightly dimly lit room, his posture tense and expressive of anger. His facial features are sharp, eyebrows furrowed, and his arms move emphatically as he speaks, conveying frustration. The background is minimalistic, with blurred neutral tones that keep the focus on Person1. The lighting highlights his emotions, casting shadows that enhance the intensity of the moment. The atmosphere is charged, suggesting a confrontation or strong reaction, but remains contained within the brief duration of the video. The emphasis is on Person1's animated gestures and the forceful delivery of his words, without including their specific content.
        
        Not rephrase, summarize, or reformat any other parts of the input. Only omit the direct speech or quoted dialogue.
        Now here's a new input text prompt, please generate the result in similar format, and do not output any other words:
    '''
    with LLM_lock:
        completion = client.chat.completions.create(
            model=qwen_model_name,
            messages=[
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": input+enhance_prompt},
                    ],
                },
            ],
        )
    return enhance_prompt, completion.choices[0].message.content




def enhance_prompt_two(input_prompt=None):

    input='''If the input contains direct speech or quoted dialogue (i.e., text enclosed in quotation marks), omit the quoted content. Instead, summarize or reinterpret the speech content in a concise and context-appropriate manner, while retaining the speaking action or attribution verb. Do not alter any other part of the input.
        Here are some examples:
        Input:
        an angry and mad man  speaks "dont prompt me anymore because  I know I am not human"
        Output:
        an angry and mad man speaks in frustration about being aware of his non-human identity.
        
        Input:
        A confident young woman stands on a brightly lit TED-style stage, speaking "everybody, good night"
        Output:
        A confident young woman stands on a brightly lit TED-style stage, speaking a closing remark to the audience.
        
        Now here's a new input text prompt, please generate the result in similar format, and do not output any other words:
    '''
    with LLM_lock:
        completion = client.chat.completions.create(
            model=qwen_model_name,
            messages=[
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": input+input_prompt},
                    ],
                },
            ],
        )
        input_prompt = completion.choices[0].message.content
        
    input='''Please help me transfer a user input to a prompt for two person conversation video generation. The input is a text prompt to describe a video with sound, and the duration of the video cannot be more than 5 seconds. You need to analyze the prompt and generate detailed prompts so that they can be used to generate the video. 
        Here are some good examples:
        1.Two person conversation. Persons: Person1: a white woman on the left wearing a gray and white plaid coat, jeans, and black shoes. Person2: a woman on the right side of the video, with dark brown hair, wearing a blue jacket and dark pants. Person1 is speaking. Scene Description: Person1 and Person2 walk side-by-side down a paved pathway, their bodies oriented forward in a deliberate, forward motion. The setting appears to be the heart of a town square. In the background, a mix of pedestrians meander along the pathway, some passing by the side of Person1 and Person2. A few people are seated at tables and chairs set up on the paved surface. Person1 gesticulates with their hands, in the midst of explaining an idea to Person2. Person2's head is turned towards Person1 and Person2 gestures with their hands, a neutral expression on their face. To the left, is a sign that partially obscures the details of the establishment. To the right, flowering plants add a touch of natural color to the scene.
        2.Two person conversation. Persons: Person1: a fair-skinned, blonde woman on the right, wearing a colorful, floral blouse, a pearl necklace, and with her hair in a ponytail. Person2: a woman on the left with long blonde hair, wearing a blue and green checkered sweater. Person1 is speaking. Scene Description: In a suburban setting, Person1 and Person2 are engaged in a conversation outside a house with a white garage door visible behind them. A third person is walking away in the background, slightly out of focus. Person1, with her hands together, is looking directly at Person2 as she speaks. Person2 is facing Person1, gesturing with her hands, and her expression suggests attentiveness and understanding. The atmosphere seems casual and friendly, suggesting a brief but meaningful interaction before parting ways.
        3.Two person conversation. Persons: Person1: a male to the right, with a bald head, a beard, and wearing a patterned shirt. Person2: a white male on the left side of the video with grey hair, wearing a blue jacket over a patterned shirt. Person2 is speaking. Scene Description: In a dimly lit room, Person1 and Person2 are seated in front of a dark wooden table. A lamp emits a soft glow in the background, illuminating a collection of objects. The atmosphere is intense as Person2 directs accusations toward Person1, leaning forward with a troubled expression and sharp gestures. Person1, seemingly taken aback by the accusations, looks back with a mixture of disbelief and defensiveness, their body language shifting between denial and contemplation. The conversation seems to escalate, and a palpable sense of tension fills the space, indicating a long-standing conflict reaching a critical point.
        4.Two person conversation. Persons: Person1: a woman to the right, wearing a striped shirt. Person2: a female, located on the left, with dark hair in a bun, wearing a light-colored shirt. Person2 is speaking. Scene Description: The video captures Person1 and Person2 seated at a table in what appears to be a dining room. A painting hangs on the wall behind them, and a small, bright light shines between them, illuminating the table and their faces. Person2 is engaged in speaking, using hand gestures as they talk. Person1 is mostly still, listening attentively. Scattered papers and other items lay on the table between them, adding to the casual, everyday feel of the scene. A decorative chair is partially visible to the left, providing a sense of domestic comfort. The room is dimly lit, accentuating the bright light on the table, and casting soft shadows around the space.

        Here is two example of how to generate a prompt from a user input:
        Input1:
        At a red carpet event, a man in a black suit and a woman in a shiny red dress are having a conversation, both holding microphones.
        Output1:
        Two person conversation. Persons: Person1: a male on the left with glasses, wearing a black suit and holding a microphone. Person2: a woman on the right, wearing a shiny red dress, with dark hair pulled back, and holding a microphone. Person1 is speaking. Scene Description: Against a backdrop of other attendees and greenery, Person1 stands to the left, holding a microphone with the other hand keeping static. His mouth is open as he addresses Person2. Person2, positioned to the right, also holds a same microphone. She has a slight smile as she turns towards Person1, engaging with his question. The environment suggests a red-carpet event, with lighting fixtures visible overhead and a structure behind the two individuals.

        Input2:
        In a quiet garden with flowers and trees, a woman in a white blouse talks to a man in a white sweater.
        Output2:
        Two person conversation. Persons: Person1: a woman on the left, wearing a white blouse with brown hair. Person2: a male on the right wearing a white sweater with black buttons. Person2 is speaking. Scene Description: In a lush garden setting, Person1 and Person2 stand together, engaged in conversation. A backdrop of vibrant pink flowers, citrus trees, and green foliage creates a tranquil atmosphere. Person2 appears to be reassuring or comforting Person1, who seems slightly worried or anxious. The setting suggests a private moment, possibly in a backyard or garden, away from the hustle and bustle of everyday life. Their gestures and expressions suggest a close relationship, possibly romantic or familial, as they navigate a situation that requires reassurance and understanding.

        Remember that If the input includes direct speech or quoted dialogue (e.g., “Breakfast is ready”), include the quoted sentence in the output and according it to determine the who is speaking.
        Now here's a new input text prompt, please generate detailed prompt in similar format, and do not output any other words:
    '''
    with LLM_lock:
        completion = client.chat.completions.create(
            model=qwen_model_name,
            messages=[
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": input+input_prompt},
                    ],
                },
            ],
        )
        enhance_prompt = completion.choices[0].message.content
    
    input='''If the input contains direct speech or quoted dialogue (i.e., text enclosed in quotation marks), remove only the quoted content, including the quotation marks. Do not change anything else in the input:
        Here is an example:
        Input:
        One person conversation. Persons: Person1: a man on the left side, wearing a dark gray jacket over a white shirt, with short brown hair and a stern expression. Person1 is speaking. Scene Description: The video captures Person1 standing in a slightly dimly lit room, his posture tense as he speaks with clear agitation. His facial expression conveys frustration, eyebrows furrowed, and his tone matches the intensity of his words: "Don't prompt me any more." The background includes a plain wall with minimal distractions, keeping the focus squarely on Person1's emotions and delivery. A faint shadow falls across his face from a single overhead light source, enhancing the dramatic effect of his irritation. Despite the brevity of the statement, the scene effectively communicates his annoyance, leaving no doubt about the strength of his feelings.
        Output:
        One person conversation. Persons: Person1: a man on the left side, wearing a gray shirt and dark pants, with short brown hair. Person1 is speaking. Scene Description: The video captures Person1 standing in a slightly dimly lit room, his posture tense and expressive of anger. His facial features are sharp, eyebrows furrowed, and his arms move emphatically as he speaks, conveying frustration. The background is minimalistic, with blurred neutral tones that keep the focus on Person1. The lighting highlights his emotions, casting shadows that enhance the intensity of the moment. The atmosphere is charged, suggesting a confrontation or strong reaction, but remains contained within the brief duration of the video. The emphasis is on Person1's animated gestures and the forceful delivery of his words, without including their specific content.
        
        Not rephrase, summarize, or reformat any other parts of the input. Only omit the direct speech or quoted dialogue.
        Now here's a new input text prompt, please generate the result in similar format, and do not output any other words:
    '''
    with LLM_lock:
        completion = client.chat.completions.create(
            model=qwen_model_name,
            messages=[
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": input+enhance_prompt},
                    ],
                },
            ],
        )
    return enhance_prompt, completion.choices[0].message.content


def generate_audio_prompt_two(input_prompt,enhanced_prompt):

    now = datetime.now()
    proj_name = now.strftime("project_%Y%m%d_%H%M%S")
    log_dir = "./logs"
    logpath = os.path.join(log_dir, proj_name + ".txt")

    # 如果目录不存在，则创建
    os.makedirs(log_dir, exist_ok=True)

    
    input='''If the input contains direct speech or quoted dialogue (i.e., text enclosed in quotation marks), extract and output only the quoted content without the quotation marks. Do not include any other text, explanation, or formatting. If there is no quoted content, return None.
        Here are some examples:
        Input:
        an angry and mad man  speaks "dont prompt me anymore because  I know I am not human"
        Output:
        dont prompt me anymore because I know I am not human
        
        Input:
        A confident young woman stands on a brightly lit TED-style stage, speaking "everybody, good night"
        Output:
        everybody, good night
        
        Input:
        A woman and a man is talking
        Output:
        None
        
        Now here's a new input text prompt, please extract the speech from the following input, and do not output any other words:
    '''
    with LLM_lock:
        completion = client.chat.completions.create(
            model=qwen_model_name,
            messages=[
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": input+input_prompt},
                    ],
                },
            ],
        )
        user_words = completion.choices[0].message.content
    
    
    
    
    input2='''
        You are provided with prompt designed for video generation model that describes a short video with sound, and the duration of the video cannot be more than 5.5 seconds. Please generate a JSON containing corresponding lines and timestamps for the persons who speak in the video, and descriptions of sound effects and background music. If the lines are short, the duration can be 4 seconds or shorter, but the timestamps must be adapted to the length of the lines, that means if Person1 or Person2 starts to speak at the start time, you must make sure he can finish before the end time.
        If user gives specific lines, your output must include it exactly. The gender of the person must be specified clearly: If a male is younger than 15 years old, he should be specified as "boy"; if he is older than 50 years old, he should be specified as "grandpa"; otherwise he will be specified as "man". Likely, if a female is younger than 15 years old, she should be specified as "girl"; if she is older than 50 years old, he should be specified as "grandma"; otherwise she will be specified as "woman". The descriptions of effects or music cannot be "None". As the clip is short, only one Person can actually speaking in the video, so the "number of lines" must be set to 1. Here's two example for you:

        input1:
        Two person conversation. Persons: Person1: a male on the left with glasses, wearing a black suit and holding a microphone. Person2: a woman on the right, wearing a shiny red dress, with dark hair pulled back, and holding a microphone. Person1 is speaking. Scene Description: Against a backdrop of other attendees and greenery, Person1 stands to the left, holding a microphone with the other hand keeping static. His mouth is open as he addresses Person2. Person2, positioned to the right, also holds a same microphone. She has a slight smile as she turns towards Person1, engaging with his question. The environment suggests a red-carpet event, with lighting fixtures visible overhead and a structure behind the two individuals.

        output1:
        {
            "gender":[
                {"Person1":"man"},
                {"Person2":"woman"}
            ],
            "number of lines":1,
            "duration":4.3,
            "lines":[
                {
                    "speaker":"Person1",
                    "text":"Hey guys, it is a question for you all, what do you think about the new game?",
                    "start":0.5,
                    "end":4.0
                }
            ],
            "effects":"Warm ambient hum, soft clothing movement, faint room reverb.",
            "music":"Low cinematic drone, tension strings, ambient suspense."
        }
        
        input2:
        Two person conversation. Persons: Person1: a woman on the left, wearing a white blouse with brown hair. Person2: a male on the right wearing a white sweater with black buttons. Person2 is speaking. Scene Description: In a lush garden setting, Person1 and Person2 stand together, engaged in conversation. A backdrop of vibrant pink flowers, citrus trees, and green foliage creates a tranquil atmosphere. Person2 appears to be reassuring or comforting Person1, who seems slightly worried or anxious. The setting suggests a private moment, possibly in a backyard or garden, away from the hustle and bustle of everyday life. Their gestures and expressions suggest a close relationship, possibly romantic or familial, as they navigate a situation that requires reassurance and understanding.

        output2:
        {
            "gender":[
                {"Person1":"woman"},
                {"Person2":"man"}
            ],
            "number of lines":1,
            "duration":5.0,
            "lines":[
                {
                    "speaker":"Person2",
                    "text":"Hey, everything will be fine, just relax and enjoy the moment.",
                    "start":0.3,
                    "end":3.5
                }
            ],
            "effects":"Distant city sounds, faint footsteps, light wind passing.",
            "music":"Upbeat acoustic guitar riff with a subtle electronic beat."
        }
        
        Now given an input as below, please generate a raw JSON object, without any markdown formatting (do not wrap it in triple backticks or use json code blocks), without any other descriptions:
    '''
    with LLM_lock:
        completion2 = client.chat.completions.create(
            model=qwen_model_name, # 此处以qwen-vl-max-latest为例，可按需更换模型名称。模型列表：https://help.aliyun.com/model-studio/getting-started/models
            messages=[
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": input2 + enhanced_prompt},
                    ],
                },
            ],
        )
        data=json.loads(completion2.choices[0].message.content)
    with open(logpath, 'w', encoding='utf-8') as f:
        f.write(input_prompt)
        f.write('\n')
        f.write(completion2.choices[0].message.content)


    voice_dict2={"girl":["t4U671CQHG58R11znrVj","BlgEcC0TfWpBak7FmvHW"],#LilyRose, Fena
                "boy":["tJHJUEHzOkMoPmJJ5jo2","E95NigJoVU5BI8HjQeN3"],#RyanQuin, StarboyNathan
                "man":["pNInz6obpgDQGcFmaJgB","ErXwobaYiN019PkySvjV"],#                    Adam,                  Antoni
                "woman":["Xb7hH8MSUJpSbSDYk0k2","9BWtsMINqrJLrRacOk9x"],#                      Alice,                 Aria
                "grandpa":["NOpBlnGInO9m6vDvFkFC","dPah2VEoifKnZT37774q"],#Grandpa,
                "grandma":["q1Hhtkt94vkD6q7p50hW","RILOU7YmBhvwJGDGjNmP"]#Alese,Jane
    }
    
    audio_dir = './tmp_audios'
    os.makedirs(audio_dir, exist_ok=True)

    genv0 = os.path.join(audio_dir, "genv0")
    genv1 = os.path.join(audio_dir, "genv1")
    geneffect = os.path.join(audio_dir, "geneffect")
    genmusic = os.path.join(audio_dir, "genmusic")
    genmix = os.path.join(audio_dir, "genmix")
    gentxt = os.path.join(audio_dir, "gentxt")
    
    os.makedirs(genv0, exist_ok=True)
    os.makedirs(genv1, exist_ok=True)
    os.makedirs(geneffect, exist_ok=True)
    os.makedirs(genmusic, exist_ok=True)
    os.makedirs(genmix, exist_ok=True)
    os.makedirs(gentxt, exist_ok=True)
    
    #这两个是生成的有声短时长，没有拓展到完整长度
    genv0path=os.path.join(genv0,proj_name)
    genv1path=os.path.join(genv1,proj_name)
    genvoiceprefix=[genv0path,genv1path]
    genvoicepath=[None,None]

    geneffectspath=os.path.join(geneffect,proj_name+".mp3")
    genmusicpath=os.path.join(genmusic,proj_name+".mp3")
    genmixpath=os.path.join(genmix,proj_name+".wav")
    promptpath=os.path.join(gentxt,proj_name+".txt")

    extendedprefixv0 = os.path.join(audio_dir, "extended_voice0")
    extendedprefixv1 = os.path.join(audio_dir, "extended_voice1")
    
    os.makedirs(extendedprefixv0, exist_ok=True)
    os.makedirs(extendedprefixv1, exist_ok=True)
    #这个是分别用静音拓展到完整长度的
    extendedprefix=[extendedprefixv0,extendedprefixv1]
    extendedpath=[None,None]

    cnt={"boy":0,"girl":0,"man":0,"woman":0,"grandpa":0,"grandma":0}
    use_dict={}

    for kv in data["gender"]:
        person_name, gender_value = next(iter(kv.items()))
        use_dict[person_name]=voice_dict2[gender_value][cnt[gender_value]]
        cnt[gender_value]=(cnt[gender_value]+1)%(len(voice_dict2[gender_value]))

    for i in range(data["number of lines"]):
        with elevenlab_lock:
            if (user_words != 'None'):
                audio = elevenlabs.text_to_speech.convert(
                    text=user_words,
                    voice_id=use_dict[data["lines"][i]["speaker"]],
                    model_id="eleven_multilingual_v2",
                    output_format="mp3_44100_128",
                    voice_settings=VoiceSettings(
                        # stability=0.5,
                        # similarity_boost=0.8,
                        speed=0.8,
                    ),
                )
            else:
                audio = elevenlabs.text_to_speech.convert(
                    text=data["lines"][i]["text"],
                    voice_id=use_dict[data["lines"][i]["speaker"]],
                    model_id="eleven_multilingual_v2",
                    output_format="mp3_44100_128",
                    voice_settings=VoiceSettings(
                        # stability=0.5,
                        # similarity_boost=0.8,
                        speed=0.8,
                    ),
                )
            person_id=int(data["lines"][i]["speaker"][-1])-1

            genvoicepath[person_id]=genvoiceprefix[person_id]+"_"+str(data["lines"][i]["start"])+"_"+str(data["lines"][i]["end"])+".mp3"
            extendedpath[person_id]=os.path.join(extendedprefix[person_id],os.path.basename(genvoicepath[person_id]))
            save(audio,genvoicepath[person_id])

    for i in range(2):
        if extendedpath[i]==None:
            extendedpath[i]=os.path.join(extendedprefix[i],proj_name+"_0.0_0.0.mp3")
        

    duration_seconds=data["duration"]
    print(data["effects"])
    with elevenlab_lock:
        effects= elevenlabs.text_to_sound_effects.convert(text=data["effects"],duration_seconds=duration_seconds)
        music  = elevenlabs.text_to_sound_effects.convert(text=data["music"],  duration_seconds=duration_seconds)

        save(effects, geneffectspath)
        save(music, genmusicpath)

    background_music = AudioSegment.from_mp3(genmusicpath)
    sound_effect = AudioSegment.from_mp3(geneffectspath)

    background_music = normalize_music(background_music)
    sound_effect = normalize_effect(sound_effect)

    output_duration = len(background_music)
    combined = AudioSegment.silent(duration=output_duration)

    # 添加背景音乐
    combined = combined.overlay(background_music)

    first_end=2147483647
    # 添加人声音频
    for i in range(2):
        silent=AudioSegment.silent(duration=output_duration)
        if genvoicepath[i] is not None:
            voice = AudioSegment.from_mp3(genvoicepath[i])
            voice = normalize_speech(voice)
            start,end=extract_times_from_filename(genvoicepath[i])
            start=start*1000.0
            end=end*1000.0
            first_end=min(first_end,end)
            
            extended= silent.overlay(voice,position=start)
            extended.export(extendedpath[i], format="mp3")
            combined= combined.overlay(voice, position=start)
        else:
            silent.export(extendedpath[i], format="mp3")


    # 添加音效
    combined = combined.overlay(sound_effect)

    # 导出最终合成的音频
    combined.export(genmixpath, format="wav")

    first_end=int(first_end)+200
    first_end=min(first_end,int(output_duration))
    save_prompt=enhanced_prompt.replace('\n', '').replace('\r', '').replace('\t', '')
    save_prompt=re.sub(r' +',' ',save_prompt).strip()

    with open(logpath, 'a', encoding='utf-8') as f:
        f.write('\n')
        f.write("use model for: ")
    with open(promptpath, 'w', encoding='utf-8') as f:
        f.write(save_prompt)
    return save_prompt, extendedpath[0], extendedpath[1], geneffectspath, genmusicpath, genmixpath


def generate_audio_prompt_one(input_prompt, enhanced_prompt):

    now = datetime.now()
    proj_name = now.strftime("project_%Y%m%d_%H%M%S")
    log_dir = "./logs"
    logpath = os.path.join(log_dir, proj_name + ".txt")

    # 如果目录不存在，则创建
    os.makedirs(log_dir, exist_ok=True)

    input='''If the input contains direct speech or quoted dialogue (i.e., text enclosed in quotation marks), extract and output only the quoted content without the quotation marks. Do not include any other text, explanation, or formatting. If there is no quoted content, return None.
        Here are some examples:
        Input:
        an angry and mad man  speaks "dont prompt me anymore because  I know I am not human"
        Output:
        dont prompt me anymore because I know I am not human
        
        Input:
        A confident young woman stands on a brightly lit TED-style stage, speaking "everybody, good night"
        Output:
        everybody, good night
        
        Input:
        A woman and a man is talking
        Output:
        None
        
        Now here's a new input text prompt, please extract the speech from the following input, and do not output any other words:
    '''
    with LLM_lock:
        completion = client.chat.completions.create(
            model=qwen_model_name,
            messages=[
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": input+input_prompt},
                    ],
                },
            ],
        )
        user_words = completion.choices[0].message.content
    

    input2='''
        You are provided with prompt designed for video generation model that describes a short video with sound, and the duration of the video cannot be more than 5.5 seconds. Please generate a JSON containing corresponding lines and timestamps for the persons who speak in the video, and descriptions of sound effects and background music. If the lines are short, the duration can be 4 seconds or shorter, but the timestamps must be adapted to the length of the lines, that means if Person1 starts to speak at the start time, you must make sure he can finish before the end time.
        If user gives specific lines, your output must include it exactly. The gender of the person must be specified clearly: If a male is younger than 15 years old, he should be specified as "boy"; if he is older than 50 years old, he should be specified as "grandpa"; otherwise he will be specified as "man". Likely, if a female is younger than 15 years old, she should be specified as "girl"; if she is older than 50 years old, he should be specified as "grandma"; otherwise she will be specified as "woman". The descriptions of effects or music cannot be "None". As the clip is short, only Person1 can actually speaking in the video, so the "number of lines" must be set to 1. Here's an example for you:

        input:
        One person conversation. Persons: Person1: a male on the left side of the frame who has short dark hair, a beard, and is wearing a black shirt. Person1 is speaking. Scene Description: In a dimly lit room, Person1 stands facing to his right, while an older, white man in a suit and glasses is positioned to his left, engaging in a conversation. A vertical arrangement of lights, centrally placed between them, casts stark reflections that dominate the backdrop. Person1 speaks, his mouth moving demonstrably, while the older listens intently, his expression serious and focused. Their discussion appears to be intense and perhaps adversarial, given the contrasting tones and the somber environment.

        output:
        {
            "gender":[
                {"Person1":"man"}
            ],
            "number of lines":1,
            "duration":5.0,
            "lines":[
                {
                    "speaker":"Person1",
                    "text":"In 20th century study of numerous hunter-gatherer tribes, though still ...",
                    "start":1.0,
                    "end":5.0
                }
            ],
            "effects":"Warm ambient hum, soft clothing movement, faint room reverb.",
            "music":"Low cinematic drone, tension strings, ambient suspense."
        }

        Now given an input as below, please generate a raw JSON object, without any markdown formatting (do not wrap it in triple backticks or use json code blocks), without any other descriptions:
    '''
    with LLM_lock:
        completion2 = client.chat.completions.create(
            model=qwen_model_name, # 此处以qwen-vl-max-latest为例，可按需更换模型名称。模型列表：https://help.aliyun.com/model-studio/getting-started/models
            messages=[
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": input2 + enhanced_prompt},
                    ],
                },
            ],
        )
        # import pdb; pdb.set_trace()
        print(completion2.choices[0].message.content)
        data=json.loads(completion2.choices[0].message.content)
    #return completion2.choices[0].message.content
    with open(logpath, 'w', encoding='utf-8') as f:
        f.write(input_prompt)
        f.write('\n')
        f.write(completion2.choices[0].message.content)


    voice_dict2={"girl":["t4U671CQHG58R11znrVj","BlgEcC0TfWpBak7FmvHW"],#LilyRose, Fena
                "boy":["tJHJUEHzOkMoPmJJ5jo2","E95NigJoVU5BI8HjQeN3"],#RyanQuin, StarboyNathan
                "man":["pNInz6obpgDQGcFmaJgB","ErXwobaYiN019PkySvjV"],#                    Adam,                  Antoni
                "woman":["Xb7hH8MSUJpSbSDYk0k2","9BWtsMINqrJLrRacOk9x"],#                      Alice,                 Aria
                "grandpa":["NOpBlnGInO9m6vDvFkFC","dPah2VEoifKnZT37774q"],#Grandpa,
                "grandma":["q1Hhtkt94vkD6q7p50hW","RILOU7YmBhvwJGDGjNmP"]#Alese,Jane
    }


    audio_dir = './tmp_audios'
    os.makedirs(audio_dir, exist_ok=True)

    genv0 = os.path.join(audio_dir, "genv0")
    genv1 = os.path.join(audio_dir, "genv1")
    geneffect = os.path.join(audio_dir, "geneffect")
    genmusic = os.path.join(audio_dir, "genmusic")
    genmix = os.path.join(audio_dir, "genmix")
    gentxt = os.path.join(audio_dir, "gentxt")
    
    os.makedirs(genv0, exist_ok=True)
    os.makedirs(genv1, exist_ok=True)
    os.makedirs(geneffect, exist_ok=True)
    os.makedirs(genmusic, exist_ok=True)
    os.makedirs(genmix, exist_ok=True)
    os.makedirs(gentxt, exist_ok=True)
    
    #这两个是生成的有声短时长，没有拓展到完整长度
    genv0path=os.path.join(genv0,proj_name)
    genv1path=os.path.join(genv1,proj_name)
    genvoiceprefix=[genv0path,genv1path]
    genvoicepath=[None,None]

    geneffectspath=os.path.join(geneffect,proj_name+".mp3")
    genmusicpath=os.path.join(genmusic,proj_name+".mp3")
    genmixpath=os.path.join(genmix,proj_name+".wav")
    promptpath=os.path.join(gentxt,proj_name+".txt")

    extendedprefixv0 = os.path.join(audio_dir, "extended_voice0")
    extendedprefixv1 = os.path.join(audio_dir, "extended_voice1")
    
    os.makedirs(extendedprefixv0, exist_ok=True)
    os.makedirs(extendedprefixv1, exist_ok=True)
    #这个是分别用静音拓展到完整长度的
    extendedprefix=[extendedprefixv0,extendedprefixv1]
    extendedpath=[None,None]

    cnt={"boy":0,"girl":0,"man":0,"woman":0,"grandpa":0,"grandma":0}
    use_dict={}

    for kv in data["gender"]:
        person_name, gender_value = next(iter(kv.items()))
        use_dict[person_name]=voice_dict2[gender_value][cnt[gender_value]]
        cnt[gender_value]=(cnt[gender_value]+1) % (len(voice_dict2[gender_value]))

    for i in range(data["number of lines"]):
        with elevenlab_lock:
            if (user_words != 'None'):
                audio = elevenlabs.text_to_speech.convert(
                    text=user_words,
                    voice_id=use_dict[data["lines"][i]["speaker"]],
                    model_id="eleven_multilingual_v2",
                    output_format="mp3_44100_128",
                    voice_settings=VoiceSettings(
                        speed=0.8,
                    ),
                )
            else:
                audio = elevenlabs.text_to_speech.convert(
                    text=data["lines"][i]["text"],
                    voice_id=use_dict[data["lines"][i]["speaker"]],
                    model_id="eleven_multilingual_v2",
                    output_format="mp3_44100_128",
                    voice_settings=VoiceSettings(
                        speed=0.8,
                    ),
                )
            person_id=int(data["lines"][i]["speaker"][-1])-1

            genvoicepath[person_id]=genvoiceprefix[person_id]+"_"+str(data["lines"][i]["start"])+"_"+str(data["lines"][i]["end"])+".mp3"
            extendedpath[person_id]=os.path.join(extendedprefix[person_id],os.path.basename(genvoicepath[person_id]))
            save(audio,genvoicepath[person_id])

    for i in range(2):
        if extendedpath[i]==None:
            extendedpath[i]=os.path.join(extendedprefix[i],proj_name+"_0.0_0.0.mp3")
        

    duration_seconds=data["duration"]
    print(data["effects"])
    with elevenlab_lock:
        effects= elevenlabs.text_to_sound_effects.convert(text=data["effects"],duration_seconds=duration_seconds)
        music  = elevenlabs.text_to_sound_effects.convert(text=data["music"],  duration_seconds=duration_seconds)
        save(effects, geneffectspath)
        save(music, genmusicpath)

    background_music = AudioSegment.from_mp3(genmusicpath)
    sound_effect = AudioSegment.from_mp3(geneffectspath)
    
    background_music = normalize_music(background_music)
    sound_effect = normalize_effect(sound_effect)

    output_duration = len(background_music)
    combined = AudioSegment.silent(duration=output_duration)

    # 添加背景音乐
    combined = combined.overlay(background_music)

    first_end=2147483647
    # 添加人声音频
    for i in range(2):
        silent=AudioSegment.silent(duration=output_duration)
        if genvoicepath[i] is not None:
            voice = AudioSegment.from_mp3(genvoicepath[i])
            voice = normalize_speech(voice)
            start,end=extract_times_from_filename(genvoicepath[i])
            start=start*1000.0
            end=end*1000.0
            first_end=min(first_end,end)
            
            extended= silent.overlay(voice,position=start)
            extended.export(extendedpath[i], format="mp3")
            combined= combined.overlay(voice, position=start)
        else:
            silent.export(extendedpath[i], format="mp3")


    # 添加音效
    combined = combined.overlay(sound_effect)

    # 导出最终合成的音频
    combined.export(genmixpath, format="wav")

    first_end=int(first_end)+200
    first_end=min(first_end,int(output_duration))
    save_prompt=enhanced_prompt.replace('\n', '').replace('\r', '').replace('\t', '')
    save_prompt=re.sub(r' +',' ',save_prompt).strip()

    with open(logpath, 'a', encoding='utf-8') as f:
        f.write('\n')
        f.write("use model for: ")


    
    with open(promptpath, 'w', encoding='utf-8') as f:
        f.write(save_prompt)
    return save_prompt, extendedpath[0], geneffectspath, genmusicpath, genmixpath


def generate_audio_prompt_effect(input_prompt,enhanced_prompt):

    now = datetime.now()
    proj_name = now.strftime("project_%Y%m%d_%H%M%S")
    log_dir = "./logs"
    logpath = os.path.join(log_dir, proj_name + ".txt")

    # 如果目录不存在，则创建
    os.makedirs(log_dir, exist_ok=True)

    input2='''
        You are provided with prompt designed for video generation model that describes a short video with sound, and the duration of the video cannot be more than 5.5 seconds. Please generate a JSON containing descriptions of sound effects and background music. Here's an example for you:

        input:
        No conversation. Scene Description: A close-up shot of a clear glass with a textured surface, placed on a wooden surface. The glass is partially filled with water, and the focus is on the water's movement as it is poured into the glass from above. The pouring action creates ripples and bubbles in the water, which are captured in detail. The background is blurred, emphasizing the glass and the water.

        output:
        {
            "duration":4.3,
            "effects":"Close-up water pouring sound, gentle bubbling, glass resonance, soft wood contact tone.",
            "music":"Minimal ambient pads, light harmonic tones, subtle shimmer for clarity and focus."
        }

        Now given an input as below, please generate a raw JSON object, without any markdown formatting (do not wrap it in triple backticks or use json code blocks), without any other descriptions:
    '''
    with LLM_lock:
        completion2 = client.chat.completions.create(
            model=qwen_model_name, # 此处以qwen-vl-max-latest为例，可按需更换模型名称。模型列表：https://help.aliyun.com/model-studio/getting-started/models
            messages=[
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": input2 + enhanced_prompt},
                    ],
                },
            ],
        )
        # import pdb; pdb.set_trace()
        print(completion2.choices[0].message.content)
        data=json.loads(completion2.choices[0].message.content)
    #return completion2.choices[0].message.content
    with open(logpath, 'w', encoding='utf-8') as f:
        f.write(input_prompt)
        f.write('\n')
        f.write(completion2.choices[0].message.content)

    audio_dir = './tmp_audios'
    os.makedirs(audio_dir, exist_ok=True)

    genv0 = os.path.join(audio_dir, "genv0")
    genv1 = os.path.join(audio_dir, "genv1")
    geneffect = os.path.join(audio_dir, "geneffect")
    genmusic = os.path.join(audio_dir, "genmusic")
    genmix = os.path.join(audio_dir, "genmix")
    gentxt = os.path.join(audio_dir, "gentxt")
    
    os.makedirs(genv0, exist_ok=True)
    os.makedirs(genv1, exist_ok=True)
    os.makedirs(geneffect, exist_ok=True)
    os.makedirs(genmusic, exist_ok=True)
    os.makedirs(genmix, exist_ok=True)
    os.makedirs(gentxt, exist_ok=True)
    
    #这两个是生成的有声短时长，没有拓展到完整长度
    genv0path=os.path.join(genv0,proj_name)
    genv1path=os.path.join(genv1,proj_name)
    genvoiceprefix=[genv0path,genv1path]
    genvoicepath=[None,None]

    geneffectspath=os.path.join(geneffect,proj_name+".mp3")
    genmusicpath=os.path.join(genmusic,proj_name+".mp3")
    genmixpath=os.path.join(genmix,proj_name+".wav")
    promptpath=os.path.join(gentxt,proj_name+".txt")

    extendedprefixv0 = os.path.join(audio_dir, "extended_voice0")
    extendedprefixv1 = os.path.join(audio_dir, "extended_voice1")
    
    os.makedirs(extendedprefixv0, exist_ok=True)
    os.makedirs(extendedprefixv1, exist_ok=True)
    #这个是分别用静音拓展到完整长度的
    extendedprefix=[extendedprefixv0,extendedprefixv1]
    extendedpath=[None,None]



    for i in range(2):
        if extendedpath[i]==None:
            extendedpath[i]=os.path.join(extendedprefix[i],proj_name+"_0.0_0.0.mp3")
        

    duration_seconds=data["duration"]
    with elevenlab_lock:
        effects= elevenlabs.text_to_sound_effects.convert(text=data["effects"],duration_seconds=duration_seconds)
        music  = elevenlabs.text_to_sound_effects.convert(text=data["music"],  duration_seconds=duration_seconds)

        save(effects, geneffectspath)
        save(music, genmusicpath)




    background_music = AudioSegment.from_mp3(genmusicpath)
    sound_effect = AudioSegment.from_mp3(geneffectspath)
    
    background_music = normalize_music(background_music)
    sound_effect = normalize_effect(sound_effect)

    # 以背景音乐长度为基础构建合成音轨（可以根据需要扩展长度）
    output_duration = len(background_music)
    combined = AudioSegment.silent(duration=output_duration)

    # 添加背景音乐
    combined = combined.overlay(background_music)

    first_end=2147483647
    # 添加人声音频
    for i in range(2):
        silent=AudioSegment.silent(duration=output_duration)
        
        silent.export(extendedpath[i], format="mp3")


    # 添加音效
    combined = combined.overlay(sound_effect)

    # 导出最终合成的音频
    combined.export(genmixpath, format="wav")

    first_end=int(first_end)+200
    first_end=min(first_end,int(output_duration))
    #保存到生成视频需要的txt
    save_prompt=enhanced_prompt.replace('\n', '').replace('\r', '').replace('\t', '')
    save_prompt=re.sub(r' +',' ',save_prompt).strip()

    with open(logpath, 'a', encoding='utf-8') as f:
        f.write('\n')
        f.write("use model for: ")

    with open(promptpath, 'w', encoding='utf-8') as f:
        f.write(save_prompt)
    return save_prompt, extendedpath[0], geneffectspath, genmusicpath, genmixpath

