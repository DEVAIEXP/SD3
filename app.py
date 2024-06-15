# Copyright 2024 DEVAIEXP. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import json, os, torch, random, argparse
import gradio as gr
import numpy as np
from pipelines.pipeline_common import quantize_4bit, torch_gc
from pipelines.pipeline_stable_diffusion_3 import StableDiffusion3PipelineV2, SD3Transformer2DModel
from transformers import BitsAndBytesConfig, T5EncoderModel
from diffusers.utils import logging
from PIL import Image

from utils import load_styles, open_folder, read_image_metadata, save_image_with_metadata

# Set up argument parser
parser = argparse.ArgumentParser(description="Gradio interface for text-to-image generation with optional features.")
parser.add_argument("--share", action="store_true", help="Enable Gradio sharing.")
parser.add_argument("--lowvram", action="store_true", help="Enable CPU offload for model operations.")
parser.add_argument("--mmdit_load_mode", default=None, type=str, choices=["4bit", "8bit"], help="Quantization mode for MMDiT")
parser.add_argument("--t5_load_mode", default=None, type=str, choices=["4bit", "8bit"], help="Quantization mode for text_encoder_3")
logger = logging.get_logger(__name__)

# Parse arguments
args = parser.parse_args()
share = args.share

mmdit_load_mode = args.mmdit_load_mode
t5_load_mode = args.t5_load_mode

ENABLE_CPU_OFFLOAD = args.lowvram
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.allow_tf32 = False

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

if not torch.cuda.is_available():
    print("Running on CPU 🥶")

dtypeQuantize = dtype
if mmdit_load_mode in ('8bit', '4bit'):
    dtypeQuantize = torch.float8_e4m3fn

print(f"used dtype {dtypeQuantize}")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

HUGGING_FACE_API_KEY = ""
config_dict = {}
model_id = "stabilityai/stable-diffusion-3-medium-diffusers"

pipe_sd_transformer = None
pipe = None

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1344

styles = load_styles()

def restart_cpu_offload():    
    from pipelines.pipeline_common import optionally_disable_offloading
    optionally_disable_offloading(pipe)    
    torch_gc()
    pipe.enable_model_cpu_offload()    

def load_config():
    global HUGGING_FACE_API_KEY, config_dict

    if os.path.exists("./config.json"):
        with open("./config.json", "r", encoding="utf8") as file:
            config = file.read()

        config_dict = json.loads(config)
        HUGGING_FACE_API_KEY = config_dict["huggingface_api_key"]        
    
    return HUGGING_FACE_API_KEY

def save_config(*args):
    global config_dict
    
    values_dict = zip(config_dict.keys(), args)
    config_dict_values = dict(values_dict)    
    status=""
    try:
        with open('./config.json', 'w') as f:
            json.dump(config_dict_values, f,indent=2)
        load_config()
        
    except:
        status = "<center><h3 style='color: #E74C3C;'>There was an error saving the settings!</h3></center>"
        pass
    
    return gr.Tabs(selected=0), status

def set_metadata_settings(image_path, style_dropdown):
    if image_path is None:
        return (gr.update(),) * 11 
    
    with Image.open(image_path) as img:
        metadata = img.info
        prompt = metadata.get("Prompt", "")
        negative_prompt = metadata.get("Negative Prompt", "")
        style = metadata.get("Style", "No Style")
        seed = int(metadata.get("Seed", "0"))
        width = int(metadata.get("Width", "1024"))
        height = int(metadata.get("Height", "1024"))
        guidance_scale = float(metadata.get("Guidance Scale (CFG)", "5.0"))        
        num_inference_steps = int(metadata.get("Inference Steps", "28"))                
        number_of_images_per_prompt = int(metadata.get("Number Of Images To Generate", "1"))        

    # Construct the updates list with gr.update calls for each setting
    updates = [
        gr.update(value=prompt),
        gr.update(value=negative_prompt),
        gr.update(value=style),
        gr.update(value=seed),
        gr.update(value=width),
        gr.update(value=height),
        gr.update(value=guidance_scale),        
        gr.update(value=num_inference_steps),
        gr.update(value=number_of_images_per_prompt),        
        gr.update(value=False)
    ]
    
    return tuple(updates)

def generate(prompt: str,
            negative_prompt: str = "",
            style: str = "No Style",
            seed: int = 0,
            width: int = 1024,
            height: int = 1024,
            guidance_scale: float = 5.0,
            num_inference_steps: int = 28, 
            randomize_seed_ck: bool = False,
            number_of_images_per_prompt: int = 1,
            loop_styles_ck: bool = False           
          ):        
        
    global pipe_sd_transformer, pipe
    status = "<center><h3 style='color: #2E86C1;'>Image generation finished!</h3></center>"

    if not HUGGING_FACE_API_KEY :        
        status="<center><h3 style='color: #E74C3C;'>You need to set the Huggingface API key in the 'Other Settings' tab before proceeding!</h3></center>"
        return status, None, seed
    
    if torch.cuda.is_available():                
        if pipe is None:
            pipe_sd_transformer = SD3Transformer2DModel.from_pretrained(
            model_id, subfolder="transformer", token=HUGGING_FACE_API_KEY).to(device, dtypeQuantize)

            if mmdit_load_mode == '4bit':
                quantize_4bit(pipe_sd_transformer, dtype)
            
            if t5_load_mode:
                pipeline_param = {
                    'pretrained_model_name_or_path': model_id,
                    'use_safetensors': True,
                    'torch_dtype': dtype,   
                    'transformer': pipe_sd_transformer,
                    'token': HUGGING_FACE_API_KEY,                        
                    'text_encoder_3': None
                }
            else:
                pipeline_param = {
                    'pretrained_model_name_or_path': model_id,
                    'use_safetensors': True,
                    'torch_dtype': dtype,   
                    'transformer': pipe_sd_transformer,
                    'token': HUGGING_FACE_API_KEY                                         
                }

            pipe = StableDiffusion3PipelineV2.from_pretrained(**pipeline_param).to(device)
            
            if ENABLE_CPU_OFFLOAD:
                pipe.enable_model_cpu_offload()  
            
            pipe.transformer.to(memory_format=torch.channels_last)
            pipe.vae.to(memory_format=torch.channels_last)
            
            #set text_encoder after move other components to cpu
            if t5_load_mode:
                kwargs = {"device_map": device}
                            
                if not device.type.startswith("cuda"):
                    kwargs['device_map'] = {"": device}

                kwargs['quantization_config'] = BitsAndBytesConfig(
                    load_in_4bit= True if t5_load_mode == '4bit' else False,
                    load_in_8bit= True if t5_load_mode == '8bit' else False,
                    llm_int8_enable_fp32_cpu_offload = True if t5_load_mode == '8bit' and ENABLE_CPU_OFFLOAD else False,
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type='nf4'
                )                      
                t3_encoder=T5EncoderModel.from_pretrained(model_id, token=HUGGING_FACE_API_KEY, low_cpu_mem_usage=True, subfolder="text_encoder_3", torch_dtype=dtype, **kwargs)             
                pipe.text_encoder_3 = t3_encoder

                del t3_encoder
                           
        else:            
            if ENABLE_CPU_OFFLOAD:
                restart_cpu_offload()
                
        torch_gc()       
        images = []  # Initialize an empty list to collect generated images
        original_seed = seed  # Store the original seed value
        
        # Parse the prompt and split it into multiple prompts if it's multi-line
        prompt_lines = prompt.split('\n')
        image_counter = 1
        for line in prompt_lines:
            original_prompt = line.strip()
            original_neg_prompt = negative_prompt

            # Use all styles if loop_styles_ck is True, otherwise use only the selected style
            selected_styles = styles if loop_styles_ck else [(style, "", "")]
            total_images = len(selected_styles) * number_of_images_per_prompt * len(prompt_lines)
            

            for style_name in selected_styles:
                get_name = style_name[0]
                if(len(get_name) < 2):
                    get_name = style_name
                style_prompt, style_negative_prompt = styles.get(get_name, ("", ""))
        
                # Replace placeholders in the style prompt
                prompt = style_prompt.replace("{prompt}", original_prompt) if style_prompt else original_prompt
                negative_prompt = style_negative_prompt if style_negative_prompt else original_neg_prompt

                print(f"\nFinal Prompt: {prompt}")       
                print(f"Final Negative Prompt: {negative_prompt}\n")     

                for i in range(number_of_images_per_prompt):
                    if randomize_seed_ck or i > 0:  # Update seed if randomize is checked or for subsequent images
                        seed = random.randint(0, MAX_SEED)      
                        
                    print(f"Image {image_counter}/{total_images} Being Generated")
                    image_counter=image_counter+1
                    generator = torch.Generator().manual_seed(seed)
                    with torch.cuda.amp.autocast(dtype=dtype):
                        output = pipe(
                            prompt = prompt, 
                            negative_prompt = negative_prompt,
                            guidance_scale = guidance_scale, 
                            num_inference_steps = num_inference_steps, 
                            width = width, 
                            height = height,
                            generator = generator,
                            device=device,
                            dtype=dtype
                        ).images

                    # Append generated images to the images list
                    images.extend(output)

                    # Optionally, save each image
                    output_folder = 'outputs'
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)
                    for image in images:
                        # Generate timestamped filename
                        timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')
                        image_filename = f"{output_folder}/{timestamp}.png"
                    
                        # Prepare metadata
                        metadata = {
                            "Prompt": original_prompt,
                            "Negative Prompt": original_neg_prompt,
                            "Style":style,
                            "Seed": seed,
                            "Width": width,
                            "Height": height,
                            "Guidance Scale (CFG)": guidance_scale,
                            "Inference Steps": num_inference_steps,                                                            
                        }
                    
                        # Save image with metadata
                        save_image_with_metadata(image, image_filename, metadata)
                    
                        torch_gc()

        return status, images, seed
    else:
        pipe = None
        return
    
examples = [
    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    "An astronaut riding a green horse",
    "A delicious ceviche cheesecake slice",
]

# Initialize configuration
load_config()

# Description
title = r"""
<h1 align="center">Stable Diffusion 3 Medium - WebApp for image generation</h1>
"""

description = r"""
<h2><b>WebApp by <a href='https://github.com/DEVAIEXP/SD3'>DEVAIEXP</a></b> for <a href='https://huggingface.co/stabilityai/stable-diffusion-3-medium'><b>Stable Diffusion 3 Medium</b></a>.</h2><br>
<b>How to use:</b><br>
1. Access <a href='https://huggingface.co/stabilityai/stable-diffusion-3-medium'><b>Stable Diffusion 3 Medium</b></a> and create an account if you don't have one. Fill out the opt-in form to access the model, then generate or get your API key em <a href='https://huggingface.co/settings/tokens'> Access Token</a><br>
2. Configure your <b>Huggingface API</b> key in the 'Other Settings' tab and save.<br>
3. Go to the 'Image geration' tab and play!
"""

about = r"""
---
📝 **More**
<br>
Learn more about the <a href='https://stability.ai/news/stable-diffusion-3'>Stable Diffusion 3 series.</a> Try on <a href='https://platform.stability.ai/docs/api-reference\#tag/Generate/paths/~1v2beta~1stable-image~1generate~1sd3/post'>Stability AI API</a>, <a href='https://stability.ai/stable-assistant'>Stable Assistant</a>. 

📧 **Contact**
<br>
If you have any questions or suggestions, feel free to send your question to <b>contact@devaiexp.com</b>.
"""

css = """
    footer {visibility: hidden},
    .gradio-container {width: 85% !important}
    """

with gr.Blocks(theme=gr.themes.Default(), css=css) as demo:
    
    gr.Markdown(title)
    gr.Markdown(description)

    with gr.Column(elem_id="col-container"):
        with gr.Tabs() as tabs:
            with gr.TabItem("Image Generation", id=0):    
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            prompt=gr.Textbox(
                                label="Prompt - Each New Line is Parsed as a New Prompt",
                                placeholder="Enter your prompt",
                                lines=3)
                        with gr.Row():
                            negative_prompt = gr.Text(
                            label="Negative prompt",
                            max_lines=1,
                            placeholder="Enter a negative prompt",
                    )
                        with gr.Accordion("Advanced Settings", open=False): 
                                 
                            with gr.Row():
                                style_dropdown = gr.Dropdown(label="Style", choices=list(styles.keys()), value="No Style")
                                loop_styles_ck = gr.Checkbox(label="Loop All Styles", value=False)  # New Checkbox for looping through all styles
                            with gr.Row():
                                seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)                                
                                randomize_seed_ck = gr.Checkbox(label="Randomize seed", value=True)
                            with gr.Row():
                                width = gr.Slider(label="Width", minimum=256, maximum=MAX_IMAGE_SIZE, step=64, value=1024)
                                height = gr.Slider(label="Height", minimum=256, maximum=MAX_IMAGE_SIZE, step=64, value=1024)
                            with gr.Row():                    
                                number_of_images_per_prompt = gr.Slider(label="Number Of Images To Generate", minimum=1, maximum=9999999, step=1, value=1) 
                            with gr.Row():                
                                guidance_scale = gr.Slider(label="Guidance Scale (CFG)", minimum=0.0, maximum=10.0, step=0.1, value=5.0)                        
                                num_inference_steps = gr.Slider(label="Number of inference steps",minimum=1,maximum=50,step=1,value=28)
                        with gr.Row():
                            run_button = gr.Button("Generate", scale=1)
                        #with gr.Row():
                            btn_open_outputs = gr.Button("Open Outputs Folder (Works on Windows & Desktop Linux)", scale=2)
                            btn_open_outputs.click(fn=open_folder)
                    with gr.Column():
                        result = gr.Gallery(label="Result", show_label=False, height=768,format="png")
                gr.Examples(
                    examples = examples,
                    inputs = [prompt]
                )
                status = gr.HTML(elem_id="status", value="")

            with gr.TabItem("Image Metadata",id=1) as TabMeta:
                with gr.Row():
                    set_metadata_button = gr.Button("Load & Set Metadata Settings")
                with gr.Row():
                    with gr.Column():
                        metadata_image_input = gr.Image(type="filepath", label="Upload Image")
                    with gr.Column():
                        metadata_output = gr.Textbox(label="Image Metadata", lines=25, max_lines=50)

                metadata_image_input.change(fn=read_image_metadata, inputs=[metadata_image_input], outputs=[metadata_output])
                set_metadata_button.click(fn=set_metadata_settings, inputs=[metadata_image_input, style_dropdown], outputs=[
                    prompt, negative_prompt, style_dropdown, seed, width, height,
                    guidance_scale, num_inference_steps, number_of_images_per_prompt, randomize_seed_ck
                ])
                
            with gr.TabItem("Other Settings", id=2) as TabConfig:
                with gr.Row():
                    with gr.Column():                  
                        huggingface_api_key = gr.Textbox(label="Huggingface API Key", placeholder="Enter your API-Key here")                
                
                save_btn = gr.Button(value="💾Save")        
                save_input_elements = huggingface_api_key
                save_btn.click(save_config,inputs=[save_input_elements], outputs=[tabs, status]) 

       
    # Set configuration inputs
    TabConfig.select(load_config, outputs=[save_input_elements])
    gr.Markdown(about)  
    gr.on(
        triggers=[run_button.click, prompt.submit, negative_prompt.submit],
        fn = generate, show_progress="full",
        inputs = [prompt, 
                  negative_prompt, 
                  style_dropdown,
                  seed, 
                  width, 
                  height, 
                  guidance_scale, 
                  num_inference_steps,
                  randomize_seed_ck, 
                  number_of_images_per_prompt,
                  loop_styles_ck],
        outputs = [status, result, seed]
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True, share=share)