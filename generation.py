from diffusers import AudioLDMPipeline
import torch
from playsound import playsound

repo_id = "cvssp/audioldm-s-full-v2"
pipe = AudioLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)

prompt = "The harbor is crowded with people, accompanied by the sound of ships going out to sea."
audio = pipe(prompt, num_inference_steps=10, audio_length_in_s=5.0).audios[0]

playsound(audio)