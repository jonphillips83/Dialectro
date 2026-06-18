# tts_gpu.py
import torch
import sounddevice as sd
from kokoro import KPipeline


# Dynamically select device based on what PyTorch can physically see
chosen_device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"PyTorch is executing on: {chosen_device.upper()}")

lang_codes = ['a', 'b']
voices = ['af_nicole', 'af_sky', 'af_heart', 'af_bella', 'af_sarah']

# Pass the resolved device variable
pipeline = KPipeline(lang_code='b', device=chosen_device)

if chosen_device == 'cuda':
    text = "GPU ACCELERATED INFERENCE: But the G7 proposal for further talks involving European leaders about Iran’s ballistic missiles and support for proxy forces is bound to be rejected by Iran. Tehran has been negotiating exclusively with the US and regards Europe as largely irrelevant. Iran is also likely to reject France and Britain’s plan for a taskforce to escort ships through the strait of Hormuz, a proposal endorsed in the G7 leaders’ statement."
else:
    text = 'CPU MODE'
    
generator = pipeline(
    text, 
    voice='bf_emma', 
    speed=1, 
    split_pattern=r'\n+|(?<=[.?!])\s+'
)

for i, (graphemes, phonemes, audio) in enumerate(generator):
    sd.play(audio, samplerate=24000)
    sd.wait()