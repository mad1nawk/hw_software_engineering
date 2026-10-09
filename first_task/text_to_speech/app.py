import torch
import soundfile as sf

device = torch.device('cpu')
torch.set_num_threads(4)

repo = 'snakers4/silero-models'
model, example_text = torch.hub.load(
    repo_or_dir=repo,
    model='silero_tts',
    language='ru',
    speaker='v5_5_ru'
)

model.to(device)
speaker = 'eugene'
sample_rate = 48000

text_to_speak = "Прив+ет! как дела? что д+елаешь?"
audio = model.apply_tts(
    text=text_to_speak,
    speaker=speaker,
    sample_rate=sample_rate,
    put_accent=True,
    put_yo=True
)

output_path = 'output_hub.wav'
sf.write(output_path, audio.numpy(), sample_rate)
