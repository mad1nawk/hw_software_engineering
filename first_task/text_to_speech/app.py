import os
import torch
import soundfile as sf

# 1. Настройка процессора
device = torch.device('cpu')
torch.set_num_threads(4)

# 2. Чистая загрузка Silero V5 через локальный JIT-хаб
# Используем встроенный метод загрузки, который гарантированно возвращает модель
repo = 'snakers4/silero-models'
model, example_text = torch.hub.load(
    repo_or_dir=repo,
    model='silero_tts',
    language='ru',
    speaker='v5_5_ru'
)

# Переносим модель на процессор
model.to(device)

# 3. Настройки озвучки
# Доступные голоса: 'xenia', 'baya', 'aidar', 'eugene'
speaker = 'eugene'
sample_rate = 48000     # Частота звука (8000, 24000 или 48000)

# Ваш текст. Используйте "+" перед гласной для ударения (например: прив+ет)
text_to_speak = "Прив+ет! Теп+ерь всё работает отл+ично. Мод+ель Силеро усп+ешно запущена!"

# 4. Генерация аудио
# В V5 рекомендуется явно передавать параметры put_accent и put_yo
audio = model.apply_tts(
    text=text_to_speak,
    speaker=speaker,
    sample_rate=sample_rate,
    put_accent=True,
    put_yo=True
)

# 5. Сохранение файла
output_path = 'output_hub.wav'
sf.write(output_path, audio.numpy(), sample_rate)