# Whisper Transcription

Скрипт для расшифровки аудиофайлов в текст с помощью OpenAI Whisper. Использует локальные `.pt` модели и сохраняет результат в текст с таймкодами раз в 5 минут.

## Требования

- Python 3.10+
- pip install -r requirements.txt
- Модели `.pt` в папке `models/`

## Структура

whisper/
├── audio/              # аудиофайлы
├── models/             # whisper модели (.pt)
├── transcripts/        # результаты (.txt)
├── whisper_transcribe.py
└── requirements.txt

## Использование

```bash
python whisper_transcribe.py [--audio <путь_к_аудио>] [--model <модель>] [--language <язык>]
```

Аргументы:
  --audio     путь к аудиофайлу (по умолчанию: audio/audio.wav)
  --model     название модели Whisper, например: tiny, base, medium, large-v3 (по умолчанию: tiny)
  --language  язык распознавания: ru, en, es и др. (по умолчанию: ru)

Примеры:

python whisper_transcribe.py --model large-v3
python whisper_transcribe.py --audio audio/meeting.wav --model medium --language en

Результат

Файл с текстом сохраняется в transcripts/<имя_аудио>.txt
Таймкоды вставляются каждые 5 минут.

