# Whisper Transcriber

Скрипт для расшифровки аудио с помощью локальной модели OpenAI Whisper. Работает с `.pt` файлами, сохраняет результат с таймкодами каждые 5 минут.

## Установка

```bash
pip install -r requirements.txt
```

При первом запуске модель Whisper будет автоматически загружена из HuggingFace и сохранена в кэш `~/.cache/whisper`. Если вы хотите скачать модель заранее, используйте флаг `--download-only`.

Возможные имена моделей: `tiny`, `base`, `small`, `medium`, `large-v3`.

| Модель     | Размер файла |
|------------|--------------|
| tiny       | ~75 MB       |
| base       | ~142 MB      |
| small      | ~466 MB      |
| medium     | ~1.5 GB      |
| large-v3   | ~2.9 GB      |

⚠️ Первый запуск может занять время из-за загрузки модели.
```

## Структура проекта

```
whisper/
├── audio/         # входные аудиофайлы
├── models/        # модели Whisper (.pt)
├── transcripts/   # расшифрованные тексты
└── whisper_transcribe.py
```

## Использование

```bash
python whisper_transcribe.py [--audio path] [--model name] [--language code]
```

Параметры:
- `--audio` путь к аудиофайлу (по умолчанию: `audio/audio.wav`)
- `--model` имя модели (tiny, base, medium, large-v3; по умолчанию: `tiny`)
- `--language` язык (например, ru, en, es; по умолчанию: `ru`)

Также доступен флаг `--download-only`, чтобы скачать модель без запуска расшифровки.

## Вывод

Результат сохраняется в `transcripts/<имя_аудио>.txt` с таймкодами каждые 3 минуты.
