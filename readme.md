# Whisper Transcriber

Скрипт для расшифровки аудиофайлов с помощью локальной модели OpenAI Whisper.

## Установка

1. Создайте виртуальное окружение:
   ```bash
   python -m venv whisper_env
   source whisper_env/bin/activate  # macOS/Linux
   whisper_env\Scripts\activate.bat  # Windows
   ```

2. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```

## Использование

1. Поместите аудиофайл в папку `audio/` (например, `audio/input.wav`).
2. Запустите скрипт одним из способов:

   ```bash
   python whisper_transcribe.py
   ```

   или

   ```bash
   ./whisper_transcribe.py
   ```

## Аргументы

Вы можете указать параметры при запуске:

```bash
python whisper_transcribe.py --audio audio/input.wav --model base --language ru
```

- `--audio`: путь к аудиофайлу (по умолчанию `audio/audio.wav`)
- `--model`: название модели (`tiny`, `base`, `small`, `medium`, `large-v3`)
- `--language`: язык речи (например, `ru`, `en`, `es`)
- `--download-only`: скачать модель и выйти
- `--help`: показать справку

## Результат

Результат сохраняется в файл `transcripts/<имя_файла>.txt`, разбитый на блоки по 3 минуты.

## Модели

При первом запуске нужная модель автоматически скачивается и сохраняется в кэш `~/.cache/whisper`.

| Модель     | Примерный размер |
|------------|------------------|
| tiny       | ~75 MB           |
| base       | ~142 MB          |
| small      | ~466 MB          |
| medium     | ~1.5 GB          |
| large-v3   | ~2.9 GB          |

Первый запуск с новой моделью может занять несколько минут.
