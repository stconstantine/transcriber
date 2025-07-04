import os
import whisper
import requests
from tqdm import tqdm
import torch
import math
import argparse

def download_model(model_name: str, dest_folder: str):
    url = f"https://huggingface.co/openai/whisper/resolve/main/{model_name}.pt"
    local_path = os.path.join(dest_folder, f"{model_name}.pt")
    os.makedirs(dest_folder, exist_ok=True)

    print(f"Модель {model_name}.pt не найдена. Скачиваем в {dest_folder}/...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(local_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=model_name) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

def format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02}:{m:02}:{s:02}"

def main():
    parser = argparse.ArgumentParser(description="Whisper Transcriber")
    parser.add_argument('--audio', type=str, default='audio/audio.wav', help='Путь к аудиофайлу')
    parser.add_argument('--model', type=str, default='tiny', help='Название модели Whisper')
    parser.add_argument('--language', type=str, default='ru', help='Язык аудио (например, ru, en, es...)')
    args = parser.parse_args()

    audio_path = args.audio
    model_name = args.model
    language = args.language

    if not os.path.isfile(audio_path):
        print(f"Ошибка: файл не найден: {audio_path}")
        return

    model_path = os.path.join("models", f"{model_name}.pt")
    if not os.path.isfile(model_path):
        download_model(model_name, "models")

    print(f"Загружаем модель из: {model_path}")
    model = whisper.load_model(model_path)

    # Загружаем аудио для получения длительности
    audio = whisper.load_audio(audio_path)
    audio_duration = len(audio) / whisper.audio.SAMPLE_RATE

    print(f"Распознавание начато: {audio_path} (язык: {language}, длительность: {audio_duration:.1f} сек)")

    # Прогресс-бар по времени (секундам)
    progress = tqdm(total=audio_duration, unit="sec", dynamic_ncols=True)

    # Здесь создадим коллбек для обновления прогресса вручную
    # Whisper не дает встроенного коллбека, поэтому будем имитировать прогресс построчно

    # Распознаем всю аудио
    result = model.transcribe(
        audio_path,
        language=language,
        task="transcribe",
        verbose=False,
        beam_size=5,
        best_of=5,
        temperature=[0.0]
    )

    progress.update(audio_duration)  # after completion, set progress to the end
    progress.close()

    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_path = os.path.join("transcripts", f"{base_name}.txt")
    os.makedirs("transcripts", exist_ok=True)

    print("\nСохраняем результат с таймкодами каждые 3 минуты...\n")

    with open(output_path, "w", encoding="utf-8") as f:
        last_marker = -1
        for segment in result["segments"]:
            current_marker = int(segment["start"] // 180)
            if current_marker != last_marker:
                timestamp = format_time(segment["start"])
                f.write(f"\n[{timestamp}]\n")
                last_marker = current_marker
            f.write(segment["text"].strip() + " ")
            print(f"[{format_time(segment['start'])} --> {format_time(segment['end'])}] {segment['text'].strip()}")

    print(f"\nГотово. Результат сохранён в: {output_path}")

if __name__ == "__main__":
    main()