#!/usr/bin/env python

import os
import whisper
from tqdm import tqdm
import argparse
from typing import TypedDict, List
import time

class Segment(TypedDict):
    start: float
    end: float
    text: str

class TranscriptionResult(TypedDict):
    text: str
    segments: List[Segment]


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

    model = whisper.load_model(model_name)

    # Загружаем аудио для получения длительности
    audio = whisper.load_audio(audio_path)
    audio_duration = len(audio) / whisper.audio.SAMPLE_RATE

    print(f"Распознавание начато: {audio_path} (язык: {language}, длительность: {audio_duration:.1f} сек)")

    # Прогресс-бар по времени (секундам)
    progress = tqdm(total=audio_duration, unit="sec", dynamic_ncols=True)

    # Здесь создадим коллбек для обновления прогресса вручную
    # Whisper не дает встроенного коллбека, поэтому будем имитировать прогресс построчно

    start_time = time.time()
    # Распознаем всю аудио
    result: TranscriptionResult = model.transcribe(  # type: ignore
        audio_path,
        language=language,
        task="transcribe",
        verbose=False,
        beam_size=5,
        best_of=5,
        temperature=0.0
    )
    end_time = time.time()
    processing_time = end_time - start_time

    progress.update(audio_duration)  # after completion, set progress to the end
    progress.close()

    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_path = os.path.join("transcripts", f"{base_name}.txt")
    os.makedirs("transcripts", exist_ok=True)

    print("\nСохраняем результат с таймкодами каждые 3 минуты...\n")

    with open(output_path, "w", encoding="utf-8") as f:
        last_marker = -1
        for segment in result.get("segments", []):
            start = float(segment["start"])
            text = str(segment["text"]).strip()
            current_marker = int(start // 180)
            if current_marker != last_marker:
                timestamp = format_time(start)
                f.write(f"\n[{timestamp}]\n")
                last_marker = current_marker
            f.write(text + " ")

    input_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    output_size_bytes = os.path.getsize(output_path)
    output_size_kb = output_size_bytes / 1024
    output_characters = sum(len(s["text"]) for s in result["segments"])
    frames_processed = len(audio)
    audio_seconds = audio_duration

    print("\nСтатистика обработки:")
    print(f"Входной файл: {audio_path}")
    print(f"  Размер: {audio_duration:.1f} сек, {input_size_mb:.2f} МБ")
    print(f"Выходной файл: {output_path}")
    print(f"  Размер: {output_characters} символов, {output_size_kb:.2f} КБ")
    print(f"Обработано фреймов: {frames_processed}")
    print(f"Время обработки: {processing_time:.2f} сек")
    print(f"Скорость: {frames_processed / processing_time:.1f} фрейм/с, "
          f"{audio_seconds / processing_time:.2f} сек аудио/с, "
          f"{output_characters / processing_time:.1f} символов/с")

    print(f"\nГотово. Результат сохранён в: {output_path}")

    print(f"Модель: {model_name}")
    print(f"Параметры модели: {str(model.dims)}")
    print(f"Язык: {language}")

if __name__ == "__main__":
    main()