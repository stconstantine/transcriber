import os
import sys
# types, pytest не используются
from unittest import mock

# Импортируем модуль как объект, чтобы можно было тестировать main
import importlib.util

SPEC_PATH = os.path.join(os.path.dirname(__file__), '../whisper_transcribe.py')
spec = importlib.util.spec_from_file_location("whisper_transcribe", SPEC_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load spec for {SPEC_PATH}")
whisper_transcribe = importlib.util.module_from_spec(spec)
sys.modules["whisper_transcribe"] = whisper_transcribe
spec.loader.exec_module(whisper_transcribe)

def test_format_time():
    assert whisper_transcribe.format_time(0) == "00:00:00"
    assert whisper_transcribe.format_time(65) == "00:01:05"
    assert whisper_transcribe.format_time(3661) == "01:01:01"

def test_main_file_not_found(capsys):  # type: ignore
    test_args = ["prog", "--audio", "not_exist.wav"]
    with mock.patch.object(sys, 'argv', test_args):
        whisper_transcribe.main()
        captured = capsys.readouterr()
        assert "Ошибка: файл не найден" in captured.out

def test_main_model_load_error(capsys):  # type: ignore
    test_args = ["prog", "--model", "nonexistent_model"]
    with mock.patch.object(sys, 'argv', test_args):
        with mock.patch("whisper.load_model", side_effect=Exception("fail")):
            whisper_transcribe.main()
            captured = capsys.readouterr()
            assert "Ошибка загрузки модели" in captured.out

def test_main_audio_load_error(tmp_path, capsys):  # type: ignore
    # Создаём пустой файл, чтобы пройти проверку существования
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"")
    test_args = ["prog", "--audio", str(audio_path)]
    with mock.patch.object(sys, 'argv', test_args):
        with mock.patch("whisper.load_model", return_value=mock.Mock()):
            with mock.patch("whisper.load_audio", side_effect=Exception("fail")):
                whisper_transcribe.main()
                captured = capsys.readouterr()
                assert "Ошибка загрузки аудио" in captured.out

def test_main_transcribe_error(tmp_path, capsys):  # type: ignore
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fake")
    test_args = ["prog", "--audio", str(audio_path)]
    mock_model = mock.Mock()
    mock_model.is_multilingual = False
    mock_model.transcribe.side_effect = Exception("fail")
    with mock.patch.object(sys, 'argv', test_args):
        with mock.patch("whisper.load_model", return_value=mock_model):
            with mock.patch("whisper.load_audio", return_value=[0]*100):
                with mock.patch("whisper.audio.SAMPLE_RATE", 1):
                    whisper_transcribe.main()
                    captured = capsys.readouterr()
                    assert "Ошибка при транскрипции" in captured.out
