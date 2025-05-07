# Make sure you have these installed:
# pip install gtts pydub

from gtts import gTTS
from pydub import AudioSegment
import io

def text_to_wav(text: str, wav_path: str, lang: str = 'en', slow: bool = False):
    """
    Convert text to a WAV file using gTTS (MP3) + pydub.

    Args:
        text: The text you want to synthesize.
        wav_path: Output path for the WAV file (e.g. "output.wav").
        lang: Language code (default 'en').
        slow: If True, speech will be slower.
    """
    # 1. Generate MP3 in-memory
    tts = gTTS(text=text, lang=lang, slow=slow)
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)

    # 2. Load MP3 from memory and export as WAV
    audio = AudioSegment.from_file(mp3_fp, format="mp3")
    audio.export(wav_path, format="wav")
    print(f"WAV file saved to: {wav_path}")

# Example usage:
if __name__ == "__main__":
    sample_text = "Hello! This is a test of gTTS converting text to WAV audio."
    text_to_wav(sample_text, "examples/audio2.wav")