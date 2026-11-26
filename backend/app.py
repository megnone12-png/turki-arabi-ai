
import os
import tempfile
import subprocess
import asyncio
import gradio as gr
import edge_tts
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
from pydub import AudioSegment


translator = GoogleTranslator(source="tr", target="ar")
model = None


def get_model():
    global model
    if model is None:
        model = WhisperModel("tiny", device="cpu", compute_type="int8")
    return model


async def make_arabic_voice(text, output_path):
    voice = edge_tts.Communicate(
        text,
        voice="ar-SA-HamedNeural"
    )
    await voice.save(output_path)


def dub_video(video):
    if video is None:
        raise gr.Error("يرجى اختيار فيديو أولاً")

    work = tempfile.mkdtemp()
    source = os.path.join(work, "source.mp4")
    audio = os.path.join(work, "original.wav")
    dubbed_audio = os.path.join(work, "arabic.wav")
    output = os.path.join(work, "dubbed.mp4")

    with open(video, "rb") as f:
        with open(source, "wb") as out:
            out.write(f.read())

    duration = float(subprocess.check_output([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        source
    ]).decode().strip())

    if duration > 600:
        raise gr.Error("للتجربة الأولى اختر فيديو لا يتجاوز 10 دقائق")

    subprocess.run([
        "ffmpeg", "-y", "-i", source,
        "-vn", "-ac", "1", "-ar", "16000", audio
    ], check=True)

    whisper = get_model()
    segments, info = whisper.transcribe(audio, language="tr")

    original = AudioSegment.silent(duration=int(duration * 1000))

    for index, segment in enumerate(segments):
        text = segment.text.strip()
        if not text:
            continue

        arabic = translator.translate(text)
        voice_file = os.path.join(work, f"voice_{index}.mp3")

        import nest_asyncio
        nest_asyncio.apply()
        asyncio.get_event_loop().run_until_complete(make_arabic_voice(arabic, voice_file))

        speech = AudioSegment.from_file(voice_file)
        speech = speech[:int((segment.end - segment.start) * 1000)]
        original = original.overlay(
            speech,
            position=int(segment.start * 1000)
        )

    original.export(dubbed_audio, format="wav")

    subprocess.run([
        "ffmpeg", "-y",
        "-i", source,
        "-i", dubbed_audio,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        output
    ], check=True)

    return output


app = gr.Interface(
    fn=dub_video,
    inputs=gr.Video(label="اختر الحلقة التركية"),
    outputs=gr.Video(label="الحلقة بالدبلجة العربية"),
    title="تركي بالعربي AI",
    description="نسخة تجريبية لدبلجة المقاطع التركية القصيرة إلى العربية"
)

app.launch()
