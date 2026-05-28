import os
import json
import base64
import tempfile
import subprocess
from pathlib import Path
from flask import Flask, request, jsonify
import requests
from openai import OpenAI
import imageio_ffmpeg

app = Flask(__name__)
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

KEYWORD = os.environ.get("STEP_KEYWORD", "cambiamos al siguiente paso")

# Use ffmpeg binary bundled with imageio-ffmpeg (no system ffmpeg needed)
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
FFPROBE = FFMPEG.replace("ffmpeg", "ffprobe")

GESTURE_PROMPT = """Analyze this image and determine if a person is showing a "rock" hand gesture:
- Index finger pointing up OR index + pinky extended
- Middle and ring fingers folded down
- Palm facing the camera
- Thumb may be extended or tucked

Respond ONLY with valid JSON, no markdown:
{"gesture_detected": true/false, "confidence": 0.0-1.0, "description": "brief description"}"""


def download_video(url: str, dest: str) -> bool:
    """Download video from URL (supports Supabase signed URLs)."""
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    return True


def extract_audio(video_path: str, audio_path: str):
    """Extract audio as mp3 using ffmpeg."""
    subprocess.run([
        FFMPEG, "-y", "-i", video_path,
        "-vn", "-ar", "16000", "-ac", "1", "-b:a", "64k",
        audio_path
    ], check=True, capture_output=True)


def extract_frames(video_path: str, frames_dir: str, fps: float = 1.0):
    """Extract frames at given fps."""
    Path(frames_dir).mkdir(exist_ok=True)
    subprocess.run([
        FFMPEG, "-y", "-i", video_path,
        "-vf", f"fps={fps},scale=640:-1",
        "-q:v", "3",
        f"{frames_dir}/frame_%06d.jpg"
    ], check=True, capture_output=True)


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds."""
    # ffprobe may not be bundled; fall back to ffmpeg stream info
    try:
        result = subprocess.run([
            FFMPEG, "-i", video_path
        ], capture_output=True, text=True)
        # ffmpeg prints duration to stderr even on "error"
        for line in result.stderr.splitlines():
            if "Duration:" in line:
                parts = line.strip().split("Duration:")[1].split(",")[0].strip()
                h, m, s = parts.split(":")
                return float(h) * 3600 + float(m) * 60 + float(s)
    except Exception:
        pass
    return 0.0


def transcribe_audio(audio_path: str) -> dict:
    """Transcribe audio with Whisper, getting segment-level timestamps."""
    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )
    return response.model_dump()


def find_keyword_timestamps(transcription: dict, keyword: str) -> list:
    """Find timestamps where the keyword appears in the transcription."""
    keyword_lower = keyword.lower().strip()
    timestamps = []
    segments = transcription.get("segments", [])
    for seg in segments:
        text = seg.get("text", "").lower()
        if keyword_lower in text:
            timestamps.append(seg.get("start", 0.0))
    return timestamps


def analyze_frame_for_gesture(frame_path: str) -> dict:
    """Use GPT-4o Vision to detect the rock gesture in a frame."""
    with open(frame_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{img_b64}",
                    "detail": "low"
                }},
                {"type": "text", "text": GESTURE_PROMPT}
            ]
        }]
    )

    try:
        text = response.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())
    except Exception:
        return {"gesture_detected": False, "confidence": 0.0, "description": "parse error"}


def find_gesture_timestamps(frames_dir: str, fps: float, audio_timestamps: list, window: float = 3.0) -> list:
    """
    Only analyze frames near audio keyword timestamps (±window seconds).
    Returns confirmed timestamps where gesture was detected, or falls back to audio-only.
    """
    frames = sorted(Path(frames_dir).glob("frame_*.jpg"))
    confirmed_timestamps = []

    for audio_ts in audio_timestamps:
        best_match = None
        best_confidence = 0.0

        for frame in frames:
            frame_num = int(frame.stem.split("_")[1])
            frame_time = (frame_num - 1) / fps

            if abs(frame_time - audio_ts) <= window:
                result = analyze_frame_for_gesture(str(frame))
                if result.get("gesture_detected") and result.get("confidence", 0) > best_confidence:
                    best_confidence = result["confidence"]
                    best_match = frame_time

        # Use gesture timestamp if found, otherwise fall back to audio timestamp
        confirmed_timestamps.append(best_match if best_match is not None else audio_ts)

    return sorted(set(confirmed_timestamps))


def get_step_frames(frames_dir: str, fps: float, start_time: float, end_time: float, max_frames: int = 3) -> list:
    """Get representative frames for a step (evenly distributed). Returns list of base64 JPEGs."""
    frames = sorted(Path(frames_dir).glob("frame_*.jpg"))
    step_frames = []

    for frame in frames:
        frame_num = int(frame.stem.split("_")[1])
        frame_time = (frame_num - 1) / fps
        if start_time <= frame_time < end_time:
            step_frames.append((frame_time, str(frame)))

    if not step_frames:
        return []

    if len(step_frames) <= max_frames:
        selected = step_frames
    else:
        step = len(step_frames) // max_frames
        selected = step_frames[::step][:max_frames]

    result = []
    for _, path in selected:
        with open(path, "rb") as f:
            result.append(base64.b64encode(f.read()).decode())
    return result


def build_steps(transcription: dict, split_timestamps: list, frames_dir: str, fps: float, duration: float) -> list:
    """Segment transcription and frames into steps based on split timestamps."""
    boundaries = [0.0] + sorted(split_timestamps) + [duration]
    segments = transcription.get("segments", [])
    steps = []

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]

        step_text = " ".join(
            seg["text"].strip()
            for seg in segments
            if seg.get("start", 0) >= start and seg.get("start", 0) < end
        ).strip()

        keyword_lower = KEYWORD.lower()
        if step_text.lower().startswith(keyword_lower):
            step_text = step_text[len(keyword_lower):].strip().lstrip(",. ")

        frames_b64 = get_step_frames(frames_dir, fps, start, end)

        if step_text or frames_b64:
            steps.append({
                "step_number": i + 1,
                "start_time": round(start, 1),
                "end_time": round(end, 1),
                "text": step_text,
                "frames_base64": frames_b64
            })

    return steps


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "ffmpeg": FFMPEG})


@app.route("/process-video", methods=["POST"])
def process_video():
    data = request.get_json()
    if not data or "video_url" not in data:
        return jsonify({"error": "video_url required"}), 400

    video_url = data["video_url"]
    keyword = data.get("keyword", KEYWORD)
    fps = float(data.get("fps", 1.0))

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "input.mp4")
        audio_path = os.path.join(tmpdir, "audio.mp3")
        frames_dir = os.path.join(tmpdir, "frames")

        try:
            download_video(video_url, video_path)
        except Exception as e:
            return jsonify({"error": f"Download failed: {str(e)}"}), 400

        duration = get_video_duration(video_path)

        try:
            extract_audio(video_path, audio_path)
            extract_frames(video_path, frames_dir, fps)
        except subprocess.CalledProcessError as e:
            return jsonify({"error": f"ffmpeg error: {e.stderr.decode()}"}), 500

        try:
            transcription = transcribe_audio(audio_path)
        except Exception as e:
            return jsonify({"error": f"Transcription failed: {str(e)}"}), 500

        audio_timestamps = find_keyword_timestamps(transcription, keyword)
        confirmed_timestamps = find_gesture_timestamps(frames_dir, fps, audio_timestamps)
        steps = build_steps(transcription, confirmed_timestamps, frames_dir, fps, duration)

        return jsonify({
            "steps": steps,
            "total_steps": len(steps),
            "duration": round(duration, 1),
            "split_timestamps": confirmed_timestamps,
            "full_transcript": transcription.get("text", "")
        })
