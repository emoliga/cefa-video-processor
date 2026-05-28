import os
import json
import base64
import tempfile
import subprocess
import threading
from pathlib import Path
from flask import Flask, request, jsonify
import requests
from openai import OpenAI
import imageio_ffmpeg

app = Flask(__name__)
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def _run_ffmpeg(cmd: list) -> int:
    """Run ffmpeg in a thread to avoid gunicorn worker signal conflicts."""
    result = {"returncode": None}

    def target():
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                 close_fds=True)
        result["returncode"] = proc.wait()

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout=540)  # 9 min max per ffmpeg call
    if t.is_alive():
        raise TimeoutError("ffmpeg timed out after 540s")
    return result["returncode"]

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
    """Download video from URL. Handles Google Drive large file confirmation."""
    import re
    session = requests.Session()

    r = session.get(url, stream=True, timeout=120)
    r.raise_for_status()

    # Google Drive returns an HTML confirmation page for large files
    content_type = r.headers.get("Content-Type", "")
    if "text/html" in content_type:
        chunk = next(r.iter_content(chunk_size=32768), b"")
        text = chunk.decode("utf-8", errors="ignore")

        # Try confirm token pattern
        match = re.search(r'confirm=([^&"]+)', text)
        if match:
            confirm_token = match.group(1)
            file_id = re.search(r'[?&]id=([^&]+)', url)
            if file_id:
                url = f"https://drive.google.com/uc?export=download&confirm={confirm_token}&id={file_id.group(1)}"
            else:
                url = url + f"&confirm={confirm_token}"
        else:
            # Newer Drive flow
            file_id_match = re.search(r'[?&]id=([^&]+)', url)
            if file_id_match:
                fid = file_id_match.group(1)
                url = f"https://drive.usercontent.google.com/download?id={fid}&export=download&confirm=t"

        r = session.get(url, stream=True, timeout=300)
        r.raise_for_status()

    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)

    size = os.path.getsize(dest)
    if size < 100_000:
        raise ValueError(f"Downloaded file too small ({size} bytes) - likely not a valid video or Drive link is not public")

    return True


def compress_video(video_path: str) -> str:
    """
    Recompress video to 720p max, CRF 28, before processing.
    Returns path to compressed file (replaces original).
    """
    compressed = video_path.replace(".mp4", "_compressed.mp4")
    # Use Popen instead of run to avoid gunicorn worker timeout killing the process
    rc = _run_ffmpeg([
        FFMPEG, "-y", "-i", video_path,
        "-vf", "scale='min(1280,iw)':'min(720,ih)':force_original_aspect_ratio=decrease,pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264", "-crf", "28", "-preset", "veryfast",
        "-c:a", "aac", "-b:a", "64k",
        "-movflags", "+faststart",
        compressed
    ])
    if rc != 0:
        raise subprocess.CalledProcessError(rc, FFMPEG)
    os.remove(video_path)
    os.rename(compressed, video_path)
    return video_path


def extract_audio(video_path: str, audio_path: str):
    """Extract audio as mp3 using ffmpeg."""
    rc = _run_ffmpeg([
        FFMPEG, "-y", "-i", video_path,
        "-vn", "-ar", "16000", "-ac", "1", "-b:a", "64k",
        audio_path
    ])
    if rc != 0:
        raise subprocess.CalledProcessError(rc, FFMPEG)


def extract_frames(video_path: str, frames_dir: str, fps: float = 1.0):
    """Extract frames at given fps, capped at 300 frames to avoid OOM."""
    Path(frames_dir).mkdir(exist_ok=True)
    rc = _run_ffmpeg([
        FFMPEG, "-y", "-i", video_path,
        "-vf", f"fps={fps},scale=480:-2",
        "-q:v", "5",
        "-frames:v", "300",
        f"{frames_dir}/frame_%06d.jpg"
    ])
    if rc != 0:
        raise subprocess.CalledProcessError(rc, FFMPEG)


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
    """Transcribe audio with Whisper, getting word and segment-level timestamps."""
    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"]
        )
    result = response.model_dump()
    # Free disk space immediately
    try:
        os.remove(audio_path)
    except Exception:
        pass
    return result


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


def _clean_keyword(text: str) -> str:
    """Remove keyword phrase from text (start or anywhere)."""
    kw = KEYWORD.lower()
    t = text.strip()
    tl = t.lower()
    if kw in tl:
        idx = tl.find(kw)
        # Remove everything from keyword onwards (it marks end of step)
        t = t[:idx].strip().rstrip(",. ")
    return t


def build_steps(transcription: dict, split_timestamps: list, frames_dir: str, fps: float, duration: float) -> list:
    """Segment transcription and frames into steps based on split timestamps."""
    boundaries = [0.0] + sorted(split_timestamps) + [duration]
    keyword_lower = KEYWORD.lower()
    steps = []

    # Try word-level timestamps first (more precise)
    words = transcription.get("words", [])

    if words:
        # Use word-level: assign each word to a step based on its timestamp
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]

            step_words = [
                w["word"] for w in words
                if w.get("start", 0) >= start and w.get("start", 0) < end
            ]
            step_text = " ".join(step_words).strip()
            step_text = _clean_keyword(step_text)

            frames_b64 = get_step_frames(frames_dir, fps, start, end)

            if step_text or frames_b64:
                steps.append({
                    "step_number": i + 1,
                    "start_time": round(start, 1),
                    "end_time": round(end, 1),
                    "text": step_text,
                    "frames_base64": frames_b64
                })
    else:
        # Fallback: use segments. If a segment spans a boundary, split it proportionally.
        segments = transcription.get("segments", [])
        full_text = transcription.get("text", "")

        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]

            step_parts = []
            for seg in segments:
                seg_start = seg.get("start", 0)
                seg_end = seg.get("end", seg_start + 1)
                seg_text = seg.get("text", "").strip()

                if seg_end <= start or seg_start >= end:
                    continue  # outside window

                if keyword_lower in seg_text.lower():
                    # This segment contains the keyword - take only the part before it
                    kw_idx = seg_text.lower().find(keyword_lower)
                    if seg_start >= start:
                        # Segment starts in this step - take text before keyword
                        part = seg_text[:kw_idx].strip()
                        if part:
                            step_parts.append(part)
                    # Don't include the keyword or anything after it
                else:
                    # Full segment in window
                    if seg_start >= start:
                        step_parts.append(seg_text)
                    else:
                        # Segment started before this window - skip (already counted)
                        pass

            step_text = " ".join(step_parts).strip()
            step_text = _clean_keyword(step_text)

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
            # Recompress to 720p to reduce RAM usage during frame extraction
            compress_video(video_path)
            extract_audio(video_path, audio_path)
            extract_frames(video_path, frames_dir, fps)
            # Delete compressed video immediately to free memory
            os.remove(video_path)
        except subprocess.CalledProcessError as e:
            stderr_text = e.stderr.decode() if e.stderr else "no stderr"
            return jsonify({"error": f"ffmpeg error: {stderr_text}"}), 500

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
