import os
import json
import base64
import tempfile
import subprocess
from pathlib import Path
from flask import Flask, request, jsonify
import requests
from openai import OpenAI

app = Flask(__name__)
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

KEYWORD = os.environ.get("STEP_KEYWORD", "cambiamos al siguiente paso")
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
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ar", "16000", "-ac", "1", "-b:a", "64k",
        audio_path
    ], check=True, capture_output=True)


def extract_frames(video_path: str, frames_dir: str, fps: float = 1.0):
    """Extract frames at given fps."""
    Path(frames_dir).mkdir(exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"fps={fps},scale=640:-1",
        "-q:v", "3",
        f"{frames_dir}/frame_%06d.jpg"
    ], check=True, capture_output=True)


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds."""
    result = subprocess.run([
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", video_path
    ], capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            return float(stream.get("duration", 0))
    return 0.0


def transcribe_audio(audio_path: str) -> dict:
    """Transcribe audio with Whisper, getting word-level timestamps."""
    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )
    return response.model_dump()


def find_keyword_timestamps(transcription: dict, keyword: str) -> list[float]:
    """Find timestamps where the keyword appears in the transcription."""
    keyword_lower = keyword.lower().strip()
    timestamps = []
    
    segments = transcription.get("segments", [])
    for seg in segments:
        text = seg.get("text", "").lower()
        if keyword_lower in text:
            # Use the start of the segment as the signal timestamp
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
        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())
    except Exception:
        return {"gesture_detected": False, "confidence": 0.0, "description": "parse error"}


def find_gesture_timestamps(frames_dir: str, fps: float, audio_timestamps: list[float], window: float = 3.0) -> list[float]:
    """
    Only analyze frames near audio keyword timestamps (±window seconds).
    Returns timestamps where both gesture AND keyword coincide.
    """
    frames = sorted(Path(frames_dir).glob("frame_*.jpg"))
    confirmed_timestamps = []
    
    for audio_ts in audio_timestamps:
        # Find frames within the time window of the audio keyword
        best_match = None
        best_confidence = 0.0
        
        for frame in frames:
            # Parse frame number from filename (1-indexed)
            frame_num = int(frame.stem.split("_")[1])
            frame_time = (frame_num - 1) / fps
            
            if abs(frame_time - audio_ts) <= window:
                result = analyze_frame_for_gesture(str(frame))
                if result.get("gesture_detected") and result.get("confidence", 0) > best_confidence:
                    best_confidence = result["confidence"]
                    best_match = frame_time
        
        if best_match is not None:
            confirmed_timestamps.append(best_match)
        else:
            # Audio keyword found but no gesture confirmed — still use audio timestamp
            # (more permissive: one signal is enough)
            confirmed_timestamps.append(audio_ts)
    
    return sorted(set(confirmed_timestamps))


def get_step_frames(frames_dir: str, fps: float, start_time: float, end_time: float, max_frames: int = 3) -> list[str]:
    """
    Get representative frames for a step (evenly distributed).
    Returns list of base64-encoded JPEGs.
    """
    frames = sorted(Path(frames_dir).glob("frame_*.jpg"))
    step_frames = []
    
    for frame in frames:
        frame_num = int(frame.stem.split("_")[1])
        frame_time = (frame_num - 1) / fps
        if start_time <= frame_time < end_time:
            step_frames.append((frame_time, str(frame)))
    
    if not step_frames:
        return []
    
    # Pick evenly spaced frames
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


def build_steps(transcription: dict, split_timestamps: list[float], frames_dir: str, fps: float, duration: float) -> list[dict]:
    """
    Segment transcription and frames into steps based on split timestamps.
    """
    # Build time boundaries: [0, ts1, ts2, ..., end]
    boundaries = [0.0] + sorted(split_timestamps) + [duration]
    
    # Map segments to steps
    segments = transcription.get("segments", [])
    steps = []
    
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        
        # Collect transcription text for this step
        step_text = " ".join(
            seg["text"].strip()
            for seg in segments
            if seg.get("start", 0) >= start and seg.get("start", 0) < end
        ).strip()
        
        # Remove the keyword phrase from the beginning of steps (it's a signal, not content)
        keyword_lower = KEYWORD.lower()
        if step_text.lower().startswith(keyword_lower):
            step_text = step_text[len(keyword_lower):].strip().lstrip(",. ")
        
        # Get representative frames
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
    return jsonify({"status": "ok"})


@app.route("/process-video", methods=["POST"])
def process_video():
    """
    Main endpoint. Expects JSON:
    {
        "video_url": "https://...",
        "keyword": "cambiamos al siguiente paso",  // optional
        "fps": 1.0  // optional, frames per second to extract
    }
    Returns:
    {
        "steps": [
            {
                "step_number": 1,
                "start_time": 0.0,
                "end_time": 45.2,
                "text": "...",
                "frames_base64": ["..."]
            }
        ],
        "total_steps": N,
        "duration": 180.0
    }
    """
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
        
        # 1. Download video
        try:
            download_video(video_url, video_path)
        except Exception as e:
            return jsonify({"error": f"Download failed: {str(e)}"}), 400
        
        # 2. Get duration
        duration = get_video_duration(video_path)
        
        # 3. Extract audio and frames in parallel (sequential here for simplicity)
        try:
            extract_audio(video_path, audio_path)
            extract_frames(video_path, frames_dir, fps)
        except subprocess.CalledProcessError as e:
            return jsonify({"error": f"ffmpeg error: {e.stderr.decode()}"}), 500
        
        # 4. Transcribe audio
        try:
            transcription = transcribe_audio(audio_path)
        except Exception as e:
            return jsonify({"error": f"Transcription failed: {str(e)}"}), 500
        
        # 5. Find keyword timestamps in audio
        audio_timestamps = find_keyword_timestamps(transcription, keyword)
        
        # 6. Confirm with gesture detection (only checks frames near keyword timestamps)
        confirmed_timestamps = find_gesture_timestamps(frames_dir, fps, audio_timestamps)
        
        # 7. Build steps
        steps = build_steps(transcription, confirmed_timestamps, frames_dir, fps, duration)
        
        return jsonify({
            "steps": steps,
            "total_steps": len(steps),
            "duration": round(duration, 1),
            "split_timestamps": confirmed_timestamps,
            "full_transcript": transcription.get("text", "")
        })
