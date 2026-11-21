# ViralClip

Simple 9:16 vertical YouTube viral clipper powered by Gemini AI + yt-dlp + FFmpeg.

## How to Run
Double-click `start.bat` (or run `python app.py`), then open http://127.0.0.1:8765.

## How It Works
1. Paste your **Gemini API Key** once at the top right (saved locally in `.env`).
2. Paste any **YouTube VOD** (even 3–4 hour streams) or **currently running live stream** link and click **Load Stream**.
3. The app grabs a real frame from the stream and auto-detects the **Facecam** box (drag/resize to fine-tune if needed).
4. Pick your vertical 9:16 layout:
   - **Facecam Top / Gameplay Bottom**
   - **Gameplay Top / Facecam Bottom**
5. Click **Find & Clip Viral Parts**. Gemini reviews the stream (or past DVR time of a live stream) and clips **only** the viral moments into ready-to-post `1080x1920` MP4s saved in `clips/`.
