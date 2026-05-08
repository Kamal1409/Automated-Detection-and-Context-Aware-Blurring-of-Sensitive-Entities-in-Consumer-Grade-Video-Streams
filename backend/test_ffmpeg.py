import sys
import os

sys.path.append("d:/Projects/GDSH/SDP/backend")
import video_processing

ffmpeg_exe = video_processing._resolve_ffmpeg_command()
print("FFMPEG:", ffmpeg_exe)

if ffmpeg_exe:
    import subprocess
    result = subprocess.run([ffmpeg_exe, "-version"], capture_output=True, text=True)
    print("FFMPEG VERSION:", result.stdout[:100])
