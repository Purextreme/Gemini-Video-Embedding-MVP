import os
import subprocess
from pathlib import Path

def convert_videos(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not output_path.exists():
        output_path.mkdir(parents=True)
        
    extensions = ('.mp4', '.mov', '.MOV', '.avi', '.mkv', '.webm')
    
    files = [f for f in input_path.iterdir() if f.suffix.lower() in extensions]
    print(f"Found {len(files)} videos to convert.")
    
    for video_file in files:
        output_file = output_path / (video_file.stem + ".mp4")
        
        # Skip if file already exists and is valid (> 1KB)
        if output_file.exists() and output_file.stat().st_size > 1024:
            print(f"Skipping {video_file.name} (already exists and valid)")
            continue
            
        print(f"Converting {video_file.name} -> {output_file.name}...")
        
        # ffmpeg command for 720p low quality
        # -i: input
        # -vf scale=-1:720: resize to 720p height, maintain aspect ratio
        # -c:v libx264: h264 codec
        # -crf 28: lower quality (range 0-51, 23 is default)
        # -preset faster: faster encoding
        # -c:a aac -b:a 128k: audio codec and bitrate
        # -y: overwrite output
        
        cmd = [
            'ffmpeg',
            '-i', str(video_file),
            '-vf', 'scale=-2:720', # -2 ensures width is even (required by some codecs)
            '-c:v', 'libx264',
            '-crf', '28',
            '-preset', 'faster',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-y',
            str(output_file)
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"Successfully converted {video_file.name}")
        except subprocess.CalledProcessError as e:
            print(f"Error converting {video_file.name}: {e.stderr.decode()}")

if __name__ == "__main__":
    convert_videos("test_video", "test_video_720")
