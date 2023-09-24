import os
import wave
import contextlib

def calculate_total_duration(folder_path):
    total_duration = 0.0

    # 遍历文件夹中的所有文件
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                with contextlib.closing(wave.open(file_path, 'r')) as f:
                    frames = f.getnframes()
                    rate = f.getframerate()
                    duration = frames / float(rate)
                    total_duration += duration

    return total_duration

folder_path = 'wavDir'
total_duration = calculate_total_duration(folder_path)
print(f'文件夹 {folder_path} 下所有.wav文件的总时长为 {total_duration} 秒')

