import os

# 获取val_test/wav目录下的文件列表
wav_files = os.listdir('val_test/wav')
#print(wav_files)
# 获取val_test/video目录下的.avi文件列表
video_files = [filename for filename in os.listdir('val_test/txt') if filename.endswith('.txt')]
print(len(video_files))
# 找到匹配的文件名
matching_files = [filename for filename in wav_files if filename[:-4] + '.avi' in video_files]

#print(len(matching_files))
# 删除没有匹配的文件
for filename in video_files:
    print(filename[:-4] + '.wav')
    if filename[:-4] + '.wav' not in wav_files:
        print(filename)
        os.remove(os.path.join('val_test/txt', filename))

print("匹配的文件已保存，未匹配的文件已删除。")

