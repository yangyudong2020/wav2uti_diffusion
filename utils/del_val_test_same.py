import os

val_test_f_folder = "val_test_f/wav"
val_test_folder = "val_test/wav"

# 获取val_test_f文件夹中存在的视频文件列表
val_test_f_videos = os.listdir(val_test_f_folder)

# 遍历val_test_folder中的视频文件
for video_filename in os.listdir(val_test_folder):
    video_path = os.path.join(val_test_folder, video_filename)
    
    # 检查视频是否只在val_test_f_folder中存在
    if video_filename in val_test_f_videos:
        # 删除val_test_f_folder中的视频
        val_test_f_video_path = os.path.join(val_test_f_folder, video_filename)
        os.remove(val_test_f_video_path)
        print(f"Deleted {val_test_f_video_path}")
    else:
        print(f"{video_filename} does not exist in val_test_f_folder, skipping.")

print("Done!")

