import os

def rename_gif_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".gif") and ".txt" in filename:
            new_filename = filename.replace(".txt", "")
            source_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)
            os.rename(source_path, new_path)
            print(f"Renamed {filename} to {new_filename}")

output_directory = "output_mul_64"  # 替换为你的实际输出目录路径
rename_gif_files(output_directory)


output_directory = "output_sp_64"  # 替换为你的实际输出目录路径
rename_gif_files(output_directory)



