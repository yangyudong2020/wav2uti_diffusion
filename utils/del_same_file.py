import os

def delete_duplicate_files(directory):
    file_dict = {}
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_name = os.path.basename(file)
            
            if file_name in file_dict:
                print(f"Deleting duplicate file: {file_path}")
                os.remove(file_path)
            else:
                file_dict[file_name] = file_path

if __name__ == "__main__":
    target_directory = "/val_test/video"
    delete_duplicate_files(target_directory)

