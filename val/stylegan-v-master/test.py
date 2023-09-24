import torch
from src.metrics import frechet_video_distance

# Create instances of VideoFramesFolderDataset for real, fake, and generated videos
real_video_dataset = VideoFramesFolderDataset(path='/mnt/shareEEx/yangyudong/utlnet/video-pytorch/fid/DNN/images/real', cfg=your_config_for_real_videos)
fake_video_dataset = VideoFramesFolderDataset(path='/mnt/shareEEx/yangyudong/utlnet/video-pytorch/fid/DNN/images/fake', cfg=your_config_for_fake_videos)
generated_video_dataset = VideoFramesFolderDataset(path='/mnt/shareEEx/yangyudong/utlnet/video-pytorch/fid/DNN/videos', cfg=your_config_for_generated_videos)

# Example function to compute FVD
def compute_fvd(real_videos, fake_videos):
    # Prepare real and fake videos
    real_videos = torch.tensor(real_videos)  # Assuming real_videos is a list of video frames
    fake_videos = torch.tensor(fake_videos)  # Assuming fake_videos is a list of video frames

    # Compute FVD
    fvd = frechet_video_distance.compute_fvd(real_videos, fake_videos)
    return fvd

# Load real and fake videos using the datasets
real_videos = []
fake_videos = []

for idx in range(len(real_video_dataset)):
    real_data = real_video_dataset[idx]
    real_videos.append(real_data['image'])

for idx in range(len(fake_video_dataset)):
    fake_data = fake_video_dataset[idx]
    fake_videos.append(fake_data['image'])

# Compute FVD
fvd_score = compute_fvd(real_videos, fake_videos)
print("FVD score:", fvd_score)

