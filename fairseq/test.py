import torch
from transformers import BertModel
import torch.nn as nn
import soundfile as sf
import fairseq
from transformers import BertTokenizer, BertModel
import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import BertTokenizer
import time
import torchvision.transforms.functional as TF
from PIL import Image
import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets
from imagen_pytorch import Unet, Imagen, ImagenTrainer
from torchvision import transforms
import imageio
from imagen_pytorch import Unet3D, ElucidatedImagen, ImagenTrainer
import torch.nn as nn
import librosa
import torch
import numpy as np
import soundfile as sf
import fairseq

MAX_FRAME = 15

cp_path = 'xlsr_53_56k.pt'
model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
model = model[0]
model = model.to(torch.float)  # Convert the model's weights to float
model.eval()
tokens = BertTokenizer.from_pretrained('bert-base-chinese')

text_content = '你好，我是小艾'
# 文本部分
text_input = tokens(text_content, return_tensors='pt', max_length=128, padding='max_length', truncation=True)

bert_model = BertModel.from_pretrained('bert-base-chinese')

# 获取BERT中间输出
with torch.no_grad():
    bert_output = bert_model(**text_input)['last_hidden_state']  # (batch_size, sequence_length, hidden_size)

#input audio
audio_input, sample_rate = sf.read("speaker00054_M_s1_stn00144.wav")
audio_input_tensor = torch.tensor(audio_input, dtype=torch.float).unsqueeze(0)  # Convert input to float

output = model.feature_extractor(audio_input_tensor)

# 调整z的维度，使其与bert_output的维度匹配
output_adjusted = output.transpose(1, 2)  # 或者 z.permute(0, 2, 1)

# 调整z的维度后，使用线性层调整尺寸
linear_bert = nn.Linear(768, 512)
bert_output_adjusted = linear_bert(bert_output)  # 调整为与z_adjusted的尺寸相同
# Concatenate the two feature representations along the feature dimension
merged_features = torch.cat((bert_output_adjusted, output_adjusted), dim=1)  # Shape: (batch_size, seq_length1 + seq_length2, hidden_size/feature_dim)

# Apply average pooling along the sequence dimension to get a fixed-length representation
fixed_length_features = torch.mean(merged_features, dim=1)  # Shape: (batch_size, hidden_size/feature_dim)

print(fixed_length_features.shape)


class CustomDataset(Dataset):
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((112, 112)),
        ])
        self.file_list = self._get_file_list()

    def _get_file_list(self):
        file_list = []
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.avi'):
                    file_path = os.path.join(root, file)
                    file_list.append(file_path)
        return file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_path = self.file_list[index]
        video_frames = self.read_video_frames(file_path)
        input_ids = None

        text_path = file_path.split(".")[0] + str(".txt")
        file_name = file_path.split("/")[-1].split(".")[0]
        if text_path.endswith('.txt'):
            with open(text_path, 'r', encoding='gbk') as file:
                content = file.read().strip()
            text_input = tokens(content, return_tensors='pt', max_length=128, padding='max_length', truncation=True)
            # 加载音频文件
            audio_file = '/mnt/shareEEx/yangyudong/utlnet/video_pytorch/wavDir/' + file_name + '.wav'
            audio, sample_rate = librosa.load(audio_file, sr=None)  # sr=None 保持原始采样率
            audio_input_tensor = torch.tensor(audio, dtype=torch.float).unsqueeze(0)  # Convert input to float
            wav2vec_output = wav2vec_model.feature_extractor(audio_input_tensor)
            wav2vec_output_adjusted = wav2vec_output.transpose(1, 2)
            linear_bert = nn.Linear(768, 512)

            bert_model = BertModel.from_pretrained('bert-base-chinese')
            # 获取BERT中间输出
            with torch.no_grad():
                bert_output = bert_model(**text_input)[
                    'last_hidden_state'].detach() # (batch_size, sequence_length, hidden_size)
            bert_output_adjusted = linear_bert(bert_output)
            merged_features = torch.cat((bert_output_adjusted, output_adjusted),
                                        dim=1)  # Shape: (batch_size, seq_length1 + seq_length2, hidden_size/feature_dim)
            input_ids = torch.mean(merged_features, dim=1)  # Shape: (batch_size, hidden_size/feature_dim)

        if input_ids is None:
            input_ids = torch.empty(0)  # Set to empty PyTorch tensor
        input_ids = input_ids.to(torch.float32)

        return video_frames, input_ids

    def read_video_frames(self, file_path):
        cap = cv2.VideoCapture(file_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame = self.transform(frame)
                frames.append(frame)
            else:
                break
        cap.release()

        if len(frames) == 0:
            print(f"error {file_path}")

        # Check if frames length exceeds the maximum limit
        if len(frames) > MAX_FRAME:
            frames = frames[:MAX_FRAME]  # Truncate frames to the maximum limit

        # Find the maximum length of frames
        max_length = max([frame.shape[0] for frame in frames])

        # Pad frames to the maximum length
        padded_frames = []
        for frame in frames:
            padding = max_length - frame.shape[0]
            padded_frame = F.pad(frame, (0, 0, 0, padding))
            padded_frames.append(padded_frame)

        # Perform additional padding if necessary to reach the maximum limit
        if len(padded_frames) < MAX_FRAME:
            padding = MAX_FRAME - len(padded_frames)
            padded_frames.extend([torch.zeros_like(padded_frames[0]) for _ in range(padding)])

        video_frames = torch.stack(padded_frames)
        video_frames = video_frames.permute(1, 0, 2, 3)  # Adjust the dimensions
        return video_frames


root_dir = "/mnt/shareEEx/yangyudong/utlnet/video_pytorch/data_frame20"

dataset = CustomDataset(root_dir)

print(dataset[0][0].shape, dataset[0][0].dtype, dataset[0][1].dtype, dataset[0][1].shape)


experiment_path = os.path.join("experiments", "conditional_video_utl_diffusion")
images_path = os.path.join(experiment_path, "video_output")
os.makedirs(images_path, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



audio_file = 'wavDir/speaker00012_M_s1_stn00096.wav'
audio, sr = librosa.load(audio_file, sr=None) # sr=None 保持原始采样率

text_content = '人们文化教育程度的增长等等'
# 文本部分
text_input = tokens(text_content, return_tensors='pt', max_length=128, padding='max_length', truncation=True)

# 加载音频文件
audio_file = 'wavDir/' + file_name + '.wav'
audio_input_tensor = torch.tensor(audio, dtype=torch.float).unsqueeze(0)  # Convert input to float
wav2vec_output = wav2vec_model.feature_extractor(audio_input_tensor)
wav2vec_output_adjusted = wav2vec_output.transpose(1, 2)
linear_bert = nn.Linear(768, 512)
bert_model = BertModel.from_pretrained('bert-base-chinese')

# 获取BERT中间输出
with torch.no_grad():
    bert_output = bert_model(**text_input)[
        'last_hidden_state'].detach()  # (batch_size, sequence_length, hidden_size)
bert_output_adjusted = linear_bert(bert_output)
merged_features = torch.cat((bert_output_adjusted, output_adjusted),
                            dim=1)  # Shape: (batch_size, seq_length1 + seq_length2, hidden_size/feature_dim)
output_features = torch.mean(merged_features, dim=1)  # Shape: (batch_size, hidden_size/feature_dim)

emb_test = output_features

unet1 = Unet3D(dim = 128,
	dim_mults = (1, 2, 4),
        max_text_len= 1,
        num_resnet_blocks = 2,
        layer_attns = False, # type: ignore
        layer_cross_attns = (False, False, True), # type: ignore
        ).cuda()
#unet1 = nn.DataParallel(unet1)
#unet2 = Unet3D(dim = 128, dim_mults = (1, 2, 4, 8), max_text_len= 1).cuda()

imagen = ElucidatedImagen(
    unets = unet1,
    image_sizes = 64,
    text_embed_dim = 512,
    random_crop_sizes = None,
    temporal_downsample_factor = 1,        # in this example, the first unet would receive the video temporally downsampled by 2x
    num_sample_steps = 256,
    cond_drop_prob = 0.1,
    sigma_min = 0.002,                          # min noise level
    sigma_max = 160,                      # max noise level, double the max noise level for upsampler
    sigma_data = 0.25,                           # standard deviation of data distribution
    rho = 7,                                    # controls the sampling schedule
    P_mean = -1.2,                              # mean of log-normal distribution from which noise is drawn for training
    P_std = 1.2,                                # standard deviation of log-normal distribution from which noise is drawn for training
    S_churn = 80,                               # parameters for stochastic sampling - depends on dataset, Table 5 in apper
    S_tmin = 0.05,
    S_tmax = 50,
    S_noise = 1.003,
).cuda()
#imagen = nn.DataParallel(imagen)

trainer = ImagenTrainer(imagen).to(device)

train_size = int(0.95 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print(f"训练集样本数: {len(train_dataset)}")
print(f"验证集样本数: {len(val_dataset)}")

# Define dataset
trainer.add_train_dataset(train_dataset,  batch_size = 1,num_workers=4)
trainer.add_valid_dataset(val_dataset, batch_size = 1, num_workers=4)

# Trainning variables
start_time = time.time()
avg_loss = 1.0
w_avg = 0.99
target_loss = 0.005

