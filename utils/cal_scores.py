# -*- coding: utf-8 -*-

import os
import numpy as np
from PIL import Image, ImageSequence
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from skimage.metrics import structural_similarity as ssim
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from PIL import Image
import numpy as np
import imageio
import cv2
from skimage.metrics import structural_similarity as compare_ssim

def resize_frame(frame, new_width, new_height):
    return frame.resize((new_width, new_height), Image.LANCZOS)

def gif_to_nparray(gif_path, new_width, new_height):
    with Image.open(gif_path) as gif_img:
        frames = [frame.convert("RGB") for frame in ImageSequence.Iterator(gif_img)]
        frames_resized = [np.array(resize_frame(frame, new_width, new_height)) for frame in frames]
    return np.array(frames_resized)


def avi_to_nparray(avi_path, new_width, new_height):
    frames_resized = []

    cap = cv2.VideoCapture(avi_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, (new_width, new_height))
        frames_resized.append(resized_frame)

    cap.release()
    return np.array(frames_resized)

lpips_scorer = LearnedPerceptualImagePatchSimilarity(net_type='squeeze') # lower better, [0-1]

def calculate_psnr(rmse, max_pixel=255.0):
    rmse = rmse
    if rmse == 0:
        return float('inf')
    return 20 * np.log10(max_pixel / rmse)

F_Size = 96
W_Size = 96

def main():
    # 排除的文件名列表
    #excluded_filenames = ["speaker00016_F_s1_stn00044", "speaker00016_F_s1_stn00098", "speaker00017_F_s1_stn00025", "speaker00017_F_s1_stn00083", "speaker00017_F_s1_stn00068", "speaker00018_M_s1_stn00016", "speaker00018_M_s1_stn00098", "speaker00021_M_s1_stn00063", "speaker00021_M_s1_stn00119", "speaker00027_F_s1_stn00060", "speaker00027_F_s1_stn00087", "speaker00038_M_s1_stn00028", "speaker00037_F_s1_stn00068", "speaker00038_M_s1_stn00033", "speaker00038_M_s1_stn00073", "speaker00038_M_s1_stn00074"]
    excluded_filenames = []
    # gif_folder = "/mnt/shareEEx/xuchangqing/yyd_output/sp_64"
    #gif_folder = "fairseq/output_mul_96_f"
    gif_folder = "output_DNN_f"
    # output_folder = "val_test/gif"
    output_folder = "data_frame20"

    new_width = F_Size
    new_height = W_Size

    avg_rmse_list = []
    avg_psnr_list = []
    avg_mse_list = []
    avg_mae_list = []
    avg_ssim_list = []
    avg_r2_list = []
    lpips_list = []
    ms_ssim_list = []
    gif_filenames = os.listdir(gif_folder)
    i = 0
    for gif_filename in gif_filenames:
        i += 1
        # 检查文件名是否在排除列表中

       # if gif_filename.split(".")[0].split("_output_")[1]  in excluded_filenames:
           # print(gif_filename.split(".")[0].split("_output_")[1])
        if gif_filename.split(".")[0] not in excluded_filenames:
            vi_file = gif_filename.split(".")[0] + ".avi"
            gif_path1 = os.path.join(output_folder, vi_file)
            gif_path2 = os.path.join(gif_folder, gif_filename)
            #lpips_score = calculate_lpips_score(gif_path1, gif_path2)
            #lpips_list.append(lpips_score)
            #print(gif_path2)
            gif1_frames = avi_to_nparray(gif_path1, new_width, new_height)
            gif2_frames = gif_to_nparray(gif_path2, new_width, new_height)
            
            #print(gif1_frames.shape, gif2_frames.shape)
    
            if gif1_frames.shape != gif2_frames.shape:
                min_frames = min(gif1_frames.shape[0], gif2_frames.shape[0])
                gif1_frames = gif1_frames[:min_frames]
                gif2_frames = gif2_frames[:min_frames]

            lpips_values = []
            rmse_values = []
            psnr_values = []
            mse_values = []
            mae_values = []
            ssim_values = []
            r2_scores = []
            ms_ssim_scores = []
            nf1 = gif1_frames *255
            nf2 = gif2_frames *255
            nf1 = torch.from_numpy(np.transpose(nf1, (0, 3, 1, 2)))  # Transpose dimensions
            nf2 = torch.from_numpy(np.transpose(nf2, (0, 3, 1, 2)))  # Transpose dimensions
            #print(nf1.shape)
            #print(nf2.shape)
            lpips_score = lpips_scorer(nf1.float(), nf2.float()).item()
            #print(lpips_score)
            lpips_values.append(lpips_score)
            for frame1, frame2 in zip(gif1_frames, gif2_frames):
                frame1_ = frame1.reshape(F_Size * W_Size, 3)
                frame2_ = frame2.reshape(F_Size * W_Size, 3)
    
                r_mse = mean_squared_error(frame1_[:, 0], frame2_[:, 0])
                g_mse = mean_squared_error(frame1_[:, 1], frame2_[:, 1])
                b_mse = mean_squared_error(frame1_[:, 2], frame2_[:, 2])
                #print(r_mse, g_mse, b_mse)
                weighted_mse = 0.2989 * r_mse + 0.5870 * g_mse + 0.1140 * b_mse
                rmse = np.sqrt(weighted_mse)
                mse = mean_squared_error(frame1_.flatten(), frame2_.flatten())
                rmse_avg_gen = np.sqrt(mse) 
    
    
                psnr = calculate_psnr(rmse)
                mae = mean_absolute_error(frame1_, frame2_)
                mae = np.mean(np.abs(np.array(frame1_) - np.array(frame2_)))
                ssim_value = ssim(frame1, frame2, multichannel=True, channel_axis=2)
                r2_score1 = r2_score(frame1_, frame2_)
                ms_ssim = compare_ssim(frame1, frame2, channel_axis=2)
                rmse_values.append(rmse_avg_gen)
                psnr_values.append(psnr)
                mse_values.append(mse)
                mae_values.append(mae)
                ssim_values.append(ssim_value)
                r2_scores.append(r2_score1)
                ms_ssim_scores.append(ms_ssim)

        avg_rmse_list.append(np.mean(rmse_values))
        avg_psnr_list.append(np.mean(psnr_values))
        avg_mse_list.append(np.mean(mse_values))
        avg_mae_list.append(np.mean(mae_values))
        avg_ssim_list.append(np.mean(ssim_values))
        avg_r2_list.append(np.mean(r2_scores))
        ms_ssim_list.append(np.mean(ms_ssim_scores))
    avg_rmse = np.mean(avg_rmse_list)
    avg_psnr = np.mean(avg_psnr_list)
    avg_mse = np.mean(avg_mse_list)
    avg_mae = np.mean(avg_mae_list)
    avg_ssim = np.mean(avg_ssim_list)
    avg_r2 = np.mean(avg_r2_list)
    avg_lps = np.mean(lpips_values)
    avg_ms = np.mean(ms_ssim_list)

    print("Average RMSE:", avg_rmse)
    print("Average PSNR:", avg_psnr)
    print("Average MSE:", avg_mse)
    print("Average MAE:", avg_mae)
    print("Average SSIM:", avg_ssim)
    print("Average MS-SSIM:", avg_ms)
    print("Average R2:", avg_r2)
    print("Average Lpips:", avg_lps)
if __name__ == "__main__":
    main()
