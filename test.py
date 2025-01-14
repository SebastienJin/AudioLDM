import pandas as pd
import torchaudio
from pathlib import Path
from diffusers import AudioLDMPipeline
import torch
import numpy as np
from scipy.linalg import sqrtm
from scipy.special import logsumexp
from scipy.stats import entropy
import torch.nn as nn
from tqdm import tqdm
import librosa
from torchvggish import vggish, vggish_input
import os


metadata_path = "./audiocaps/dataset/test.csv"
metadata = pd.read_csv(metadata_path)

device = torch.device("cuda:3")
repo_id = "cvssp/audioldm-s-full-v2"
pipe = AudioLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
pipe = pipe.to(device)

vgg = vggish()
vgg.eval()

length = len(metadata)

fd_scores = []
is_scores = []
kl_scores = []

def generate_audio(prompt, num_inference_steps=10, audio_length_in_s=5.0):
    audio = pipe(prompt, num_inference_steps=num_inference_steps, audio_length_in_s=audio_length_in_s).audios[0]
    return audio

def extract_vggish_features(waveform, sr=16000):
    # 如果 waveform 是 PyTorch 张量，则转换为 NumPy 数组
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.cpu().numpy()

    # 如果是立体声，转换为单声道
    if waveform.ndim == 2:
        waveform = np.mean(waveform, axis=0)

    # 确保采样率为 16000 Hz
    expected_sr = 16000
    if sr != expected_sr:
        raise ValueError(f"采样率不符，当前采样率为 {sr}，需要 {expected_sr}")

    # VGGish 参数：窗口长度 0.025 秒，对应 400 个采样点
    window_seconds = 0.025
    min_length = int(window_seconds * expected_sr)

    # 检查并填充 waveform 长度
    if len(waveform) < min_length:
        pad_width = min_length - len(waveform)
        waveform = np.pad(waveform, (0, pad_width), mode='constant')

    # 提取 log-mel 特征
    log_mel = vggish_input.waveform_to_examples(waveform, sr)

    # 检查 log_mel 是否为空
    if log_mel.size == 0:
        raise ValueError("提取的 log-mel 特征为空，请检查音频输入是否有效。")

    # 转换为 PyTorch 张量并传入 VGGish 模型
    log_mel_tensor = torch.FloatTensor(log_mel)
    # 假设已加载的 VGGish 模型实例为 vgg
    embeddings = vgg(log_mel_tensor)

    return embeddings


def compute_statistics(features):
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    # 计算平方根矩阵
    covmean = sqrtm(sigma1.dot(sigma2))
    # 如果结果存在虚部，则取实部
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fd = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fd

def calculate_inception_score(features, splits=10):
    # 将 features 转换为概率分布
    preds = nn.Softmax(dim=1)(torch.tensor(features)).numpy()
    split_scores = []
    n = preds.shape[0]
    for k in range(splits):
        part = preds[k * (n // splits): (k + 1) * (n // splits), :]
        # 计算该分片的边缘分布
        py = np.mean(part, axis=0)
        scores = []
        # 对于每个样本计算 KL 散度
        for p in part:
            # 避免 log(0) 问题，可加入微小常数 epsilon
            epsilon = 1e-16
            p = np.clip(p, epsilon, 1.0)
            py = np.clip(py, epsilon, 1.0)
            scores.append(np.sum(p * (np.log(p) - np.log(py))))
        # 对当前分片求平均后取指数
        split_scores.append(np.exp(np.mean(scores)))
    return np.mean(split_scores)

def calculate_kl_divergence(mu1, sigma1, mu2, sigma2, epsilon=1e-6):
    # 对 sigma1 和 sigma2 进行正则化，确保它们可逆
    sigma1_reg = sigma1 + epsilon * np.eye(sigma1.shape[0])
    sigma2_reg = sigma2 + epsilon * np.eye(sigma2.shape[0])

    # 计算 sigma2 的逆矩阵
    inv_sigma2 = np.linalg.inv(sigma2_reg)
    diff = mu2 - mu1

    # 计算各项
    term1 = np.trace(inv_sigma2.dot(sigma1_reg))
    term2 = diff.T.dot(inv_sigma2).dot(diff)

    # 使用绝对值和 epsilon 防止行列式为零或负数的问题
    det1 = np.abs(np.linalg.det(sigma1_reg)) + epsilon
    det2 = np.abs(np.linalg.det(sigma2_reg)) + epsilon
    term3 = np.log(det2 / det1)

    k = len(mu1)
    kl_div = 0.5 * (term1 + term2 - k + term3)
    return kl_div

# loop
for i in tqdm(range(length)):

    name = metadata["audiocap_id"][i]
    audio_path_real = "./test/"+str(name)+".mp3"

    if os.path.exists(audio_path_real):

        waveform, sr = torchaudio.load(audio_path_real, sr=16000, mono=True)
        real_vgg = extract_vggish_features(waveform, sr)
        audio_gen = generate_audio(metadata["caption"][i])
        gen_vgg = extract_vggish_features(audio_gen, sr)

        mu_real, sigma_real = compute_statistics(real_vgg)
        mu_gen, sigma_gen = compute_statistics(gen_vgg)
        fd = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
        fd_scores.append(fd)

        is_mean, is_std = calculate_inception_score(gen_vgg)
        is_scores.append(is_mean)

        kl = calculate_kl_divergence(mu_real, sigma_real, mu_gen, sigma_gen)
        kl_scores.append(kl)

        average_fd = np.mean(fd_scores)
        average_is = np.mean(is_scores)
        average_kl = np.mean(kl_scores)

print(f"平均 Frechet 距离 (FD): {average_fd}")
print(f"平均 Inception Score (IS): {average_is}")
print(f"平均 Kullback–Leibler 散度 (KL): {average_kl}")
