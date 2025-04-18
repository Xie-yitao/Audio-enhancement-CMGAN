import numpy as np
from models import generator
import os
from utils import *
import torchaudio
import argparse
import soundfile as sf
import torch
import torchaudio.functional as F
from tqdm import tqdm
from scipy.io.wavfile import write
# 定义参数配置
INFER_CONFIG = {
    # 音频处理参数
    "target_sample_rate": 16000,  # 目标采样率
    "cut_len": 16000 * 2,         # 分块处理长度（2秒）
    "overlap_ratio": 0.0,         # 分块重叠比例（0表示不重叠）
    
    # STFT参数
    "n_fft": 400,                 # FFT点数
    "hop_length": None,           # 帧移长度（默认为n_fft//4）
    "win_length": None,           # 窗口长度（默认等于n_fft）
    
    # 模型推理参数
    "batch_size": 8,              # 推理批次大小
    "num_channels": 64,           # 模型通道数
    
    # 归一化参数
    "epsilon": 1e-8,              # 防止除零的极小值
}

@torch.no_grad()
def enhance_one_track(
    model, 
    audio_path, 
    saved_dir, 
    config=INFER_CONFIG,
    save_tracks=True
):
    """
    增强单个音频文件
    
    Args:
        model: 预训练的音频增强模型
        audio_path: 输入音频路径
        saved_dir: 保存增强音频的目录
        config: 推理配置参数
        save_tracks: 是否保存增强后的音频
    
    Returns:
        est_audio: 增强后的音频数据
        saved_path: 增强音频保存路径
    """
    # 提取音频文件名
    name = os.path.split(audio_path)[-1]
    
    # 加载音频
    noisy, sr = torchaudio.load(audio_path)
    
    # 转换到目标采样率
    if sr != config["target_sample_rate"]:
        noisy = F.resample(noisy, orig_freq=sr, new_freq=config["target_sample_rate"])
        sr = config["target_sample_rate"]
    # 如果音频是单通道，复制到双通道
    if noisy.shape[0] == 1:
        noisy = torch.cat([noisy, noisy], dim=0)
    # 移动到GPU
    noisy = noisy.cuda()
    
    # 归一化处理
    energy = torch.sum(noisy**2.0, dim=-1, keepdim=True)
    energy = torch.where(energy == 0, torch.tensor(config["epsilon"]).to(energy), energy)
    c = torch.sqrt(noisy.size(-1) / energy)
    noisy = noisy * c  # 保持形状为 (channels, time)
    
    # 获取通道数
    num_channels = noisy.size(0)
    
    # 获取音频长度
    length = noisy.size(-1)
    
    # 计算帧移长度（如果未指定则使用n_fft//4）
    hop_length = config["hop_length"] or config["n_fft"] // 4
    
    # 计算需要的帧数和填充长度
    frame_num = int(np.ceil(length / hop_length))
    padded_len = frame_num * hop_length
    padding_len = padded_len - length
    
    # 填充音频到整数帧
    noisy = torch.cat([noisy, noisy[:, :padding_len]], dim=-1)
    
    # 分块处理逻辑
    if padded_len > config["cut_len"]:
        # 计算分块参数
        chunk_size = config["cut_len"]
        overlap = int(chunk_size * config["overlap_ratio"])
        stride = chunk_size - overlap
        
        # 分割音频为多个块
        chunks = []
        for i in range(0, padded_len - chunk_size + 1, stride):
            chunks.append(noisy[:, i:i+chunk_size])
        
        # 处理最后一块
        if padded_len % chunk_size != 0:
            chunks.append(noisy[:, -chunk_size:])
        
        # 合并为批量
        noisy = torch.stack(chunks, dim=0)  # (chunks, channels, time)
    else:
        # 单块处理
        noisy = noisy.unsqueeze(0)  # (1, channels, time)
    
    # 推理参数
    batch_size = config["batch_size"]
    est_audio_segments = []
    
    # 分批次处理
    for i in range(0, noisy.shape[0], batch_size):
        batch = noisy[i:i+batch_size].cuda()
        
        # 合并通道和批次维度
        batch = batch.view(-1, batch.size(-1))
        
        # STFT处理
        noisy_spec = torch.stft(
            batch, 
            n_fft=config["n_fft"], 
            hop_length=hop_length, 
            win_length=config["win_length"] or config["n_fft"],
            window=torch.hamming_window(config["n_fft"]).cuda(), 
            onesided=True, 
            return_complex=True
        )
        
        # 特征处理
        noisy_spec_real = torch.real(noisy_spec)
        noisy_spec_imag = torch.imag(noisy_spec)
        noisy_spec_input = torch.stack([noisy_spec_real, noisy_spec_imag], dim=-1)
        noisy_spec_input = power_compress(noisy_spec_input).permute(0, 1, 3, 2)
        
        # 模型推理
        est_real, est_imag = model(noisy_spec_input)
        est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)
        
        # 后处理
        est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
        est_spec_complex = torch.complex(
            est_spec_uncompress[..., 0], 
            est_spec_uncompress[..., 1]
        )
        
        # ISTFT逆变换
        est_batch = torch.istft(
            est_spec_complex,
            n_fft=config["n_fft"],
            hop_length=hop_length,
            win_length=config["win_length"] or config["n_fft"],
            window=torch.hamming_window(config["n_fft"]).cuda(),
            onesided=True,
        )
        
        # 恢复通道结构
        est_batch = est_batch.view(-1, num_channels, chunk_size)
        
        # 保存当前批次结果
        est_audio_segments.append(est_batch.cpu())
        
        # 显存管理
        del batch, noisy_spec, est_real, est_imag
        torch.cuda.empty_cache()
    
    # 合并所有音频段
    est_audio = torch.cat(est_audio_segments, dim=0)
    
    # 处理多通道和单通道情况
    if num_channels > 1:
        est_audio = est_audio.permute(1, 0, 2)  # (channels, chunks, time)
        est_audio = est_audio.reshape(num_channels, -1)[:, :length]
    else:
        est_audio = est_audio.squeeze(1).reshape(-1)[:length]
    
    # 反归一化
    est_audio = est_audio.numpy() / c.cpu().numpy()
    
    # 保存增强音频
    if save_tracks:
        saved_path = os.path.join(saved_dir, name)
        
        # 处理多通道音频格式
        if num_channels > 1:
            est_audio = est_audio.T  # 转为 (time, channels)
        
        # 保存音频
        # write(saved_path, est_audio, sr)
        sf.write(saved_path, est_audio, sr)
        
        return est_audio, saved_path
    
    return est_audio, None

def inference(model_path, input_path, saved_dir, config=INFER_CONFIG):
    """
    批量音频增强推理
    
    Args:
        model_path: 模型权重路径
        input_path: 输入音频或目录
        saved_dir: 保存结果的目录
        config: 推理配置参数
    """
    # 初始化模型
    model = generator.TSCNet(
        num_channel=config["num_channels"], 
        num_features=config["n_fft"]//2+1
    ).cuda()
    
    # 加载模型权重
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 创建保存目录
    os.makedirs(saved_dir, exist_ok=True)
    
    # 处理输入路径
    if os.path.isdir(input_path):
        # 批量处理
        audio_files = []
        for root, _, files in os.walk(input_path):
            for f in files:
                if f.endswith(('.wav', '.flac', '.mp3')):
                    audio_files.append(os.path.join(root, f))
        
        # 创建批量保存目录
        batch_saved_dir = os.path.join(saved_dir, "batch_enhanced")
        os.makedirs(batch_saved_dir, exist_ok=True)
        
        # 处理每个音频文件
        for audio_file in tqdm(audio_files, desc="Processing"):
            try:
                enhance_one_track(
                    model, 
                    audio_file, 
                    batch_saved_dir, 
                    config=config
                )
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error processing {audio_file}: {str(e)}")
                continue
    else:
        # 单文件处理
        enhance_one_track(
            model, 
            input_path, 
            saved_dir, 
            config=config
        )

if __name__ == "__main__":
    # 参数解析
    parser = argparse.ArgumentParser(description="Audio Enhancement Inference")
    parser.add_argument("--model_path", type=str, default='./best_ckpt/ckpt_80', help="Path to model weights")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input audio or directory")
    parser.add_argument("--save_dir", type=str, default='./enhanced_audio', help="Directory to save enhanced audio")
    
    # 高级参数（可选）
    parser.add_argument("--target_sr", type=int, help="Target sample rate (default: 16000)")
    parser.add_argument("--cut_len", type=int, help="Cut length in samples (default: 32000)")
    parser.add_argument("--overlap", type=float, help="Overlap ratio for chunks (default: 0.0)")
    parser.add_argument("--n_fft", type=int, help="FFT size (default: 400)")
    parser.add_argument("--batch_size", type=int, help="Inference batch size (default: 8)")
    
    args = parser.parse_args()
    
    # 更新配置参数（如果提供了命令行参数）
    if args.target_sr:
        INFER_CONFIG["target_sample_rate"] = args.target_sr
    if args.cut_len:
        INFER_CONFIG["cut_len"] = args.cut_len
    if args.overlap is not None:
        INFER_CONFIG["overlap_ratio"] = args.overlap
    if args.n_fft:
        INFER_CONFIG["n_fft"] = args.n_fft
    if args.batch_size:
        INFER_CONFIG["batch_size"] = args.batch_size
    
    # 设置CUDA内存分配配置
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # 执行推理
    inference(args.model_path, args.input_path, args.save_dir, config=INFER_CONFIG)