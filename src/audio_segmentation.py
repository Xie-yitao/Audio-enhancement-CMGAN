from pydub import AudioSegment
import os
os.environ["PATH"] += os.pathsep + r"D:\兼职\基于CMGAN的音频增强（降噪）\ffmpeg-2025-03-31-git-35c091f4b7-essentials_build\bin"
def split_wav(input_file, output_dir, segment_length=10000):
    """
    将 WAV 文件切分为指定长度的段
    :param input_file: 输入的 WAV 文件路径
    :param output_dir: 输出目录
    :param segment_length: 每段的长度（毫秒），默认 10000 毫秒（10 秒）
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载音频文件
    audio = AudioSegment.from_wav(input_file)
    
    # 计算总时长（毫秒）
    total_duration = len(audio)
    
    # 计算需要切分的段数
    num_segments = total_duration // segment_length
    if total_duration % segment_length != 0:
        num_segments += 1
    
    # 切分并保存每段
    for i in range(num_segments):
        start_time = i * segment_length
        end_time = min((i + 1) * segment_length, total_duration)
        
        # 切分当前段
        segment = audio[start_time:end_time]
        
        # 保存当前段
        output_file = os.path.join(output_dir, f"output_{i+1}.wav")
        segment.export(output_file, format="wav")
        print(f"已保存: {output_file}")

# 示例用法
input_file = "./testdata/答辩记录.wav"  # 输入的 30 分钟 WAV 文件
output_dir = "output_segments"  # 输出目录
split_wav(input_file, output_dir)