import os
from pydub import AudioSegment

# 指定 ffmpeg 和 ffprobe 的路径
# os.environ["PATH"] += os.pathsep + r"C:\path\to\ffmpeg\bin" #需要修改为实际的路径
os.environ["PATH"] += os.pathsep + r"D:\兼职\基于CMGAN的音频增强（降噪）\ffmpeg-2025-03-31-git-35c091f4b7-essentials_build\bin"

def convert_audio_files(input_folder, output_folder):
    """
    将指定文件夹中的所有非 WAV 音频文件转换为 WAV 格式，并保存到目标文件夹。
    
    参数:
        input_folder (str): 输入文件夹路径
        output_folder (str): 输出文件夹路径
    """
    # 检查输出文件夹是否存在，不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"已创建输出文件夹：{output_folder}")
    
    # 支持的音频格式（可以根据需要扩展）
    supported_formats = ["mp3", "m4a", "ogg", "flac", "aac"]
    
    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        
        # 跳过文件夹
        if os.path.isdir(file_path):
            continue
        
        # 获取文件扩展名
        file_ext = filename.split(".")[-1].lower()
        
        # 检查是否是支持的非 WAV 音频格式
        if file_ext in supported_formats:
            # 构建输出文件路径
            output_filename = os.path.splitext(filename)[0] + ".wav"
            output_path = os.path.join(output_folder, output_filename)
            
            # 加载音频文件
            try:
                audio = AudioSegment.from_file(file_path, format=file_ext)
                # 导出为 WAV 格式
                audio.export(output_path, format="wav")
                print(f"已转换：{filename} -> {output_filename}")
            except Exception as e:
                print(f"转换文件 {filename} 时出错：{e}")
        elif file_ext == "wav":
            print(f"跳过 WAV 文件：{filename}")
        else:
            print(f"不支持的文件格式：{filename}")

if __name__ == "__main__":
    # 输入和输出文件夹路径（根据需要修改）
    input_folder = "testdata"  # 输入文件夹
    output_folder = "处理后的文件"   # 输出文件夹
    
    # 执行转换
    convert_audio_files(input_folder, output_folder)
    print("所有文件转换完成！")