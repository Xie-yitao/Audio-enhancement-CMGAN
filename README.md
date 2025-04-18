### 1、项目介绍，基于CMGAN的音频增强，主要对音频背景人声噪音进行去除（针对单声道语音）
**文件结构**
```
src                         #核心代码

|-utils.py                  #模型参数处理函数
|-train.py                  #训练代码
|-evaluation.py             #评估代码
|-predict.py                #推理代码，指定音频文件路径，输出增前后的音频文件
|-data_processing.py        #数据预处理函数，将非wav格式音频文件统一转换为wav格式
|-audio_segmentation.py     #将长音频进行切分为小段，方便模型推理
|-requirements.txt          #环境配置
|-models                    #模型结构代码
    |-conformer.py
    |-discriminator.py
    |-generator.py
|-data                      #加载数据集代码
    |-dataloader.py
|-tools                     #工具文件，对测试集结果进行可视化等
    |-compute_metrics.py
    |-Noisy_metrics_results

LICENSE
README.md
```
### 2、参考CMGAN项目，此项目为分布式训练，本项目改为单机训练主要修改如下

1. **将训练代码src/train.py进行修改，把分布式训练修改为单GPU**


2. **工具函数src/utils.py修改**

```
def power_compress(x):
    real = x[..., 0]
    imag = x[..., 1]
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = mag**0.3
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return torch.stack([real_compress, imag_compress], 1)


def power_uncompress(real, imag):
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = mag ** (1.0 / 0.3)
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return torch.stack([real_compress, imag_compress], -1)
```
**修改为:**

```
def power_compress(x):
    real = x[..., 0]  # 实部
    imag = x[..., 1]  # 虚部
    spec = torch.complex(real, imag)  # 组合为复数
    mag = torch.abs(spec) ** 0.3
    phase = torch.angle(spec)
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return torch.stack([real_compress, imag_compress], dim=1)



def power_uncompress(real, imag):
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = mag ** (1.0 / 0.3)
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return torch.stack([real_compress, imag_compress], -1)
```


3. **数据载入src/dataloader.py修改：**

  ```
  def load_data(ds_dir, batch_size, n_cpu, cut_len):
   torchaudio.set_audio_backend("sox_io")  # in linux
   train_dir = os.path.join(ds_dir, "train")
   test_dir = os.path.join(ds_dir, "test")
  
   train_ds = DemandDataset(train_dir, cut_len)
   test_ds = DemandDataset(test_dir, cut_len)
  
   train_dataset = torch.utils.data.DataLoader(
       dataset=train_ds,
       batch_size=batch_size,
       pin_memory=True,
       shuffle=False,
       sampler=DistributedSampler(train_ds),
       drop_last=True,
       num_workers=n_cpu,
   )
   test_dataset = torch.utils.data.DataLoader(
       dataset=test_ds,
       batch_size=batch_size,
       pin_memory=True,
       shuffle=False,
       sampler=DistributedSampler(test_ds),
       drop_last=False,
       num_workers=n_cpu,
   )
  
   return train_dataset, test_dataset
  ```

  **修改为：**
```
def load_data(ds_dir, batch_size, n_cpu, cut_len):
    torchaudio.set_audio_backend("sox_io")  # in linux
    train_dir = os.path.join(ds_dir, "train")
    test_dir = os.path.join(ds_dir, "test")

    train_ds = DemandDataset(train_dir, cut_len)
    test_ds = DemandDataset(test_dir, cut_len)

    train_dataset = torch.utils.data.DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        num_workers=n_cpu,
    )
    test_dataset = torch.utils.data.DataLoader(
        dataset=test_ds,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
        num_workers=n_cpu,
    )
```
4. **模型评估src/evaluation.py：**
  对验证代码也进行的小修改，主要适配RuntimeError: istft requires a complex-valued input tensor matching the output from stft with return_complex=True.报错。
5. **增加如下代码**
   - audio_segmentation.py	
   - data_processing.py
   - predict.py

### 3、源码训练与测试代码，不变，可参考https://github.com/ruizhecao96/CMGAN/tree/main项目中的readme
## 如何训练:

### 环境准备在src文件夹中运行:
python版本：Python 3.10.16 
```pip install -r requirements.txt```

### 下载数据集:
下载数据集按照如下结构放入src，数据集可以在飞浆下载（https://aistudio.baidu.com/datasetdetail/62188）:
```
-VCTK-DEMAND/
  -train/
    -noisy/
    -clean/
  -test/
    -noisy/
    -clean/
```

### 训练模型:
运行src中的train训练模型
```
python3 train.py --data_dir <dir to VCTK-DEMAND dataset>
```

### 模型评估:
使用训练好的进行评估:
```
python3 evaluation.py --test_dir <dir to VCTK-DEMAND/test> --model_path <path to the best ckpt>
```

### 模型推理：
1. 运行数据预处理代码（data_processing.py），将音频文件统一转换为wav格式
2. 运行音频切分代码（audio_segmentation.py	），对音频文件进行切分，方便模型处理
2. 使用如下指令运行推理代码：
```
python predict.py --model_path <path to the best ckpt> --input_path .<path to audio or dir to audio>
```

### 第三方依赖项

该项目包含来自以下第三方项目的代码：

- **Project Name**: CMGAN
  - **Author**: Ruizhe Cao
  - **License**: MIT License
  - **Source**: [\[链接到原始项目的仓库或页面\]](https://github.com/ruizhecao96/CMGAN/tree/main)