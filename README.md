### 1、项目介绍，基于CMGAN的音频增强，对音频噪音进行去除
### 2、参考https://github.com/ruizhecao96/CMGAN/tree/main 项目，此项目为分布式训练，本项目改为单机训练主要修改如下

1. **训练代码src/train.py做了修改，把分布式训练修改为单GPU**


2. **工具函数src/utils.py**

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


3. **数据载入src/dataloader.py：**

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
### 3、源码训练与测试代码，不变，可参考https://github.com/ruizhecao96/CMGAN/tree/main项目中的readme
## 如何训练:

### 环境准备在src文件夹中运行:
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


### 第三方依赖项

该项目包含来自以下第三方项目的代码：

- **Project Name**: CMGAN
  - **Author**: Ruizhe Cao
  - **License**: MIT License
  - **Source**: [\[链接到原始项目的仓库或页面\]](https://github.com/ruizhecao96/CMGAN/tree/main)