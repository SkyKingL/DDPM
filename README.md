<h1 align="center">DDPM and DDIM implement in church_outdoor of LSUN</h1>

<div style="text-align:center; font-family:'Times New Roman', serif;">
    <p style="font-size:18px;"><b>Author: ZhengRong Yue</b></p>
    <p style="font-size:16px;">Date: August 15, 2023</p>
</div>


| ![DDPM](image/DDPM.jpg) | ![DDIM](image/DDIM.jpg) | ![Real](image/Real.jpg) | 
|:---:|:-----------------------:|:-----------------------:|
| DDPM |          DDIM           |          Real           |


### Project Structure
```
|--checkpoint: 训练权重
|--church_outdoor_train: 训练集图片
|--data
|  |--train: 训练集imdb
|  |  |--val: 验证集imdb
|--image: 一些可视化图片
|--img_ddim: ddim采样生成的图片
|--img_ddpm: ddpm采样生成的图片
|--model
|  |--Diffusion.py: Diffusion模型实现
|  |--UNet.py: UNet模型实现
|--reference_tutorial
|  |--ddim_mnist.ipynb:ddim在mnist数据集上的实现
|  |--ddpm_cifar10.ipynb:ddim在cifar10数据集上的实现
|  |--ddpm_mnist.ipynb:ddpm在mnist数据集上的实现
|--utils
|  |--FID.py: FID计算相关函数
|  |--networkHelper.py: 模型实现的辅助函数
|  |--trainHelper.py: 模型训练实现的辅助函数
|  |--varianceSchedule.py: DDPM噪声权重参数调整策略
|--data_preprocess.py: 从imdb读取数据的辅助函数
|--lsun_dataset.py: lsun数据集Dataset实现
|--main.py: 模型训练与推理脚本
|--test.py: 模型加噪去噪过程可视化函数,计算FID函数,及DDPM/DDIM生成测试图像的函数
```

### Dataset
LSUN数据集是真的难搞，我复现的时候下载数据集就花了1周，速度太慢了，慢到怀疑人生，这里放出我下载好的网盘链接：[LSUN Dataset](https://pan.baidu.com/s/1JrSycoTs45wcphDWJfsu_g?pwd=sjtu 
)，提取码：`sjtu`
- church_outdoor_train: 126227张图像 <br> 
- church_outdoor_val: 300张图像


### Model

<center>

| Method                           | FID             | Pretrain                                            |
|----------------------------------|-----------------|-----------------------------------------------------|
| DDPM                             | 279.25 (T=1000) | [Pretrain_DDPM_60_epoch](https://example.com/link3) |
| DDIM                             | 247.54 (T=100)  | [Pretrain_DDIM_60_epoch](https://example.com/link6) |

</center>
- 因为在AutoDL上租用云服务器训练，本地没有保存train dataset, test dataset, 训练结果, 以及生成的大量图片。 <br>
- 因为GPU资源的问题,我的batch_size只能设置为4,复现效果不理想,如果GPU充足,读者可以把btach_size调大重新train一遍，效果应该会好很多。 <br>
- 经过这次的复现我算是体会到了GPU的重要性，稍微大的模型和数据集，30系列根本不够用，起码得整张V100或A100。wu wu wu~~




### DDPM Denoise
![DDPM Denoise](image/ddpm_gen_pro.jpg)

### DDIM Denoise
![DDPM Denoise](image/ddim_gen_pro.jpg)



### Reference
[1]: DDPM: Denoising Diffusion Probabilistic Models. https://arxiv.org/abs/2006.11239 <br>
[2]: DDIM: Denoising Diffusion Implicit Models. https://arxiv.org/abs/2010.02502
