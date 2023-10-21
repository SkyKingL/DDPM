import os
import warnings
import torch
from torch.optim import Adam
from utils.networkHelper import *
from torch.utils.data import DataLoader
from model.UNet import Unet
from utils.trainHelper import SimpleDiffusionTrainer
from model.Diffusion import DiffusionModel
from lsun_dataset import LSUNChurchDataset, reverse_transform

# 训练超参数
image_size = 128  # 图片resize长宽
channels = 3  # 图片channel size
batch_size = 4  # batch_size
timesteps = 1000  # 时间步T
epoches = 20  # 迭代次数
schedule_name = "linear_beta_schedule"
device = "cuda" if torch.cuda.is_available() else "cpu"
dim_mults = (1, 2, 4,)  # 指定unet每个下采样(channel增加)和上采样块(channel减小)的通道数倍数

# 路径设置
root_path = "./checkpoints"
setting = "imageSize{}_channels{}_dimMults{}_timeSteps{}_scheduleName{}".format(image_size, channels, dim_mults,
                                                                                timesteps, schedule_name)
# 生成模型保存路径
saved_path = os.path.join(root_path, setting)
if not os.path.exists(saved_path):
    os.makedirs(saved_path)

# 实例化dataset和datsloader
data_set = LSUNChurchDataset(data_dir="./church_outdoor_train", image_size=image_size)
data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)


def train(resume_checkpoint_path=None):
    # 实例化unet噪声预测模型denoise_model
    denoise_model = Unet(
        dim=32,
        channels=channels,
        dim_mults=dim_mults
    )
    # 实例化DiffusionModel( 前向扩散beta/alpha + 去噪模型unet )
    DDPM = DiffusionModel(schedule_name=schedule_name,
                          timesteps=timesteps,
                          beta_start=0.0001,
                          beta_end=0.02,
                          denoise_model=denoise_model).to(device)
    # 实例化训练器optimizer
    optimizer = Adam(DDPM.parameters(), lr=1e-4)
    # 实例化trainer(forward就是train)
    Trainer = SimpleDiffusionTrainer(epoches=epoches,
                                     train_loader=data_loader,
                                     optimizer=optimizer,
                                     device=device,
                                     timesteps=timesteps)

    # 加载checkpoint
    if resume_checkpoint_path:
        DDPM.load_state_dict(torch.load(resume_checkpoint_path))

    # 进行训练
    Trainer(DDPM, model_save_path=saved_path)


def inference():
    # 实例化unet噪声预测模型
    denoise_model = Unet(
        dim=image_size,
        channels=channels,
        dim_mults=dim_mults
    )
    # 实例化DiffusionModel
    DDPM = DiffusionModel(schedule_name=schedule_name,
                          timesteps=timesteps,
                          beta_start=0.0001,
                          beta_end=0.02,
                          denoise_model=denoise_model).to(device)
    # 加载训练好的模型
    best_model_path = 'checkpoints/BestModel.pth'
    DDPM.load_state_dict(torch.load(best_model_path))

    # ddpm采样:sample 64 images(一次生成batch_size张图像，放在samples中，记得reshape(CWH))
    samples = DDPM(mode="ddpm", image_size=image_size, batch_size=batch_size, channels=channels)
    # 随机挑一张显示
    random_index = 1
    generate_image = samples[-1][random_index].reshape(channels, image_size, image_size)
    # 假设您已经定义了 reverse_transform 函数用于逆转图像预处理操作
    figtest = reverse_transform(torch.from_numpy(generate_image))
    # 保存生成的图像,确保 figtest 对象是一个 PIL.Image 对象。
    image_path = "./ddpm_image.jpg"  # 图像保存路径
    figtest.save(image_path)  # 保存图像
    figtest.show()

    # ddim采样:sample 64 images
    samples = DDPM(mode="ddpm", image_size=image_size, batch_size=64, channels=channels)
    # 随机挑一张显示
    random_index = 1
    generate_image = samples[-1][random_index].reshape(channels, image_size, image_size)
    # 假设您已经定义了 reverse_transform 函数用于逆转图像预处理操作
    figtest = reverse_transform(torch.from_numpy(generate_image))
    # figtest = torch.from_numpy(generate_image)
    # 保存生成的图像,确保 figtest 对象是一个 PIL.Image 对象。
    image_path = "./ddpm_image.jpg"  # 图像保存路径
    figtest.save(image_path)  # 保存图像
    figtest.show()


if __name__ == "__main__":
    checkpoint_path = "checkpoints/Model_epoch0_loss0.012339583598077297.pth"
    train(checkpoint_path)

    # 推理
    # inference()