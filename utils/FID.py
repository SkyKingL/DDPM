# python -m pytorch_fid img_ddim img_ddpm
"""
conda activate virtual_pytorch
cd D:\Deep_Learning_ALL\Opencv_pytorch_Project\Diffusion-pytorch
python -m pytorch_fid D:\Deep_Learning_ALL\Opencv_pytorch_Project\Diffusion-pytorch\img_ddim D:\Deep_Learning_ALL\Opencv_pytorch_Project\Diffusion-pytorch\img_ddpm
"""
import torch
import torch.nn.functional as F
from torchvision.models import inception_v3
from torchvision.transforms import ToTensor
from torch.nn.functional import adaptive_avg_pool2d
import os
from PIL import Image


def calculate_activation_statistics(images, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    act_values = []

    with torch.no_grad():
        for img in images:
            img_tensor = ToTensor()(img).unsqueeze(0).to(device)
            act = model(img_tensor)
            print(act.shape)
            act = F.adaptive_avg_pool2d(act)
            act_values.append(act.squeeze())

    act_values = torch.stack(act_values, dim=0)
    mu = torch.mean(act_values, dim=0)
    sigma = torch.cov(act_values, rowvar=False)

    return mu, sigma


def frechet_distance(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean, _ = torch.sqrtm(sigma1 @ sigma2, True)
    if torch.iscomplex(covmean):
        covmean = covmean.real

    return torch.norm(diff) + torch.trace(sigma1 + sigma2 - 2 * covmean)


def frechet_inception_distance(real_images, generated_images, batch_size=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = inception_v3(pretrained=True, transform_input=False).to(device)
    model.eval()

    mu_real, sigma_real = calculate_activation_statistics(real_images, model)
    mu_fake, sigma_fake = calculate_activation_statistics(generated_images, model)

    fid_score = frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
    return fid_score.item()


# 准备真实数据分布和生成模型的图像数据，两个文件夹里面图片数量和大小需要一样
ddpm_images_folder = 'img_ddpm'
ddim_images_folder = 'img_ddim'
test_images_folder = 'img_test'

# 创建空列表用于存储图像数据
real_images = []
generated_images = []

# 遍历真实图像文件夹，将图像数据添加到 real_images 列表中
for filename in os.listdir(ddpm_images_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(ddpm_images_folder, filename)
        image = Image.open(image_path)
        real_images.append(image)

# 遍历生成图像文件夹，将图像数据添加到 generated_images 列表中
for filename in os.listdir(ddim_images_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(ddim_images_folder, filename)
        image = Image.open(image_path)
        generated_images.append(image)

# 遍历生成图像文件夹，将图像数据添加到 generated_images 列表中
for filename in os.listdir(test_images_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(test_images_folder, filename)
        image = Image.open(image_path)
        generated_images.append(image)

# 调用示例
fid_score = frechet_inception_distance(ddpm_images_folder, test_images_folder)
print(f"DDPM FID Score: {fid_score}")

fid_score = frechet_inception_distance(ddim_images_folder, test_images_folder)
print(f"DDIM FID Score: {fid_score}")
