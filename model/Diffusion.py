from utils.networkHelper import *
from utils.varianceSchedule import VarianceSchedule


class DiffusionModel(nn.Module):
    def __init__(self,
                 schedule_name="linear_beta_schedule",
                 timesteps=300,
                 beta_start=0.0001,
                 beta_end=0.02,
                 denoise_model=None):
        super(DiffusionModel, self).__init__()

        self.denoise_model = denoise_model  # unet

        # 方差beta生成(噪声的权重系数)
        variance_schedule_func = VarianceSchedule(schedule_name=schedule_name, beta_start=beta_start, beta_end=beta_end)
        self.timesteps = timesteps
        self.betas = variance_schedule_func(timesteps)
        # 生成alphas权重
        self.alphas = 1. - self.betas  # alpha
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)  # alpha累乘:\bar{alpha}
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)  # sqrt(1/alpha)

        # 前向扩散系数calculations for diffusion q(x_t | x_{t-1}) and others
        # x_t  = sqrt(alphas_cumprod)*x_0 + sqrt(1 - alphas_cumprod)*z_t
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # 反向去噪系数calculations for posterior q(x_{t-1} | x_t, x_0)
        # 这里用的不是简化后的方差而是算出来的:betas / sqrt{1-\bar{alpha}}
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):  # ddpm前向扩散: 根据 x_0 和 t 计算任意 x_t
        # forward diffusion (using the nice property)
        # x_t  = sqrt(alphas_cumprod)*x_0 + sqrt(1 - alphas_cumprod)*z_t
        if noise is None:
            noise = torch.randn_like(x_start)  # 和x0大小相同的noise

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def compute_loss(self, x_start, t, noise=None, loss_type="l1"):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.denoise_model(x_noisy, t)

        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss

    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        # ddpm反向去噪公式: 根据xt和t计算 xt-1 = 1/sqrt{alpha} * (xt - posterior_variance_t * pre_epsilon) + sigma * noise
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean(u)
        model_mean = sqrt_recip_alphas_t * (x - betas_t * self.denoise_model(x, t) / sqrt_one_minus_alphas_cumprod_t)

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape):  # 从纯噪声开始，迭代T步，调用p_sample进行反向去噪
        device = next(self.denoise_model.parameters()).device

        b = shape[0]
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        imgs = []

        # 进行 T 步采样，每次将采样结果存入
        for i in tqdm(reversed(range(0, self.timesteps)), desc='ddpm_sampling loop time step', total=self.timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), i)
            imgs.append(img.cpu().numpy())
        return imgs

    @torch.no_grad()
    def ddpm_sample(self, image_size, batch_size=16, channels=3):
        return self.p_sample_loop(shape=(batch_size, channels, image_size, image_size))

    # use ddim to sample
    @torch.no_grad()
    def ddim_sample(
            self,
            image_size,
            batch_size=16,
            channels=3,
            ddim_timesteps=50,
            ddim_discr_method="uniform",  # ddim子序列划分方法
            ddim_eta=0.0,
            clip_denoised=True):
        # make ddim timestep sequence
        if ddim_discr_method == 'uniform':  # 线性划分
            c = self.timesteps // ddim_timesteps  # c是间隔c步取采样一次
            ddim_timestep_seq = np.asarray(list(range(0, self.timesteps, c)))  # 得到采样步的子序列ddim_timestep_seq
        elif ddim_discr_method == 'quad':  # 平方划分
            ddim_timestep_seq = ((np.linspace(0, np.sqrt(self.timesteps * .8), ddim_timesteps)) ** 2).astype(int)
        else:
            raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')
        # add one to get the final alpha values right (the ones from first scale to data during sampling)
        ddim_timestep_seq = ddim_timestep_seq + 1
        # previous sequence
        ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])   # 错位，后移一位取 alpha_{t-1}
        device = next(self.denoise_model.parameters()).device
        imgs = []
        # start from pure noise (for each example in the batch) 从纯噪声出发开始ddim_timesteps步去噪
        sample_img = torch.randn((batch_size, channels, image_size, image_size), device=device)

        for i in tqdm(reversed(range(0, ddim_timesteps)), desc='ddim_sampling loop time step', total=ddim_timesteps):
            # 从子序列ddim_timestep_seq中得到具体的采样步t
            t = torch.full((batch_size,), ddim_timestep_seq[i], device=device, dtype=torch.long)
            prev_t = torch.full((batch_size,), ddim_timestep_prev_seq[i], device=device, dtype=torch.long)
            # 1. get current and previous alpha_cumprod 计算公式的系数
            alpha_cumprod_t = extract(self.alphas_cumprod, t, sample_img.shape)
            alpha_cumprod_t_prev = extract(self.alphas_cumprod, prev_t, sample_img.shape)
            # 2. predict noise using model
            pred_noise = self.denoise_model(sample_img, t)  # unet预测噪声pred_noise
            # 3. get the predicted x_0
            # 根据预测的噪声pred_noise和xt 计算x0：(xt-sqrt{1-alpha_cumprod_t}*pred_noise)/sqrt{alpha_cumprod_t}
            pred_x0 = (sample_img - torch.sqrt((1. - alpha_cumprod_t)) * pred_noise) / torch.sqrt(alpha_cumprod_t)
            if clip_denoised:
                pred_x0 = torch.clamp(pred_x0, min=-1., max=1.)
            # 4. compute variance: "sigma_t(η)" -> see formula (16) 根据系数alpha_cumprod_t_prev和alpha_cumprod_t计算sigma随机噪声的系数
            # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
            sigmas_t = ddim_eta * torch.sqrt(
                (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))
            # 5. compute "direction pointing to x_t" of formula (12) 根据预测的噪声pred_noise和xt和sigma计算指向xt的部分
            pred_dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigmas_t ** 2) * pred_noise
            # 6. compute x_{t-1} of formula (12)  上面三部分组合起来就是xt-1
            x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt + sigmas_t * torch.randn_like(
                sample_img)
            sample_img = x_prev  # 将xt-1作为下一个timesteps的输入
            imgs.append(sample_img.cpu().numpy())  # 保存本timestep去噪得到的图像
        return imgs  # 返回每个timestep采样得到的图像

    def forward(self, mode, **kwargs):
        if mode == "train":
            # 先判断必须参数
            if "x_start" and "t" in kwargs.keys():
                # 接下来判断一些非必选参数
                if "loss_type" and "noise" in kwargs.keys():
                    return self.compute_loss(x_start=kwargs["x_start"], t=kwargs["t"],
                                             noise=kwargs["noise"], loss_type=kwargs["loss_type"])
                elif "loss_type" in kwargs.keys():
                    return self.compute_loss(x_start=kwargs["x_start"], t=kwargs["t"], loss_type=kwargs["loss_type"])
                elif "noise" in kwargs.keys():
                    return self.compute_loss(x_start=kwargs["x_start"], t=kwargs["t"], noise=kwargs["noise"])
                else:
                    return self.compute_loss(x_start=kwargs["x_start"], t=kwargs["t"])

            else:
                raise ValueError("扩散模型在训练时必须传入参数x_start和t！")

        elif mode == "ddpm":
            if "image_size" and "batch_size" and "channels" in kwargs.keys():
                _ = kwargs["ddim_timesteps"]
                return self.ddpm_sample(image_size=kwargs["image_size"],
                                        batch_size=kwargs["batch_size"],
                                        channels=kwargs["channels"])
            else:
                raise ValueError("扩散模型在生成图片时必须传入image_size, batch_size, channels等三个参数")

        elif mode == "ddim":
            if "image_size" and "batch_size" and "channels" in kwargs.keys():
                return self.ddim_sample(image_size=kwargs["image_size"],
                                        batch_size=kwargs["batch_size"],
                                        channels=kwargs["channels"],
                                        ddim_timesteps=kwargs["ddim_timesteps"])
            else:
                raise ValueError("扩散模型在生成图片时必须传入image_size, batch_size, channels等三个参数")

        else:
            raise ValueError("mode参数必须从{train}, {ddpm}, {ddim}两种模式中选择")
