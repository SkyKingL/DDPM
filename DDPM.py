import lightning
import torch

from model import GaussianDiffusion, UNetModel


class DDPM(lightning.LightningModule):
    def __init__(self, timesteps: int = 1000) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.hparams["timesteps"] = timesteps
        self.model = UNetModel(
            in_channels=3,
            model_channels=32,
            out_channels=3,
            channel_mult=(1, 2, 2, 2),
            attention_resolutions=(2,),
            dropout=0.1,  # 控制unte模型通道大小，要求可被32整除
        )
        self.gaussian_diffusion = GaussianDiffusion(timesteps=timesteps)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return self.model(x, timesteps)

    def training_step(self, batch: torch.Tensor, batch_idx):
        images, labels = batch
        # sample t uniformally for every example in the batch
        batch_size = images.size(0)
        t = torch.randint(0, self.hparams["timesteps"], (batch_size,), device=self.device).long()
        loss = self.gaussian_diffusion.train_losses(self.model, images, t)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=2e-4)
