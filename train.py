import lightning
import torch
import torchvision.transforms as T
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from config import *
from dataset import LSUNChurchDataset
from DDPM import DDPM


def f(t: torch.Tensor) -> torch.Tensor:
    return t * 2 - 1


if __name__ == "__main__":
    lightning.seed_everything(SEED)
    model = DDPM(TIMESTEPS)

    transform = T.Compose(
        [
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),  # Scale data int [0,1]
            T.Lambda(f),  # turn [0, 1] into [-1, 1]
        ]
    )
    dataset = LSUNChurchDataset(data_dir=DATA_DIR, image_size=IMAGE_SIZE, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=DATALOADER_WORKERS, persistent_workers=True)

    checkpoint_callback = ModelCheckpoint(every_n_epochs=SAVE_EVERY_N_EPOCHS, every_n_train_steps=SAVE_EVERY_N_STEPS, save_top_k=-1, save_last=True)
    trainer = lightning.Trainer(
        devices=DEVICES,
        accelerator="gpu",
        max_epochs=MAX_EPOCHES,
        callbacks=[checkpoint_callback],
        deterministic=DETERMINISTIC,
    )
    trainer.fit(model, train_dataloaders=dataloader)
