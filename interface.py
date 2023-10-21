import matplotlib.pyplot as plt
import numpy as np

from DDPM import DDPM

if __name__ == "__main__":
    model = DDPM.load_from_checkpoint("lightning_logs/version_0/checkpoints/last.ckpt")
    gaussian_diffusion = model.gaussian_diffusion
    model.eval()
    generated_images = gaussian_diffusion.sample(model, image_size=64, batch_size=64, channels=3)
    fig = plt.figure(figsize=(12, 12), constrained_layout=True)
    gs = fig.add_gridspec(8, 8)

    # generated_images[-1] is the lastest timesteps image
    imgs = generated_images[-1].reshape(8, 8, 3, 64, 64)  # (n_row, n_col, c, w,h)
    # visualize 64 image
    for n_row in range(8):
        for n_col in range(8):
            f_ax = fig.add_subplot(gs[n_row, n_col])
            img = np.array((imgs[n_row, n_col].transpose([1, 2, 0]) + 1.0) * 255 / 2, dtype=np.uint8)
            f_ax.imshow(img)
            f_ax.axis("off")
    plt.show()
    plt.close()
