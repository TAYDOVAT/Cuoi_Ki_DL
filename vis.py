import matplotlib.pyplot as plt
import torch


def _to_numpy(img_tensor):
    if img_tensor.dim() == 4:
        img_tensor = img_tensor[0]
    img_tensor = img_tensor.detach().cpu().clamp(0.0, 1.0)
    img_np = img_tensor.permute(1, 2, 0).numpy()
    return img_np


def show_lr_sr_hr(lr, sr, hr):
    lr_np = _to_numpy(lr)
    sr_np = _to_numpy(sr)
    hr_np = _to_numpy(hr)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(lr_np)
    axs[0].set_title('LR')
    axs[0].axis('off')

    axs[1].imshow(sr_np)
    axs[1].set_title('SR')
    axs[1].axis('off')

    axs[2].imshow(hr_np)
    axs[2].set_title('HR')
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()


def plot_curves(history):
    fig, axs = plt.subplots(1, len(history), figsize=(5 * len(history), 4))
    if len(history) == 1:
        axs = [axs]

    for ax, (key, vals) in zip(axs, history.items()):
        ax.plot(vals.get('train', []), label='train')
        ax.plot(vals.get('val', []), label='val')
        ax.set_title(key)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()