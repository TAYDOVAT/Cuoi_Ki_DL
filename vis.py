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
    n_plots = len(history)
    ncols = min(3, n_plots)
    nrows = (n_plots + ncols - 1) // ncols
    
    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if n_plots == 1:
        axs = [axs]
    else:
        axs = axs.flatten()

    for i, (key, vals) in enumerate(history.items()):
        ax = axs[i]
        ax.plot(vals.get('train', []), label='train')
        ax.plot(vals.get('val', []), label='val')
        ax.set_title(key)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    # Hide unused axes
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')

    plt.tight_layout()
    plt.show()