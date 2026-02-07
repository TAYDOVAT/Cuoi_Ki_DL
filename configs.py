# Minimal configs; override in notebooks as needed.
CFG = {
    'scale': 4,
    'hr_crop': 128,
    'train': {
        'batch_size': 16,
        'num_workers': 4,
        'epochs': 100,
        'lr': 1e-4,
    },
    'gan': {
        'batch_size': 8,
        'num_workers': 4,
        'epochs': 200,
        'lr_g': 1e-4,
        'lr_d': 1e-4,
        'adv_weight': 1e-3,
        'perc_weight': 6e-3,
        'pixel_weight': 1.0,
        'r1_weight': 0.0,
    },
    'paths': {
        'train_lr': '../../input/anh-ve-tinh-2/Anh_ve_tinh_2/train/train_lr',
        'train_hr': '../../input/anh-ve-tinh-2/Anh_ve_tinh_2/train/train_hr',
        'val_lr': '../../input/anh-ve-tinh-2/Anh_ve_tinh_2/val/val_lr',
        'val_hr': '../../input/anh-ve-tinh-2/Anh_ve_tinh_2/val/val_hr',
        'test_lr': '../../input/anh-ve-tinh-2/Anh_ve_tinh_2/test/test_lr',
        'test_hr': '../../input/anh-ve-tinh-2/Anh_ve_tinh_2/test/test_hr',
    },
}
