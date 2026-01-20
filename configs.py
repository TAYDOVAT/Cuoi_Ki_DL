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
    },
    'paths': {
        'train_lr': 'train/train_lr',
        'train_hr': 'train/train_hr',
        'val_lr': 'val/val_lr',
        'val_hr': 'val/val_hr',
        'test_lr': 'test/test_lr',
        'test_hr': 'test/test_hr',
    },
}
