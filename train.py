import argparse
import logging
import os
from pathlib import Path

import torch
from thop.vision.basic_hooks import count_parameters

from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from DataLoader.dataloader import BasicDataset
from Loss.Dice_Loss import DiceLoss
from Loss.Tversky_Loss import TverskyLoss
from eval import evaluate


# 利用自己的wandb账号对训练过程进行监测，对实验过程及结果进行记录
os.environ["WANDB_API_KEY"] = ""

# 设置'online'进行在线监测，或'offline'进行离线监测
os.environ["WANDB_MODE"] = ""

# 数据集的图片及其标注所在位置
dir_img = r''
dir_mask = r''

# 模型参数保存位置
dir_checkpoint = Path('')

def train_model(
        model,
        device,
        epochs: int = 2,
        batch_size: int = 4,
        learning_rate: float = 1e-7,
        val_percent: float = 0.2,
        save_checkpoint: bool = True,
        img_scale: float = 0.8,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        dir_checkpoint: str = "Checkpoint",
        load_path: str = None,
        start_epoch: int = 1,
):
    dir_checkpoint = Path(dir_checkpoint)
    # 1. Create dataset
    try:
        dataset = BasicDataset(dir_img, dir_mask)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    config_dict = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "val_percent": val_percent,
        "save_checkpoint": save_checkpoint,
        "amp": amp,
        "total_params": param_info['Total Params (M)'],
        "trainable_params": param_info['Trainable Params (M)'],

        "Training size": n_train,
        "Validation size": n_val,
        "optimizer": "Adam",
        "lr": learning_rate,
        "weight_decay": weight_decay,
        "momentum": momentum,
        "scheduler": "ReduceLROnPlateau",
        "scheduler_patience": 5,
        "loss_function": "CrossEntropy" if model.n_classes > 1 else "BCEWithLogits"
    }
    experiment = wandb.init(
        project="", #设置自己的项目名
        name="", #项目中这个实验的名称
        config=config_dict,
        resume="never",  # 如果你需要接着之前的 run 继续
        id="",
        anonymous="never", # 可以改成 'never' 绑定到你账户
        settings = wandb.Settings(init_timeout=30)  # 设置为 120 秒或更长
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.amp.GradScaler(enabled=amp)
    # criterion = get_weighted_ce_loss(device)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # boundary_loss_fn = BoundaryLoss()
    tversky_loss_fn = TverskyLoss(alpha=0.3, beta=0.7, gamma=1.0)  # 可调

    # 加载预训练权重（如果有）
    if load_path:
        checkpoint = torch.load(load_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'scaler_state_dict' in checkpoint:
                grad_scaler.load_state_dict(checkpoint['scaler_state_dict'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
            logging.info(f" Loaded full checkpoint from {load_path}, starting at epoch {start_epoch}")
        else:
            model.load_state_dict(checkpoint)
            logging.info(f" Loaded state_dict weights from {load_path}")

    # 训练过程
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)

                    if model.n_classes == 1:
                        # 二分类
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += DiceLoss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)

                    else:
                        # 多分类
                        ce_loss = criterion(masks_pred, true_masks)
                        probs = F.softmax(masks_pred, dim=1).float()
                        one_hot = F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float()

                        loss = 0.5 * ce_loss
                        # loss += 0.5 * dice_loss(probs, one_hot, multiclass=True)
                        loss += 0.5 * tversky_loss_fn(probs, one_hot)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()

                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })

                pbar.set_postfix(**{'loss (batch)': loss.item()})

        #  每个 epoch 结束后保存模型
        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': grad_scaler.state_dict()
            }
            torch.save(checkpoint, str(dir_checkpoint / f'checkpoint_epoch{epoch}.pth'))

        #  每个 epoch 结束后进行一次验证
        metrics = evaluate(model, val_loader, device, amp)

        # combined_score = 0.4 * metrics['dice'] + 0.3 * metrics['mpa'] + 0.3 * metrics['miou']
        combined_score = 0.5 * metrics['dice'] +  0.5 * metrics['miou']
        scheduler.step(combined_score)

        logging.info(f"""
            Validation Metrics after Epoch {epoch}:
            Dice Score            : {metrics['dice']:.4f}
            Mean Pixel Accuracy   : {metrics['mpa']:.4f}
            Mean IoU              : {metrics['miou']:.4f}
            Boundary F1 Score     : {metrics['bf']:.4f}
            Global Accuracy       : {metrics['gacc']:.4f}
            Combined Score (LR)   : {combined_score:.4f}
            Current Learning Rate : {optimizer.param_groups[0]["lr"]:.8f}
        """)
        experiment.log({
            'validation Dice': metrics['dice'],
            'validation MPA': metrics['mpa'],
            'validation mIOU': metrics['miou'],
            'validation BF_Score': metrics['bf'],
            'validation GAcc': metrics['gacc'],
            'combined score (scheduler input)': combined_score,
            'Learning rate': float(optimizer.param_groups[0]["lr"]),
            'step': global_step,
            'epoch': epoch
        })

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=4, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-6,help='Learning rate', dest='lr')

    parser.add_argument('--scale', '-s', type=float, default=1.0, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=20.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=8, help='Number of classes')

    #在此处可添加预训练的模型权重
    parser.add_argument('--load', type=str, default=r" ",
                        help='Path to .pth file to load weights')

    parser.add_argument('--start-epoch', type=int, default=88, help='Epoch number to start training from')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    model = Model(n_channels=3, n_classes=args.classes,bilinear = False)

    model = model.to(memory_format=torch.channels_last)
    param_info = count_parameters(model)

    logging.info(f"""
    Network Summary:
        Input Channels         : {model.n_channels}
        Output Channels        : {model.n_classes} (classes)
        Total Parameters       : {param_info['Total Params (M)']:.2f} M
        Trainable Parameters   : {param_info['Trainable Params (M)']:.2f} M
    """)

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            load_path=args.load,
            start_epoch=args.start_epoch
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
