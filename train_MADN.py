from dataload import *
from torch.utils.data import DataLoader
from datetime import datetime
import torch.optim as optim
from config import config
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm


from utils import *
from models.Simple_Unet import UNet_Dual_Decoder as simple_unet_dual_decoder
from models.Simple_Unet import MotionMapNet
import os

# 强制使用第 0 块 GPU（单卡）
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

t_config = config.get_config()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def get_t_and_d(bs, device):
    set_d = [128, 64, 32, 16, 8, 4, 2, 1]
    index = torch.randint(0, len(set_d), (1,))
    max_val = set_d[index]
    d = 1 / (torch.ones(bs, device=device) * max_val)
    t = torch.randint(0, max_val, size=(bs,), device=device)
    t = t / max_val
    d = d.view(-1, 1, 1, 1)
    t = t.view(-1, 1, 1, 1)
    return t, d


# 单卡训练函数（不再需要 rank/world_size）
def train_single_gpu():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 日志设置（仅主进程，现在就是唯一进程）
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
    log_dir = os.path.join(t_config.ResultPath, time_str)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    motion_net = MotionMapNet().to(device)
    motion_ckpt = torch.load(
        "./Checkpoints/motion_Unet/checkpoints/model_at_epoch_101.dat",
        map_location=device
    )
    motion_net.load_state_dict(motion_ckpt['state_dict'])
    motion_net.eval()

    # 模型定义
    model = simple_unet_dual_decoder(
        ch=32,
        ch_mult=[1, 2, 4, 8],
        attn=[],
        num_res_blocks=2,
        dropout=0.0,
        condition=t_config.use_cond
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=t_config.lr,
        betas=(t_config.beta1, t_config.beta2),
        eps=1e-8,
        weight_decay=t_config.weight_decay
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=t_config.StepSize,
        gamma=t_config.Gamma
    )

    # 加载 checkpoint（如果存在）
    if os.path.exists(t_config.ModelSaveDir):
        latest_file = find_lastest_file(t_config.ModelSaveDir)
        if latest_file:
            checkpoint = torch.load(latest_file, map_location=device)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('Loaded checkpoint successfully!')

    # 数据集加载
    with open('/home/lintong/Supervised_flow/data/tiny_motion_list.txt', 'r') as f:
        data_list = [line.strip() for line in f.readlines()]

    if t_config.use_cond and not t_config.use_level:
        print("go")
        train_dataset = MixPairedImageDataset(
            root_path=t_config.DataPath,
            data_list=data_list,
            restored_cond=t_config.use_restored_cond,
            image_shape=t_config.ImageShape
        )
    elif t_config.use_level:
        train_dataset = Multi_Level_ImageDataset(
            root_path=t_config.DataPath,
            data_list=data_list,
            image_shape=t_config.ImageShape,
            levels=3
        )
    else:
        train_dataset = ImageDataset(
            root_path=t_config.DataPath,
            data_list=data_list,
            image_shape=t_config.ImageShape
        )

    # 单卡用普通 DataLoader，shuffle=True
    train_loader = DataLoader(
        train_dataset,
        batch_size=t_config.TrainBatchSize,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    MSE = torch.nn.MSELoss()
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Current learning rate: {current_lr}")

    # 开始训练
    for epoch in range(0, t_config.Epoch):
        model.train()
        meter_flow = AverageMeter()
        meter_denoise = AverageMeter()
        meter_restored = AverageMeter()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        sparse_phi = torch.zeros((720, 1)).to(device)

        for data in progress_bar:
            optimizer.zero_grad()

            if t_config.use_cond:
                x, cond = data
                x = x.to(device, non_blocking=True)
                cond = cond.to(device, non_blocking=True)
            else:
                x = data.to(device, non_blocking=True)

            guassian_noise = torch.randn_like(x)

            if t_config.use_distance:
                t, d = get_t_and_d(x.shape[0], x.device)
                d[:t_config.FMKBatchSize] = 0.0
            else:
                t = torch.rand((x.shape[0], 1, 1, 1), device=x.device) * (1 - 0.01) + 0.01

            x_t = t * x + (1.0 - t) * guassian_noise

            if not t_config.use_distance:
                with torch.no_grad():
                    jitter_map = motion_net(cond)
                denoised_x0, restored_x0 = model(x_t,t[:, 0, 0, 0],cond,jitter_map)
                loss_denoise = MSE(denoised_x0, x)
                loss_restored = MSE(restored_x0, x)
                loss = loss_denoise + loss_restored
            else:
                # FMK 部分（保持原逻辑）
                x_fm = x_t[:t_config.FMKBatchSize]
                s_target_1 = x[:t_config.FMKBatchSize] - guassian_noise[:t_config.FMKBatchSize]
                with torch.no_grad():
                    x_shortcut = x_t[t_config.FMKBatchSize:]
                    d_shortcut = d[t_config.FMKBatchSize:]
                    t_shortcut = t[t_config.FMKBatchSize:]
                    cond_shortcut = cond[t_config.FMKBatchSize:]
                    sparse_phi_shortcut = sparse_phi[t_config.FMKBatchSize:]

                    v1 = model(x_shortcut, t_shortcut[:, 0, 0, 0], cond_shortcut, sparse_phi_shortcut,
                               d_shortcut[:, 0, 0, 0])
                    t2 = t_shortcut + d_shortcut
                    x_t2 = x_shortcut + d_shortcut * v1
                    v2 = model(x_t2, t2[:, 0, 0, 0], cond_shortcut, sparse_phi_shortcut, d_shortcut[:, 0, 0, 0])
                    s_target_2 = (v1 + v2) / 2

                x_cat = torch.cat([x_fm, x_shortcut], dim=0)
                s_target = torch.cat([s_target_1, s_target_2], dim=0)
                pred_s = model(x_cat, t[:, 0, 0, 0], cond, sparse_phi, d[:, 0, 0, 0] * 2)
                loss = MSE(s_target, pred_s)
                # 注意：此时 loss_denoise/loss_restored 未定义，避免记录
                meter_denoise.update(0)
                meter_restored.update(0)

            meter_flow.update(loss.item())
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix({
                't': float(t[0, 0, 0, 0]),
                'loss': loss.item(),
            })

        scheduler.step()

        # 保存模型 & 写日志（每轮都做）
        if epoch % 10 ==0:
            save_model(model, optimizer, epoch + 1, t_config.ModelSaveDir)
        writer.add_scalars('train loss/flow', {'train_flow_loss': meter_flow.avg}, epoch + 1)
        if not t_config.use_distance:
            writer.add_scalars('train loss/IR', {'train_denoised_loss': meter_denoise.avg}, epoch + 1)
            writer.add_scalars('train loss/IR', {'train_restored_loss': meter_restored.avg}, epoch + 1)

        # 可视化（取第一个样本）
        writer.add_image('train img/x img', normalization(x[0]), epoch + 1)
        writer.add_image('train img/x_t img', normalization(x_t[0]), epoch + 1)
        if t_config.use_cond:
            writer.add_image('train img/cond img', normalization(cond[0]), epoch + 1)
            writer.add_image('train img/denoised img', normalization(denoised_x0[0]), epoch + 1)
            writer.add_image('train img/restored img', normalization(restored_x0[0]), epoch + 1)

        print('Train Epoch: {}\t train_loss: {:.6f}\t'.format(epoch + 1, meter_flow.avg))

    writer.close()


if __name__ == "__main__":
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # 直接调用单卡训练
    train_single_gpu()