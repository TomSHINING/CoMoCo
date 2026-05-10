from dataload import *
from torch.utils.data import DataLoader
from datetime import datetime
import torch.optim as optim
from config import config
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from utils import *
from models.Simple_Unet import MotionMapNet
import os

# -----------------------------------------
# GPU
# -----------------------------------------

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

t_config = config.get_config()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


# -----------------------------------------
# Train
# -----------------------------------------

def train_single_gpu():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # -----------------------------------------
    # TensorBoard
    # -----------------------------------------

    now_time = datetime.now()

    time_str = datetime.strftime(
        now_time,
        '%m-%d_%H-%M-%S'
    )

    log_dir = os.path.join(
        t_config.ResultPath,
        "motion_map_" + time_str
    )

    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)

    # -----------------------------------------
    # Model
    # -----------------------------------------

    model = MotionMapNet(
        in_ch=1,
        base_ch=32
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

    # -----------------------------------------
    # Resume
    # -----------------------------------------

    if os.path.exists(t_config.ModelSaveDir):

        latest_file = find_lastest_file(
            t_config.ModelSaveDir
        )

        if latest_file:

            checkpoint = torch.load(
                latest_file,
                map_location=device
            )

            model.load_state_dict(
                checkpoint['state_dict']
            )

            optimizer.load_state_dict(
                checkpoint['optimizer_state_dict']
            )

            print('Loaded checkpoint successfully!')

    # -----------------------------------------
    # Dataset
    # -----------------------------------------

    with open(
            './data/tiny_motion_list.txt',
            'r'
    ) as f:

        data_list = [
            line.strip()
            for line in f.readlines()
        ]

    train_dataset = MixPairedImageDataset(
        root_path=t_config.DataPath,
        data_list=data_list,
        restored_cond=t_config.use_restored_cond,
        image_shape=t_config.ImageShape
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=t_config.TrainBatchSize,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # -----------------------------------------
    # Loss
    # -----------------------------------------

    L1 = torch.nn.L1Loss()

    current_lr = optimizer.param_groups[0]['lr']

    print(f"Current learning rate: {current_lr}")

    # -----------------------------------------
    # Train Loop
    # -----------------------------------------

    for epoch in range(0, t_config.Epoch):

        model.train()

        meter_loss = AverageMeter()

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}"
        )

        for data in progress_bar:

            optimizer.zero_grad()

            x, cond = data

            x = x.to(
                device,
                non_blocking=True
            )

            cond = cond.to(
                device,
                non_blocking=True
            )

            # -----------------------------------------
            # motion gt
            # -----------------------------------------
            motion_gt = torch.abs(cond - x)
            motion_gt = torch.clamp(
                motion_gt * 3.0,
                0,
                1
            )

            motion_gt = motion_gt.detach()

            # -----------------------------------------
            # pred
            # -----------------------------------------

            pred_motion = model(cond)

            # -----------------------------------------
            # loss
            # -----------------------------------------

            loss_l1 = L1(
                pred_motion,
                motion_gt
            )

            # smooth regularization
            loss_smooth = (
                torch.mean(
                    torch.abs(
                        pred_motion[:, :, :, :-1] -
                        pred_motion[:, :, :, 1:]
                    )
                )
                +
                torch.mean(
                    torch.abs(
                        pred_motion[:, :, :-1, :] -
                        pred_motion[:, :, 1:, :]
                    )
                )
            )

            loss = loss_l1 + 0.1 * loss_smooth

            loss.backward()

            optimizer.step()

            meter_loss.update(loss.item())

            progress_bar.set_postfix({
                'loss': loss.item(),
            })

        scheduler.step()

        # -----------------------------------------
        # save
        # -----------------------------------------

        if epoch % 10 == 0:

            save_model(
                model,
                optimizer,
                epoch + 1,
                t_config.ModelSaveDir
            )

        # -----------------------------------------
        # tensorboard
        # -----------------------------------------

        writer.add_scalars(
            'train loss/motion',
            {'motion_loss': meter_loss.avg},
            epoch + 1
        )

        writer.add_image(
            'motion/x',
            normalization(x[0]),
            epoch + 1
        )

        writer.add_image(
            'motion/cond',
            normalization(cond[0]),
            epoch + 1
        )

        writer.add_image(
            'motion/gt',
            normalization(motion_gt[0]),
            epoch + 1
        )

        writer.add_image(
            'motion/pred',
            normalization(pred_motion[0]),
            epoch + 1
        )

        print(
            'Train Epoch: {}\t motion_loss: {:.6f}\t'.format(
                epoch + 1,
                meter_loss.avg
            )
        )

    writer.close()


# -----------------------------------------
# Main
# -----------------------------------------

if __name__ == "__main__":

    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    train_single_gpu()