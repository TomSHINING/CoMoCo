import torch
from models.Simple_Unet import UNet_Dual_Decoder as simple_unet_dual_decoder
from models.Simple_Unet import MotionMapNet
from utils import find_lastest_file, load_model


# 全局变量用于缓存模型，避免重复加载
_MODEL_CACHE = {}




# 全局缓存，避免重复加载权重
_MODEL_CACHE = {}


def predictOT_CFM(input_tensor, K=8):
    """
    One-line call function: Processes (D, H, W) Tensor on CUDA.
    Returns (D, H, W) result Tensor on CUDA.
    """
    global _MODEL_CACHE
    device = input_tensor.device

    # 1. Model Loading (Lazy initialization: only loads on the first call)
    if 'm_net' not in _MODEL_CACHE:
        # Initialize and load MotionMapNet
        m_net = MotionMapNet().to(device)
        m_ckpt = torch.load("/home/lintong/Supervised_flow/Checkpoints/motion_Unet/checkpoints/model_at_epoch_011.dat",
                            map_location=device)
        m_net.load_state_dict(m_ckpt['state_dict'])
        m_net.eval()

        # Initialize and load Main UNet model
        u_net = simple_unet_dual_decoder(
            ch=32, ch_mult=[1, 2, 4, 8], attn=[], num_res_blocks=2, dropout=0.0, condition=True
        ).to(device)
        ckpt_path = find_lastest_file('/home/lintong/Supervised_flow/Checkpoints/MADN/checkpoints/')
        u_net = load_model(u_net, torch.load(ckpt_path, map_location=device))
        u_net.eval()

        # Cache models to avoid redundant GPU memory allocation
        _MODEL_CACHE['m_net'] = m_net
        _MODEL_CACHE['u_net'] = u_net

    m_net = _MODEL_CACHE['m_net']
    u_net = _MODEL_CACHE['u_net']

    # 2. Data Preparation: (D, H, W) -> (D, 1, H, W)
    # Ensure float32 type and add channel dimension
    cond_all = input_tensor.unsqueeze(1).float()
    D = cond_all.shape[0]
    results = []

    # 3. Slice-wise Inference
    with torch.no_grad():
        for d in range(D):
            cond = cond_all[d:d + 1]

            jitter_map = m_net(cond)

            x_t = torch.randn_like(cond)

            for k in range(K):
                t_val = k / K
                t_tensor = torch.full((1,), t_val, device=device)


                _, pred_x0 = u_net(x_t, t_tensor, cond, jitter_map)

                if pred_x0.shape[1] > 1:
                    pred_x0 = pred_x0[:, :1, :, :]

                # ODE Step: v = (x0 - xt) / (1 - t)
                v = (pred_x0 - x_t) / (1.0 - t_val + 1e-6)

                # Update x_t for the next time step
                x_t = x_t + (1.0 / K) * v

            results.append(x_t)

    # 4. Concatenate slices and squeeze back to (D, H, W)
    return torch.cat(results, dim=0).squeeze(1)