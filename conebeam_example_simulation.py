from torch.optim import SGD
from torch.nn import MSELoss
from geometry import Geometry
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from helper import params_3d_proj_matrix
from backprojector_cone import DifferentiableConeBeamBackprojector
from Inference_utility import *
import tifffile as tiff
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import time
import numpy as np



os.environ["CUDA_VISIBLE_DEVICES"] = "0"
plt.ion()
device = torch.device('cuda')
n_iter = 300





def setup_plotting(initial_reconstruction):
    x = list(range(n_iter))
    y = [None for i in range(n_iter)]
    fig, ax = plt.subplots(ncols=2, figsize=(8, 3.4))
    image = ax[0].imshow(initial_reconstruction.cpu().numpy(), cmap='gray', vmin=0, vmax=20)
    ax[0].axis('off')
    ax[0].set_title('Reconstruction')
    loss_plot = ax[1].plot(x, y)
    ax[1].set_title('Loss')
    ax[1].set_xlim([0, n_iter])
    ax[1].set_ylim([0, 20])
    ax[1].set_xlabel('Iterations')
    plt.tight_layout()
    plt.draw()
    return image, loss_plot, x, y

def interpolate_control_to_time_smooth(control_tensor, target_len, smoothing_sigma=2.0):
    """
    Smooth interpolation using linear interpolation + differentiable Gaussian smoothing.
    Works for control_tensor of shape (N_ctrl, C, ...).
    """
    orig_shape = control_tensor.shape
    N_ctrl = orig_shape[0]
    C_rest = orig_shape[1:]

    # Flatten non-control dims: (N_ctrl, C_prod)
    control_flat = control_tensor.view(N_ctrl, -1).permute(1, 0).unsqueeze(0)  # (1, C_prod, N_ctrl)

    # Linear interpolation (only valid 1D mode in F.interpolate for this shape)
    interpolated = F.interpolate(
        control_flat,
        size=target_len,
        mode='linear',
        align_corners=True
    )  # (1, C_prod, target_len)

    # Reshape back to (target_len, C, ...)
    interpolated = interpolated.squeeze(0).permute(1, 0)  # (target_len, C_prod)
    interpolated = interpolated.view(target_len, *C_rest)

    # Gaussian smoothing along time dimension (dim=0)
    if smoothing_sigma > 0:
        kernel_size = int(2 * np.ceil(3 * smoothing_sigma) + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        ax = torch.arange(kernel_size, dtype=torch.float32, device=control_tensor.device) - kernel_size // 2
        gauss = torch.exp(-0.5 * (ax / smoothing_sigma) ** 2)
        gauss = gauss / gauss.sum()
        gauss = gauss.view(1, 1, -1)  # (1, 1, K)

        T = target_len
        C_total = interpolated.shape[1:].numel()

        # (T, C_total) --> (1, C_total, T)
        smoothed = interpolated.view(T, C_total).permute(1, 0).unsqueeze(0)  # (1, C_total, T)

        pad = kernel_size // 2
        smoothed = F.pad(smoothed, (pad, pad), mode='reflect')
        smoothed = F.conv1d(smoothed, gauss.expand(C_total, 1, -1), groups=C_total)  # (1, C_total, T)

        # Back to (T, C_total) --> original shape
        smoothed = smoothed.squeeze(0).permute(1, 0)  # (T, C_total)
        interpolated = smoothed.view_as(interpolated)

    return interpolated

def interpolate_control_to_time(control_tensor, target_len):
    """
    control_tensor: shape (N_ctrl, C, ...)  --> we treat N_ctrl as "spatial" dim for interpolation
    returns: (target_len, C, ...)
    """
    # Add batch dim and move control dim to last for interpolation
    # Input to interpolate: (batch=1, C*..., N_ctrl)
    orig_shape = control_tensor.shape  # e.g., (20, 3) or (20, 3, 1)
    C_rest = orig_shape[1:]
    control_flat = control_tensor.view(orig_shape[0], -1).permute(1, 0).unsqueeze(0)  # (1, C_prod, N_ctrl)

    # Interpolate along last dimension (N_ctrl -> target_len)
    interpolated = torch.nn.functional.interpolate(
        control_flat,
        size=target_len,
        mode='linear',  # or 'linear' if you want linear; bicubic is smoother
        align_corners=True
    )  # (1, C_prod, target_len)

    # Reshape back: (target_len, C, ...)
    interpolated = interpolated.squeeze(0).permute(1, 0)  # (target_len, C_prod)
    interpolated = interpolated.view(target_len, *C_rest)
    return interpolated


def main():

    losses = []

    # === Geometry ===
    geometry = Geometry((80, 512, 512),  # 3D volume now
                        (-180, -255.5, -255.5),
                        (5., 0.7, 0.7),
                        (-767., -511.), (1., 1.))

    angles = -np.linspace(0, 2*np.pi, 400, endpoint=False)
    dsd = 1300 * np.ones_like(angles)
    dsi = 670 * np.ones_like(angles)

    tx = np.zeros_like(angles)
    ty = np.zeros_like(angles)

    real_proj_matrices, _, _ = params_3d_proj_matrix(
        angles, dsd, dsi, tx, ty, 1.0,(767/1.0,511/1.0),0
    )

    real_proj_matrices = torch.from_numpy(real_proj_matrices).float().to(device)


    # === Load sino ===
    sino = tiff.imread("./data/proj/filtered_projections.tif")
    sino = torch.from_numpy(sino).to(device)
    backprojector = DifferentiableConeBeamBackprojector.apply

    # === 3D motion parameters ===
    n_control = 40
    n_time = 400

    # 控制点（可学习参数）
    control_angles = torch.zeros(n_control, 3, device=device, requires_grad=True)
    control_translation = torch.zeros(n_control, 3, 1, device=device, requires_grad=True)




    angles = interpolate_control_to_time_smooth(control_angles, n_time)  # (400, 3)
    translation = interpolate_control_to_time_smooth(control_translation, n_time)  # (400, 3, 1)

    theta_x, theta_y, theta_z = angles[:, 0], angles[:, 1], angles[:, 2]



    Rx = torch.eye(3, device=device).unsqueeze(0).repeat(400, 1, 1)
    Rx[:, 1, 1] = torch.cos(theta_x)
    Rx[:, 1, 2] = -torch.sin(theta_x)
    Rx[:, 2, 1] = torch.sin(theta_x)
    Rx[:, 2, 2] = torch.cos(theta_x)


    Ry = torch.eye(3, device=device).unsqueeze(0).repeat(400, 1, 1)
    Ry[:, 0, 0] = torch.cos(theta_y)
    Ry[:, 0, 2] = torch.sin(theta_y)
    Ry[:, 2, 0] = -torch.sin(theta_y)
    Ry[:, 2, 2] = torch.cos(theta_y)


    Rz = torch.eye(3, device=device).unsqueeze(0).repeat(400, 1, 1)
    Rz[:, 0, 0] = torch.cos(theta_z)
    Rz[:, 0, 1] = -torch.sin(theta_z)
    Rz[:, 1, 0] = torch.sin(theta_z)
    Rz[:, 1, 1] = torch.cos(theta_z)


    rotation = Rz @ Ry @ Rx  # shape: (400, 3, 3)


    bottom = torch.zeros((400,1,4), device=device)
    bottom[:,:,3] = 1   # make 4×4 rigid matrix




    motion = torch.cat((torch.cat((rotation, translation), dim=2), bottom),dim=1)   # shape (400,4,4)
    raw_proj_matrices = torch.einsum("nij,njk->nik", real_proj_matrices, motion)




    # === Initial reconstruction ===
    with torch.no_grad():
        initial_recon =  backprojector(sino, raw_proj_matrices, geometry)
        target_recon = predictOT_CFM(initial_recon)
        #target_recon = torch.tensor(tiff.imread("/home/lintong/Supervised_flow/test_output/Severe_input_result.tif")).cuda()




    optimizer_angles = SGD([control_angles], lr=1)
    optimizer_translation = SGD([control_translation], lr=2)

    scheduler_angles = ReduceLROnPlateau(optimizer_angles, 'min', patience=5, factor=0.5)
    scheduler_translation = ReduceLROnPlateau(optimizer_translation, 'min', patience=5, factor=0.5)


    loss_fn = MSELoss()

    start_time = time.time()  
    # === Optimization ===
    for i in range(n_iter):
        print('n_iter',i)
        iter_start = time.time()  
        optimizer_angles.zero_grad()
        optimizer_translation.zero_grad()

        angles = interpolate_control_to_time_smooth(control_angles, n_time)  # (400, 3)
        translation = interpolate_control_to_time_smooth(control_translation, n_time)  # (400, 3, 1)


        theta_x = angles[:, 0]
        theta_y = angles[:, 1]
        theta_z = angles[:, 2]
        
        Rx = torch.eye(3, device=device).unsqueeze(0).repeat(400, 1, 1)
        Rx[:, 1, 1] = torch.cos(theta_x)
        Rx[:, 1, 2] = -torch.sin(theta_x)
        Rx[:, 2, 1] = torch.sin(theta_x)
        Rx[:, 2, 2] = torch.cos(theta_x)

       
        Ry = torch.eye(3, device=device).unsqueeze(0).repeat(400, 1, 1)
        Ry[:, 0, 0] = torch.cos(theta_y)
        Ry[:, 0, 2] = torch.sin(theta_y)
        Ry[:, 2, 0] = -torch.sin(theta_y)
        Ry[:, 2, 2] = torch.cos(theta_y)

        
        Rz = torch.eye(3, device=device).unsqueeze(0).repeat(400, 1, 1)
        Rz[:, 0, 0] = torch.cos(theta_z)
        Rz[:, 0, 1] = -torch.sin(theta_z)
        Rz[:, 1, 0] = torch.sin(theta_z)
        Rz[:, 1, 1] = torch.cos(theta_z)

        
        rotation = Rz @ Ry @ Rx  # shape: (400, 3, 3)



        motion = torch.cat((
                    torch.cat((rotation, translation), dim=2),
                    bottom),
                dim=1)

        perturbed = torch.einsum("nij,njk->nik", real_proj_matrices, motion)
        pred = backprojector(sino, perturbed, geometry)
        target_recon = predictOT_CFM(pred)





        loss = loss_fn(pred, target_recon)
        losses.append(loss.item())
        print('loss:',loss.item())
        loss.backward()
        optimizer_translation.step()
        optimizer_angles.step()

        scheduler_angles.step(loss)
        scheduler_translation.step(loss)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        iter_end = time.time()
        print(f'| loss: {loss.item():.6f} | time: {iter_end - iter_start:.3f}s')


    # === Final reconstruction ===
    with torch.no_grad():
        motion = torch.cat((
                    torch.cat((rotation, translation), dim=2),
                    bottom),
                dim=1)
        perturbed = torch.einsum("nij,njk->nik", real_proj_matrices, motion)
        recovered = backprojector(sino, perturbed, geometry)



        # recovered =  recovered.detach().cpu().numpy()
        # tiff.imwrite("data/recovered_rec_leg.tif", recovered)
        #
        # initial_recon =  initial_recon.detach().cpu().numpy()
        # tiff.imwrite("data/initial_recon_rec_leg.tif", initial_recon)
        #
        # target_recon =  target_recon.detach().cpu().numpy()
        # tiff.imwrite("data/target_recon_rec_leg.tif", target_recon)






if __name__ == "__main__":
    main()
