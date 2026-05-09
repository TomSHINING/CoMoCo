import torch
import numpy as np
from torch.optim import SGD
from torch.nn import MSELoss
from geometry import Geometry
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from helper import params_3d_proj_matrix
from backprojector_cone import DifferentiableConeBeamBackprojector
import tifffile as tiff


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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

     # === Geometry ===
    geometry = Geometry((80, 512, 512),  # 3D volume now
                        (-300, -300, -290),
                        (5. / 0.417, 1.2, 1.2),
                        (-512 - 256, -512), (1, 1))



    file_path = "./up_parameter_small_detector/06/angle.seq"
    with open(file_path,'rb') as file:
        file.seek(0)
        angles = np.fromfile(file,dtype=np.float32)-2*np.pi/360*90
        angles = -angles

    file_path = "./up_parameter_small_detector/06/SOD.seq"
    with open(file_path, 'rb') as file:
        file.seek(0)
        dsd = np.fromfile(file, dtype=np.float32)/0.417

    file_path = "./up_parameter_small_detector/06/SysM.seq"
    with open(file_path, 'rb') as file:
        file.seek(0)
        SysM = np.fromfile(file, dtype=np.float32)
        dsi = dsd/SysM

    file_path = "./up_parameter_small_detector/06/DetX.seq"
    with open(file_path, 'rb') as file:
        file.seek(0)
        DetX = np.fromfile(file, dtype=np.float32)

    file_path = "./up_parameter_small_detector/06/DetY.seq"
    with open(file_path, 'rb') as file:
        file.seek(0)
        DetY = np.fromfile(file, dtype=np.float32)

    file_path = "./up_parameter_small_detector/06/pY.seq"
    with open(file_path, 'rb') as file:
        file.seek(0)
        pY = np.fromfile(file, dtype=np.float32)-1



    nu = nv = 1024
    dv = du = 0.417
    zSrc = (pY - DetY) * 1 / SysM
    tx = (nu/2-0.5-DetX)*0.7
    ty = (DetY-nv/2+0.5)*dv+(pY - DetY) * dv/ SysM


    true_proj_matrices, _, _ = params_3d_proj_matrix(
            angles, dsd, dsi, tx, ty, 1,(512*1+256*1,512*1),-zSrc
        )

    real_proj_matrices = torch.from_numpy(true_proj_matrices).float().to(device)


    # === Load sino ===
    sino1 = tiff.imread("./data/real_data/pig/pig_moved_leg_50mA_filter.tif")[:,:,::-1]
    sino = sino1.copy()
    sino = torch.from_numpy(sino).to(device)



    backprojector = DifferentiableConeBeamBackprojector.apply

    # === 3D motion parameters ===
    n_control = 150
    n_time = 300

    # 控制点（可学习参数）
    control_angles = torch.zeros(n_control, 3, device=device, requires_grad=True)
    control_translation = torch.zeros(n_control, 3, 1, device=device, requires_grad=True)

    angles = interpolate_control_to_time(control_angles, n_time)  # (400, 3)
    translation = interpolate_control_to_time(control_translation, n_time)  # (400, 3, 1)

    theta_x, theta_y, theta_z = angles[:, 0], angles[:, 1], angles[:, 2]
    #print(theta_x,theta_y,theta_z)

    # 构造绕 X 轴的旋转矩阵 Rx（此时 cos(0)=1, sin(0)=0 → 单位矩阵）
    Rx = torch.eye(3, device=device).unsqueeze(0).repeat(300, 1, 1)
    Rx[:, 1, 1] = torch.cos(theta_x)
    Rx[:, 1, 2] = -torch.sin(theta_x)
    Rx[:, 2, 1] = torch.sin(theta_x)
    Rx[:, 2, 2] = torch.cos(theta_x)

    # 构造绕 Y 轴的旋转矩阵 Ry
    Ry = torch.eye(3, device=device).unsqueeze(0).repeat(300, 1, 1)
    Ry[:, 0, 0] = torch.cos(theta_y)
    Ry[:, 0, 2] = torch.sin(theta_y)
    Ry[:, 2, 0] = -torch.sin(theta_y)
    Ry[:, 2, 2] = torch.cos(theta_y)

    # 构造绕 Z 轴的旋转矩阵 Rz
    Rz = torch.eye(3, device=device).unsqueeze(0).repeat(300, 1, 1)
    Rz[:, 0, 0] = torch.cos(theta_z)
    Rz[:, 0, 1] = -torch.sin(theta_z)
    Rz[:, 1, 0] = torch.sin(theta_z)
    Rz[:, 1, 1] = torch.cos(theta_z)

    # 合并旋转（此时 R = I @ I @ I = I）
    rotation = Rz @ Ry @ Rx  # shape: (400, 3, 3)


    bottom = torch.zeros((300,1,4), device=device)
    bottom[:,:,3] = 1   # make 4×4 rigid matrix




    motion = torch.cat(
                    (torch.cat((rotation, translation), dim=2), bottom),
                    dim=1
                 )   # shape (400,4,4)

    raw_proj_matrices = torch.einsum("nij,njk->nik", real_proj_matrices, motion)




    # === Initial reconstruction ===
    with torch.no_grad():
        initial_recon =  backprojector(sino, raw_proj_matrices, geometry)
        target_recon = torch.tensor(tiff.imread("./data/real_data/pig/real_fixed_leg.tif")).cuda()




    optimizer_angles = SGD([control_angles], lr=10)
    optimizer_translation = SGD([control_translation], lr=50)


    loss_fn = MSELoss()


    # === Optimization ===
    for i in range(n_iter):
        print('n_iter',i)

        optimizer_angles.zero_grad()
        optimizer_translation.zero_grad()

        angles = interpolate_control_to_time(control_angles, n_time)  # (400, 3)
        translation = interpolate_control_to_time(control_translation, n_time)  # (400, 3, 1)


        theta_x = angles[:, 0]
        theta_y = angles[:, 1]
        theta_z = angles[:, 2]
        # 构造绕 X 轴的旋转矩阵 Rx（此时 cos(0)=1, sin(0)=0 → 单位矩阵）
        Rx = torch.eye(3, device=device).unsqueeze(0).repeat(300, 1, 1)
        Rx[:, 1, 1] = torch.cos(theta_x)
        Rx[:, 1, 2] = -torch.sin(theta_x)
        Rx[:, 2, 1] = torch.sin(theta_x)
        Rx[:, 2, 2] = torch.cos(theta_x)

        # 构造绕 Y 轴的旋转矩阵 Ry
        Ry = torch.eye(3, device=device).unsqueeze(0).repeat(300, 1, 1)
        Ry[:, 0, 0] = torch.cos(theta_y)
        Ry[:, 0, 2] = torch.sin(theta_y)
        Ry[:, 2, 0] = -torch.sin(theta_y)
        Ry[:, 2, 2] = torch.cos(theta_y)

        # 构造绕 Z 轴的旋转矩阵 Rz
        Rz = torch.eye(3, device=device).unsqueeze(0).repeat(300, 1, 1)
        Rz[:, 0, 0] = torch.cos(theta_z)
        Rz[:, 0, 1] = -torch.sin(theta_z)
        Rz[:, 1, 0] = torch.sin(theta_z)
        Rz[:, 1, 1] = torch.cos(theta_z)

        # 合并旋转（此时 R = I @ I @ I = I）
        rotation = Rz @ Ry @ Rx  # shape: (400, 3, 3)



        motion = torch.cat((
                    torch.cat((rotation, translation), dim=2),
                    bottom),
                dim=1)

        perturbed = torch.einsum("nij,njk->nik", real_proj_matrices, motion)
        pred = backprojector(sino, perturbed, geometry)



        #loss = loss_fn(pred, target_recon)
        loss = loss_fn(pred[45:50,:,:], target_recon[45:50,:,:])
        losses.append(loss.item())  # 记录每个迭代的loss值
        print('loss:',loss.item())
        loss.backward()
        optimizer_translation.step()
        optimizer_angles.step()








    # === Final reconstruction ===
    with torch.no_grad():

        motion = torch.cat((
                    torch.cat((rotation, translation), dim=2),
                    bottom),
                dim=1)
        perturbed = torch.einsum("nij,njk->nik", real_proj_matrices, motion)
        #torch.save(perturbed, "data/real_data/pig/small_leg/perturbed_matrix.pt")
        recovered = backprojector(sino, perturbed, geometry)
        # initial_recon = backprojector(sino, real_proj_matrices, geometry)
        #
        # recovered =  recovered.detach().cpu().numpy()
        # tiff.imwrite("data/real_data/pig/small_leg/recovered_rec_leg.tif", recovered)
        #
        # initial_recon =  initial_recon.detach().cpu().numpy()
        # tiff.imwrite("data/real_data/pig/small_leg/initial_recon_rec_leg.tif", initial_recon)
        #
        # target_recon =  target_recon.detach().cpu().numpy()
        # tiff.imwrite("data/real_data/pig/small_leg/target_recon_rec_leg.tif", target_recon)





if __name__ == "__main__":
    main()
