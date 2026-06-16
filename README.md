# Differentiable CT Reconstruction & Motion Compensation Framework

A high-performance, end-to-end **differentiable CT backprojection reconstruction and motion compensation framework** built on top of **PyTorch** and **Numba CUDA**. This framework supports both 2D Fan-Beam and 3D Cone-Beam (CBCT) geometries, allowing analytical gradients to be backpropagated directly from the image space to projection geometry parameters (e.g., rotation angles, translation vectors, and projection matrices). It is ideally suited for online motion artifact correction, geometric calibration, and solving inverse problems integrated with deep learning priors.

## 🌟 Key Features

- **Parametric Geometric Optimization**: Supports direct gradient-based optimization of both 3x4 cone-beam projection matrices and explicit physical parameters (Angles, DSD, DSI, detector shifts).
- **Deep Learning Prior Integration (AI + Physics)**: Features an interleaved optimization interface compatible with **OT-CFM (Optimal Transport Conditional Flow Matching)** generative models, enabling self-supervised artifact correction guided by advanced data-driven priors.

---

## 📂 Repository Structure

```text 
models                          # models
config                          # config
up_parameter_small_detector     # Geometrical parameters of our WB-CBCT
├── backprojector_fan.py        # 2D differentiable fan-beam backprojector (PyTorch + Numba CUDA)
├── backprojector_cone.py       # 3D differentiable cone-beam backprojector (PyTorch + Numba CUDA)
├── geometry.py                 # Configuration class defining reconstruction volume and detector parameters
├── gradients.py                # CUDA gradient kernel functions for physical parameters (Angle, DSD, DSI, etc.)
├── helper.py                   # Coordinate transformations, 1D/2D CUDA bilinear interpolations, and projection matrix utilities
├── check_gradients.py          # Gradient verification script using PyTorch's `gradcheck`
├── conebeam_example_real.py    # Rigid motion artifact correction workflow on real specimen data (e.g., bone scans)
├── conebeam_example_simulation.py # Joint physics-AI optimization using simulated data and OT-CFM priors
├── Inference_utility.py        # Inference pipeline wrapper for the OT-CFM network (MotionMapNet + Dual-Decoder UNet)
├── dataload.py                 # PyTorch Dataset implementations for loading 2D/3D .tif medical images
├── train_MADN.py               # train_MADN
├── train_motionUnet.py         # train_motionUnet
└── dataload.py                 # dataloader


## 🙏 Acknowledgements

This project acknowledges and benefits from the following open-source repositories and academic works:

* **Motion Artifact Simulation**: The generation of our paired motion artifact simulation dataset was implemented using the Taichi-based forward/backward projection reconstruction framework (utilizing projection matrices) from [SEU-CT-Recon/Reconstruction_program_taichi](https://github.com/SEU-CT-Recon/Reconstruction_program_taichi).
* **Differentiable Backprojector**: The design and optimization methodology of our differentiable backprojection operators are referenced from and inspired by the following work:
  > Thies, M., Wagner, F., Maul, N., Yu, H., Goldmann, M., Schneider, L.S., Gu, M., Mei, S., Folle, L., Preuhs, A., et al. (2024). **A gradient-based approach to fast and accurate head motion compensation in cone-beam ct.** *IEEE Transactions on Medical Imaging*, 44(2), 1098–1109.

