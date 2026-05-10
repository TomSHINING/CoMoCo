from torch.utils.data import Dataset
import os
import numpy as np
import torch
import tifffile
class ImageDataset(Dataset):
    def __init__(self, root_path, data_list, image_shape=[512,512]):
        super().__init__()
        self.image_shape = image_shape
        self.root_path = root_path
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        file_name = self.data_list[index]
        # input = np.fromfile(os.path.join(self.root_path, file_name), dtype=np.float32)
        input = tifffile.imread(os.path.join(self.root_path, file_name))
        input = torch.FloatTensor(input).reshape(1, self.image_shape[0], self.image_shape[1])
        
        return input


class PairedImageDataset(Dataset):
    def __init__(self, root_path, data_list, restored_cond=False, image_shape=[512,512]):
        super().__init__()
        self.image_shape = image_shape
        self.root_path = root_path
        self.data_list = data_list
        self.restored_cond = restored_cond

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        file_name = self.data_list[index]
        # input = np.fromfile(os.path.join(self.root_path, file_name), dtype=np.float32)
        
        try:
            label = tifffile.imread(os.path.join(self.root_path, 'tiny_label_512_2d_tif', file_name))
            if not self.restored_cond:
                input = tifffile.imread(os.path.join(self.root_path, 'big_input_512_2d_tif', file_name))
            else:
                input = tifffile.imread(os.path.join(self.root_path, 'supervised_cond_sp60', file_name))
            label = torch.FloatTensor(label).reshape(1, self.image_shape[0], self.image_shape[1])
            input = torch.FloatTensor(input).reshape(1, self.image_shape[0], self.image_shape[1])

        except Exception as e:
            print(f"Error loading index {index}: {e}")
            return torch.zeros((1, self.image_shape[0], self.image_shape[1])), torch.zeros((1, self.image_shape[0], self.image_shape[1]))
        
        
        return label, input

class Refine_Dataset(Dataset):
    def __init__(self, root_path, image_shape=[512,512]):
        super().__init__()
        self.image_shape = image_shape
        self.root_path = root_path
        self.data_list = os.listdir(self.root_path)

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        file_name = self.data_list[index]
        k = file_name.split('_')[-1].split('.')[0]
        k = torch.FloatTensor([int(k) / 128])
        # print(k)
        # input = np.fromfile(os.path.join(self.root_path, file_name), dtype=np.float32)
        data = tifffile.imread(os.path.join(self.root_path, file_name))
        data = torch.FloatTensor(data).reshape(4,1,self.image_shape[0], self.image_shape[1])
        return {'noise':data[0], 'x_t':data[1], 'degraded_img':data[2], 'reference_img':data[3], 'k':k}


class ImageDataset_Test(Dataset):
    def __init__(self, root_path, proj_path, data_list, image_shape=[512,512]):
        super().__init__()
        self.image_shape = image_shape
        self.root_path = root_path
        self.proj_path = proj_path
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        file_name = self.data_list[index]
        # input = np.fromfile(os.path.join(self.root_path, file_name), dtype=np.float32)
        input = tifffile.imread(os.path.join(self.root_path, file_name))
        input = torch.FloatTensor(input).reshape(1,self.image_shape[0],self.image_shape[1])
        if self.proj_path is not None:
            proj = tifffile.imread(os.path.join(self.proj_path, file_name))
            proj = torch.FloatTensor(proj).reshape(1,720, 512)
        
        return input, proj, file_name


class Multi_Level_ImageDataset(Dataset):
    def __init__(self, root_path, data_list, levels=3, image_shape=[512,512]):
        super().__init__()
        self.image_shape = image_shape
        self.root_path = root_path
        self.data_list = data_list
        self.levels = levels

    def __len__(self):
        return len(self.data_list) * self.levels
    
    def __getitem__(self, index):
        file_name = self.data_list[index // self.levels]
        # input = np.fromfile(os.path.join(self.root_path, file_name), dtype=np.float32)
        label = tifffile.imread(os.path.join(self.root_path, 'Label_Iter_Recon', file_name))
        sparse_ratio = 0
        if index % self.levels==0:
            input = tifffile.imread(os.path.join(self.root_path, 'sp60_views_2d', file_name))
            sparse_ratio = 12
        if index % self.levels==1:
            input = tifffile.imread(os.path.join(self.root_path, 'sp72_views_2d', file_name))
            sparse_ratio = 10
        else:
            input = tifffile.imread(os.path.join(self.root_path, 'sp90_views_2d', file_name))
            sparse_ratio = 8

        phi = torch.linspace(0, torch.pi * 2 - 1e-8, 720)
        sparse_phi = torch.zeros_like(phi)
        sparse_phi[::sparse_ratio] = phi[::sparse_ratio]
        label = torch.FloatTensor(label).reshape(1, self.image_shape[0], self.image_shape[1])
        input = torch.FloatTensor(input).reshape(1, self.image_shape[0], self.image_shape[1])
        
        return label, input, sparse_phi



class MixPairedImageDataset(Dataset):
    def __init__(
        self,
        root_path,
        data_list,
        input_dirs=('tiny_input_512_2d_tif', 'big_input_512_2d_tif'),
        restored_cond=False,
        image_shape=[256, 256]
    ):
        super().__init__()
        self.root_path = root_path
        self.image_shape = image_shape
        self.restored_cond = restored_cond
        self.input_dirs = input_dirs

        # 将 (file_name, input_dir) 展开成新的索引列表
        self.samples = []
        for fname in data_list:
            for idir in input_dirs:
                self.samples.append((fname, idir))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        file_name, input_dir = self.samples[index]

        try:
            label = tifffile.imread(
                os.path.join(self.root_path, 'tiny_label_512_2d_tif', file_name)
            )

            if not self.restored_cond:
                input_img = tifffile.imread(
                    os.path.join(self.root_path, input_dir, file_name)
                )
            else:
                input_img = tifffile.imread(
                    os.path.join(self.root_path, 'supervised_cond_sp60', file_name)
                )

            label = torch.FloatTensor(label).reshape(1, *self.image_shape)
            input_img = torch.FloatTensor(input_img).reshape(1, *self.image_shape)

            # clamp 到 [-1, 1]
            label = torch.clamp(label, -1, 1)
            input_img = torch.clamp(input_img, -1, 1)

        except Exception as e:
            print(f"Error loading index {index}: {e}")
            return (
                torch.zeros((1, *self.image_shape)),
                torch.zeros((1, *self.image_shape))
            )

        return label, input_img