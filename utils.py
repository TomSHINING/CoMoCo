import os
import torch
import numpy as np
import torch.nn.functional as F


def load_model(net, checkpoint):
    net.load_state_dict(checkpoint['state_dict'])
    # if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    #     net = torch.nn.DataParallel(net).cuda()
    # elif torch.cuda.is_available() and torch.cuda.device_count() == 1:
    #     net = net.cuda()
    return net


def save_model(net, optimizer, epoch, save_dir, scheduler=None):
    '''save model'''

    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)

    if 'module' in dir(net):
        state_dict = net.module.state_dict()
    else:
        state_dict = net.state_dict()

    if scheduler is None:
        torch.save({
            'state_dict': state_dict,
            'optimizer_state_dict': optimizer.state_dict()},
            os.path.join(save_dir, 'model_at_epoch_%03d.dat' % (epoch)))

    else:
        torch.save({
            'state_dict': state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler': scheduler.state_dict()},
            os.path.join(save_dir, 'model_at_epoch_%03d.dat' % (epoch)))

    print(os.path.join(save_dir, 'model_at_epoch_%03d.dat' % (epoch)))


def find_lastest_file(file_dir):
    lists = os.listdir(file_dir)
    lists.sort(key=lambda x: os.path.getmtime((file_dir + x)))
    file_latest = os.path.join(file_dir, lists[-1])
    print(f'Find latest file: {file_latest}')
    return file_latest

def generate_sample(K, model, device):
    x_0 = torch.randn((1, 1, 512, 512), dtype=torch.float32, device=device)
    for k in range(K):
        t = torch.tensor(k / K, device=device).repeat(x_0.shape[0])
        with torch.no_grad():
            x_0 = x_0 + 1/K * model(x_0, t, torch.tensor([1/K]).to(device))
    return x_0

def generate_consistency_sample(K, model, device):
    x_0 = torch.randn((1, 1, 512, 512), dtype=torch.float32, device=device)
    for k in range(K):
        t = torch.tensor(k / K, device=device).repeat(x_0.shape[0]).view(-1,1,1,1)
        t_1 = torch.tensor((k+1) / K, device=device).repeat(x_0.shape[0]).view(-1,1,1,1)
        noise = torch.randn_like(x_0, dtype=torch.float32, device=device)
        s = 1-t
        with torch.no_grad():
            if k + 1 < K:
                x_0 = x_0 + (1-t)[0] * model(x_0, t[:,0,0,0], s[:,0,0,0])
                x_0 = t_1 * x_0 + (1-t_1) * noise
            else:
                x_0 = x_0 + (1-t)[0] * model(x_0, t[:,0,0,0], s[:,0,0,0])
    return x_0

def normalization(tensor):
    return (tensor - torch.min(tensor)) / (torch.max(tensor) - torch.min(tensor))





def x_derivative(input):
    # inputs: batch, channel, width, height
    _, C, _, _ = input.shape
    fiter_first = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=np.float32)
    fiter_first = fiter_first[np.newaxis, np.newaxis, ...]
    fiter_first = fiter_first.repeat(C, axis=0)
    fiter_kernel = torch.FloatTensor(fiter_first).cuda(input.get_device())
    derivative_first = F.conv2d(input, fiter_kernel, padding=(3 - 1) // 2, groups=C)

    fiter_second = np.array([[1, 2, 1],
                             [0, 0, 0],
                             [-1, -2, -1]], dtype=np.float32)
    fiter_second = fiter_second[np.newaxis, np.newaxis, ...]
    fiter_second = fiter_second.repeat(C, axis=0)
    fiter_kernel = torch.FloatTensor(fiter_second).cuda(input.get_device())
    derivative_second = F.conv2d(derivative_first, fiter_kernel, padding=(3 - 1) // 2, groups=C)

    return torch.cat([input, derivative_first, derivative_second], 1)


def cartesian2polar(image, num_angle=360):
    N = image.shape[0]
    M = image.shape[-1]
    theta = torch.linspace(0, np.deg2rad(179), num_angle)
    r = torch.linspace(0, M - 1, M) + 1 - (M + 1) / 2
    theta, r = torch.meshgrid(theta, r)
    z = torch.polar(r, theta)
    X, Y = z.real.to(image.device), z.imag.to(image.device)
    grid = torch.stack((X / torch.max(X), Y / torch.max(Y)), 2).unsqueeze(0)  # (1, nv, nu, 2)  <= (N, H, W, 2)
    grid = grid.repeat(N, 1, 1, 1)
    return F.grid_sample(image, grid, align_corners=False)


def polar2cartesian(image):
    N = image.shape[0]
    image_left, image_right = image.clone(), image.clone()
    image_left[..., :image.shape[-1] // 2] = image[..., :image.shape[-1] // 2]
    image_right[..., image.shape[-1] // 2:] = image[..., image.shape[-1] // 2:]
    image_left2right = torch.flip(image_left, dims=[-1])
    image_cat = torch.cat((image_right, image_left2right), dim=-2)

    M = image.shape[-1]
    X, Y = torch.arange(M).to(image.device), torch.arange(M).to(image.device)
    # X, Y = torch.meshgrid(X, Y, indexing='ij')
    X, Y = torch.meshgrid(X, Y)
    X, Y = X - (M - 1) / 2, Y - (M - 1) / 2
    r = torch.sqrt(X ** 2 + Y ** 2)
    theta = torch.atan2(Y, X)

    grid = torch.stack((r / (M // 2), theta / np.pi), 2).unsqueeze(0)  # (1, nv, nu, 2)  <= (N, H, W, 2)
    grid = grid.repeat(N, 1, 1, 1)
    img = F.grid_sample(image_cat, grid, align_corners=False)

    return torch.flip((torch.rot90(img, dims=[-2, -1], k=1)), dims=[-1])


def extract_patches_online(tensor, num=2):
    if tensor.ndim == 5:

        split_w = torch.chunk(tensor, chunks=num, dim=3)
        stack_w = torch.reshape(torch.stack(split_w, dim=0),
                                [num * tensor.shape[0], tensor.shape[1],
                                 tensor.shape[2], tensor.shape[3] // num, tensor.shape[4]])
        split_h = torch.chunk(stack_w, chunks=num, dim=4)
        stack_h = torch.reshape(torch.stack(split_h, dim=0),
                                [num * num * tensor.shape[0], tensor.shape[1],
                                 tensor.shape[2], tensor.shape[3] // num, tensor.shape[4] // num])

        return stack_h

    elif tensor.ndim == 4:

        split_w = torch.chunk(tensor, chunks=num, dim=2)
        # print('split_w', split_w.size())
        stack_w = torch.reshape(torch.stack(split_w, dim=0),
                                [num * tensor.shape[0], tensor.shape[1],
                                 tensor.shape[2] // num, tensor.shape[3]])
        # print('stack_w', stack_w.size())
        split_h = torch.chunk(stack_w, chunks=num, dim=3)
        # print('split_h', split_h.size())
        stack_h = torch.reshape(torch.stack(split_h, dim=0),
                                [num * num * tensor.shape[0], tensor.shape[1],
                                 tensor.shape[2] // num, tensor.shape[3] // num])
        # print('stack_h', stack_h.size())

        return stack_h

    else:
        print('Expect for the tensor with dim==5 or 4, other cases are not yet implemented.')

def compute_edge_map(image): # B 1 H W
    kernel = torch.tensor([[-1, -1, -1],
                           [-1, 8, -1],
                            [-1, -1, -1]], dtype=torch.float32)
    kernel = kernel.view(1, 1, 3, 3).to(image.device)
    edge_map = F.conv2d(image, kernel, padding=1)
    return edge_map

def compute_edge_map_with_random_mask(image): # B 1 H W
    kernel = torch.tensor([[-1, -1, -1],
                           [-1, 8, -1],
                            [-1, -1, -1]], dtype=torch.float32)
    
    kernel = kernel.view(1, 1, 3, 3).to(image.device)
    edge_map_tmp = F.conv2d(image, kernel, padding=1)
    edge_map = torch.zeros_like(edge_map_tmp)
    edge_map[edge_map_tmp >= 0.8] = 1.0
    # random mask edge map in a continuous area
    mask = torch.ones_like(edge_map)
    for i in range(edge_map.shape[0]):
        num_mask = np.random.randint(1, 5)
        for _ in range(num_mask):
            x_start = np.random.randint(0, edge_map.shape[2] - 31)
            y_start = np.random.randint(0, edge_map.shape[3] - 31)
            w = np.random.randint(10, 30)
            h = np.random.randint(10, 30)
            mask[i, :, x_start:x_start + w, y_start:y_start + h] = 0.0
    edge_map = edge_map * mask
    return edge_map
    

### mix two images
class MixUp_AUG:
    def __init__(self):
        self.dist = torch.distributions.beta.Beta(torch.tensor([1.2]), torch.tensor([1.2]))

    def aug(self, rgb_gt, rgb_noisy):
        bs = rgb_gt.size(0)
        indices = torch.randperm(bs)
        rgb_gt2 = rgb_gt[indices]
        rgb_noisy2 = rgb_noisy[indices]

        if rgb_gt.ndim == 4:

            lam = self.dist.rsample((bs, 1)).view(-1, 1, 1, 1).cuda(rgb_gt.get_device())

        elif rgb_gt.ndim == 5:

            lam = self.dist.rsample((bs, 1)).view(-1, 1, 1, 1, 1).cuda(rgb_gt.get_device())

        else:

            print('Dim is not implemented for MixUp_AUG!')

        rgb_gt = lam * rgb_gt + (1 - lam) * rgb_gt2
        rgb_noisy = lam * rgb_noisy + (1 - lam) * rgb_noisy2

        return rgb_gt, rgb_noisy


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    input = torch.randn(4, 18, 256, 256).cuda()
    print(x_derivative(input).shape)
