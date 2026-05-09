import copy
import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from .layers import *

# from config import DEVICE
DEVICE = 'cuda:0'


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    
class Tanh(nn.Module):
    def forward(self, x):
        return torch.tanh(x)


class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Tanh(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t).to(DEVICE)
        return emb


class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        x = self.main(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        _, _, H, W = x.shape
        x = F.interpolate(
            x, scale_factor=2, mode='nearest')
        x = self.main(x)
        return x

class DownSample_For_Mean_Flow(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, remb, temb):
        x = self.main(x)
        return x


class UpSample_For_Mean_Flow(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, remb, temb):
        _, _, H, W = x.shape
        x = F.interpolate(
            x, scale_factor=2, mode='nearest')
        x = self.main(x)
        return x


class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(4, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(4, in_ch),
            Tanh(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Tanh(),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(4, out_ch),
            Tanh(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(out_ch)
            self.attn = nn.Identity()
        else:
            self.attn = nn.Identity()
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb):
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None]
        h = self.block2(h)

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h


class MotionMapNet(nn.Module):

    def __init__(self, in_ch=1, base_ch=32):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_ch * 2, base_ch * 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_ch, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        feat = self.encoder(x)

        motion_map = self.decoder(feat)

        return motion_map

class ResBlock_For_Mean_Flow(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(4, in_ch),
            Tanh(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.remb_proj = nn.Sequential(
            Tanh(),
            nn.Linear(tdim, out_ch),
        )
        self.temb_proj = nn.Sequential(
            Tanh(),
            nn.Linear(tdim, out_ch),
        )

        self.block2 = nn.Sequential(
            nn.GroupNorm(4, out_ch),
            Tanh(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(out_ch)
            # self.attn = nn.Identity()
        else:
            self.attn = nn.Identity()
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, remb, temb):
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None]
        h += self.remb_proj(remb)[:, :, None, None]
        h = self.block2(h)

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h

class UNet(nn.Module):
    def __init__(self, 
                 ch, 
                 ch_mult, 
                 attn, 
                 num_res_blocks, 
                 dropout, 
                 condition=False, 
                 use_level=False,
                 use_distance=False):
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        tdim = ch * 4
        # self.time_embedding = TimeEmbedding(T, ch, tdim)
        self.dim = tdim
        self.condition = condition
        self.use_level = use_level
        self.use_distance = use_distance
        if condition:
            in_ch = 2
        else:
            in_ch = 1
        
        
        if use_level:
            self.lv_embed = nn.Sequential(
                nn.Linear(720, tdim),
                nn.ReLU(),
                nn.Linear(tdim, tdim)
            )
        
        if use_distance:
            self.dist_emd = nn.Sequential(
                nn.Linear(1, tdim),
                nn.ReLU(),
                nn.Linear(tdim, tdim)
            )

        self.head = nn.Conv2d(in_ch, ch, kernel_size=3, stride=1, padding=1)          
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(
                    in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(4, now_ch),
            Tanh(),
            nn.Conv2d(now_ch, 1, 3, stride=1, padding=1)
        )
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def forward(self, x, t, condition=None, angles=0, d=None):
        if self.condition:
            x = torch.cat([x, condition], dim=1)
        temb = get_timestep_embedding(t, self.dim)
        if self.use_level:
            lv_emb = self.lv_embed(angles)
            temb = temb + lv_emb
        if self.use_distance:
            d_emb = self.dist_emd(d[:,None])
            temb = temb + d_emb
        
        # Downsampling
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb)
            # print(h.shape)
            hs.append(h)
        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb)
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)
        h = self.tail(h)
        # assert len(hs) == 0
        used_sigmas = t.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
        h = x[:,:1] - h * used_sigmas
        return h


class UNet_Dual_Decoder(nn.Module):
    def __init__(self,
                 ch,
                 ch_mult,
                 attn,
                 num_res_blocks,
                 dropout,
                 condition=False):
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        tdim = ch * 4
        # self.time_embedding = TimeEmbedding(T, ch, tdim)
        self.dim = tdim
        self.condition = condition
        if condition:
            in_ch = 3
        else:
            in_ch = 1

        self.head = nn.Conv2d(in_ch, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs_for_decoder1 = [ch]  # record output channel when dowmsample for upsample
        chs_for_decoder2 = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(
                    in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
                chs_for_decoder1.append(now_ch)
                chs_for_decoder2.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs_for_decoder1.append(now_ch)
                chs_for_decoder2.append(now_ch)

        now_ch_copy = now_ch
        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
        ])

        self.upblocks1 = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks1.append(ResBlock(
                    in_ch=chs_for_decoder1.pop() + now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
            if i != 0:
                self.upblocks1.append(UpSample(now_ch))
        assert len(chs_for_decoder1) == 0

        now_ch = now_ch_copy
        self.upblocks2 = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks2.append(ResBlock(
                    in_ch=chs_for_decoder2.pop() + now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
            if i != 0:
                self.upblocks2.append(UpSample(now_ch))
        assert len(chs_for_decoder2) == 0

        self.tail_1 = nn.Sequential(
            nn.GroupNorm(4, now_ch),
            Tanh(),
            nn.Conv2d(now_ch, 1, 3, stride=1, padding=1)
        )

        self.tail_2 = nn.Sequential(
            nn.GroupNorm(4, now_ch),
            Tanh(),
            nn.Conv2d(now_ch, 1, 3, stride=1, padding=1)
        )
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail_1[-1].weight, gain=1e-5)
        init.zeros_(self.tail_1[-1].bias)
        init.xavier_uniform_(self.tail_2[-1].weight, gain=1e-5)
        init.zeros_(self.tail_2[-1].bias)

    def forward(self, x, t, condition=None, jitter_map=None):
        if self.condition:
            x = torch.cat([x, condition, jitter_map], dim=1)
        temb = get_timestep_embedding(t, self.dim)


        # Downsampling
        h = self.head(x)
        hs1 = [h]
        hs2 = [h]
        for layer in self.downblocks:
            h = layer(h, temb)
            # print(h.shape)
            hs1.append(h)
            hs2.append(h)
        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb)

        h1 = h
        h2 = h

        # Upsampling
        for i, layer in enumerate(self.upblocks1):
            if isinstance(layer, ResBlock):
                h1 = torch.cat([h1, hs1.pop()], dim=1)
            h1 = layer(h1, temb)

        h1 = self.tail_1(h1)

        for i, layer in enumerate(self.upblocks2):
            if isinstance(layer, ResBlock):
                h2 = torch.cat([h2, hs2.pop()], dim=1)
            h2 = layer(h2, temb)

        h2 = self.tail_2(h2)
        # used_sigmas = t.reshape((x.shape[0], *([1] * len(x.shape[1:]))))

        h_denoised = x[:, :1] + h1
        h_restored = x[:, 1:] + h2
        return h_denoised, h_restored




if __name__ == '__main__':
    batch_size = 1
    model = UNet(
        ch=16, ch_mult=[1, 2, 4], attn=[2],
        num_res_blocks=2, dropout=0.1, condition=False).to(DEVICE)
    x = torch.randn(batch_size, 1, 512, 512).to(DEVICE)
    t = torch.randint(1000, (batch_size,)).to(DEVICE)
    r = torch.randint(1000, (batch_size,)).to(DEVICE)
    y = model(x, r, t).to(DEVICE)
    print(y.shape)
