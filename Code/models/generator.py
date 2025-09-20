import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Utils / Norms / Regularizers
# -----------------------------

class DropPath(nn.Module):
    """Stochastic Depth per sample (from timm, simplified)."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x / keep_prob * random_tensor

class RMSNorm2d(nn.Module):
    """RMSNorm across channel dim for 2D feature maps."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        # rms over channels
        rms = torch.sqrt(x.pow(2).mean(dim=1, keepdim=True) + self.eps)
        x = x / rms
        return x * self.weight

# -----------------------------
# Basic Blocks
# -----------------------------

class ConvBlock(nn.Module):
    """Conv -> SiLU -> (optional) GN; kept light for speed."""
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, norm=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p)
        self.act = nn.SiLU(inplace=True)
        self.norm = nn.GroupNorm(1, out_ch) if norm else nn.Identity()

    def forward(self, x):
        return self.norm(self.act(self.conv(x)))

class SEGate(nn.Module):
    """Squeeze-and-Excitation gating for skip connections."""
    def __init__(self, ch, r=8):
        super().__init__()
        hidden = max(8, ch // r)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(ch, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, ch, 1),
            nn.Sigmoid()
        )

    def forward(self, skip):
        w = self.fc(self.pool(skip))
        return skip * w

class PixelShuffleUp(nn.Module):
    def __init__(self, in_ch, out_ch, scale=2):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * scale * scale, 3, 1, 1),
            nn.PixelShuffle(scale),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        )

    def forward(self, x):
        return self.up(x)

# -----------------------------
# Windowed Self-Attention (Swin-style)
# -----------------------------

def window_partition(x, window_size):
    """B,C,H,W -> (num_windows*B), C, w, w"""
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()  # B, H/ws, W/ws, C, ws, ws
    x = x.view(-1, C, window_size, window_size)
    return x

def window_unpartition(windows, window_size, H, W, B):
    """(num_windows*B), C, w, w -> B,C,H,W"""
    x = windows.view(B, H // window_size, W // window_size, -1, window_size, window_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()  # B,C,H/ws,ws,W/ws,ws
    x = x.view(B, -1, H, W)
    return x

class WindowSelfAttention(nn.Module):
    """MHA over flattened windows. Uses batch_first=True on tokens."""
    def __init__(self, dim, num_heads=4, window_size=8):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        ws = self.window_size
        assert H % ws == 0 and W % ws == 0, "H and W must be divisible by window_size"

        # Partition into windows
        xw = window_partition(x, ws)            # (nW*B, C, ws, ws)
        nWB = xw.shape[0]
        xw = xw.view(nWB, C, ws * ws).transpose(1, 2)  # (nW*B, ws*ws, C)

        # Self-attention per window
        xw, _ = self.mha(xw, xw, xw)            # (nW*B, ws*ws, C)

        # Back to (nW*B, C, ws, ws)
        xw = xw.transpose(1, 2).view(nWB, C, ws, ws)

        # Unpartition
        out = window_unpartition(xw, ws, H, W, B)
        return out

class SwinTransformerBlock2D(nn.Module):
    """Pre-norm -> W-MSA (optionally shifted) -> residual (res_scale, droppath)
       -> Pre-norm -> MLP -> residual"""
    def __init__(self, dim, num_heads=4, window_size=8, drop_path=0.1, mlp_ratio=2.0, res_scale=0.1, shift=False):
        super().__init__()
        self.shift = shift
        self.window_size = window_size
        self.norm1 = RMSNorm2d(dim)
        self.attn = WindowSelfAttention(dim, num_heads=num_heads, window_size=window_size)
        self.drop1 = DropPath(drop_path)
        self.res_scale = res_scale

        hidden = int(dim * mlp_ratio)
        self.norm2 = RMSNorm2d(dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, dim, 1)
        )
        self.drop2 = DropPath(drop_path)

    def forward(self, x):
        # (Optional) shifted windows via tensor roll
        if self.shift:
            shift = self.window_size // 2
            x = torch.roll(x, shifts=(-shift, -shift), dims=(2, 3))

        # Attn block
        y = self.attn(self.norm1(x))
        x = x + self.drop1(self.res_scale * y)

        # Undo roll before MLP so spatial alignment remains
        if self.shift:
            shift = self.window_size // 2
            x = torch.roll(x, shifts=(shift, shift), dims=(2, 3))

        # MLP block
        y = self.mlp(self.norm2(x))
        x = x + self.drop2(self.res_scale * y)
        return x

class RSTB(nn.Module):
    """Residual Swin Transformer Block stack with a conv tail."""
    def __init__(self, dim, depth=4, num_heads=4, window_size=8, drop_path=0.1, mlp_ratio=2.0, res_scale=0.1):
        super().__init__()
        blocks = []
        for i in range(depth):
            blocks.append(
                SwinTransformerBlock2D(
                    dim=dim, num_heads=num_heads, window_size=window_size,
                    drop_path=drop_path, mlp_ratio=mlp_ratio, res_scale=res_scale,
                    shift=bool(i % 2)  # alternate shift
                )
            )
        self.blocks = nn.Sequential(*blocks)
        self.conv_tail = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x):
        identity = x
        x = self.blocks(x)
        x = self.conv_tail(x)
        return x + identity

# -----------------------------
# Cross-Attention Fusion (VV↔VH) — memory safe
# -----------------------------

class LocalCrossAttentionFuse(nn.Module):
    """
    Memory-safe cross-attn:
      1) AvgPool to low-res grid
      2) Windowed cross-attn on low-res (batch-safe)
      3) Bilinear upsample back to full-res
    """
    def __init__(self, dim, num_heads=4, window_size=8, pool_stride=4):
        super().__init__()
        self.ws = window_size
        self.s  = pool_stride

        self.q_vv = nn.Conv2d(dim, dim, 1)
        self.k_vh = nn.Conv2d(dim, dim, 1)
        self.v_vh = nn.Conv2d(dim, dim, 1)

        self.q_vh = nn.Conv2d(dim, dim, 1)
        self.k_vv = nn.Conv2d(dim, dim, 1)
        self.v_vv = nn.Conv2d(dim, dim, 1)

        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.proj = nn.Conv2d(dim * 2, dim, 1)
        self.pool = nn.AvgPool2d(kernel_size=pool_stride, stride=pool_stride, ceil_mode=True)

    def _window_part(self, x):
        # x: [B,C,h,w]
        B, C, h, w = x.shape
        ws = self.ws
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
            h, w = x.shape[-2:]
        # [B,C,h,w] -> [B, nH, nW, C, ws, ws] -> [B*nH*nW, ws*ws, C]
        nH, nW = h // ws, w // ws
        x = x.view(B, C, nH, ws, nW, ws).permute(0, 2, 4, 1, 3, 5).contiguous()  # B,nH,nW,C,ws,ws
        tokens = x.view(B * nH * nW, ws * ws, C)
        meta = (B, C, h, w, pad_h, pad_w, nH, nW)
        return tokens, meta

    def _window_unpart(self, tokens, meta):
        # tokens: [B*nH*nW, ws*ws, C]
        B, C, h, w, pad_h, pad_w, nH, nW = meta
        ws = self.ws
        x = tokens.view(B, nH, nW, ws, ws, C)           # B, nH, nW, ws, ws, C
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()    # B, C, nH, ws, nW, ws
        x = x.view(B, C, h, w)
        if pad_h or pad_w:
            x = x[:, :, :h - pad_h, :w - pad_w]
        return x

    def _attend(self, q_map, k_map, v_map):
        # 1) Downsample
        ql = self.pool(q_map)   # [B,C,h,w]
        kl = self.pool(k_map)
        vl = self.pool(v_map)

        # 2) Windowed cross-attention at low-res (batch-safe)
        q_tok, meta = self._window_part(ql)            # [B*nW, ws*ws, C]
        k_tok, _    = self._window_part(kl)
        v_tok, _    = self._window_part(vl)
        out_tok, _  = self.attn(q_tok, k_tok, v_tok)   # [B*nW, ws*ws, C]

        low = self._window_unpart(out_tok, meta)       # [B,C,h,w]

        # 3) Upsample to full-res of q_map
        full = F.interpolate(low, size=q_map.shape[-2:], mode="bilinear", align_corners=False)
        return full

    def forward(self, vv, vh):
        # vv attends to vh, and vh attends to vv; then fuse
        out1 = self._attend(self.q_vv(vv), self.k_vh(vh), self.v_vh(vh))
        out2 = self._attend(self.q_vh(vh), self.k_vv(vv), self.v_vv(vv))
        fused = torch.cat([out1, out2], dim=1)
        return self.proj(fused)

# -----------------------------
# ConvNeXt-style Refinement Block
# -----------------------------

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, layer_scale_init=0.75):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim)
        self.norm = RMSNorm2d(dim)
        self.pw1 = nn.Conv2d(dim, dim * 4, 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(dim * 4, dim, 1)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(1, dim, 1, 1))

    def forward(self, x):
        residual = x
        x = self.dw(x)
        x = self.norm(x)
        x = self.pw2(self.act(self.pw1(x)))
        return residual + self.gamma * x

# -----------------------------
# Upgraded Generator
# -----------------------------

class Generator(nn.Module):
    """
    Input:  SAR (VV+VH) tensor [B, 2, H, W]
    Output: Optical RGB in [0,1], [B, 3, H, W]
    """
    def __init__(self,
                 dim=64,
                 depths=(2, 4, 4),        # RSTB depth per stage
                 heads=(2, 4, 4),
                 window_size=8,
                 drop_path=0.1,
                 res_scale=0.1,
                 mlp_ratio=2.0):
        super().__init__()

        # Dual stems for VV and VH (1ch each)
        self.stem_vv = nn.Sequential(
            ConvBlock(1, dim),
            ConvBlock(dim, dim)
        )
        self.stem_vh = nn.Sequential(
            ConvBlock(1, dim),
            ConvBlock(dim, dim)
        )

        # Memory-safe dual-branch fusion
        self.fuse = LocalCrossAttentionFuse(
            dim, num_heads=heads[0],
            window_size=window_size,   # try 8 or 4 if you still see pressure
            pool_stride=4              # try 2–4; larger = cheaper
        )

        # Encoder stages (downsample with stride-2 conv)
        self.down1 = nn.Conv2d(dim, dim * 2, 3, 2, 1)  # -> 1/2
        self.enc1 = RSTB(dim * 2, depth=depths[0], num_heads=heads[0],
                         window_size=window_size, drop_path=drop_path,
                         mlp_ratio=mlp_ratio, res_scale=res_scale)

        self.down2 = nn.Conv2d(dim * 2, dim * 4, 3, 2, 1)  # -> 1/4
        self.enc2 = RSTB(dim * 4, depth=depths[1], num_heads=heads[1],
                         window_size=window_size, drop_path=drop_path,
                         mlp_ratio=mlp_ratio, res_scale=res_scale)

        # Bottleneck
        self.bottleneck = RSTB(dim * 4, depth=depths[2], num_heads=heads[2],
                               window_size=window_size, drop_path=drop_path,
                               mlp_ratio=mlp_ratio, res_scale=res_scale)

        # Attention-gated skips
        self.skip1_gate = SEGate(dim * 2)
        self.skip0_gate = SEGate(dim)

        # Decoder with PixelShuffle and gated skip fusion
        self.up1 = PixelShuffleUp(dim * 4, dim * 2, scale=2)     # 1/4 -> 1/2
        self.dec1 = nn.Sequential(
            ConvBlock(dim * 4, dim * 2),  # concat with gated skip1
            ConvBlock(dim * 2, dim * 2)
        )

        self.up0 = PixelShuffleUp(dim * 2, dim, scale=2)         # 1/2 -> 1
        self.dec0 = nn.Sequential(
            ConvBlock(dim * 2, dim),      # concat with gated skip0
            ConvBlock(dim, dim)
        )

        # Refinement head (ConvNeXt blocks)
        self.refine = nn.Sequential(
            ConvNeXtBlock(dim),
            ConvNeXtBlock(dim),
            ConvNeXtBlock(dim)
        )

        # Output
        self.out_conv = nn.Conv2d(dim, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # x: [B, 2, H, W] -> split VV/VH
        vv = x[:, 0:1, :, :]
        vh = x[:, 1:2, :, :]

        vv = self.stem_vv(vv)  # [B, dim, H, W]
        vh = self.stem_vh(vh)  # [B, dim, H, W]

        x0 = self.fuse(vv, vh)  # fused at full res, [B, dim, H, W]

        # Encoder
        s0 = x0                                 # skip at 1×
        x1 = self.enc1(self.down1(x0))          # [B, 2dim, H/2, W/2]
        s1 = x1                                 # skip at 1/2×
        x2 = self.enc2(self.down2(x1))          # [B, 4dim, H/4, W/4]

        # Bottleneck
        x2 = self.bottleneck(x2)

        # Decoder with gated skips
        u1 = self.up1(x2)                       # [B, 2dim, H/2, W/2]
        s1g = self.skip1_gate(s1)
        d1 = torch.cat([u1, s1g], dim=1)        # [B, 4dim, H/2, W/2]
        d1 = self.dec1(d1)                      # [B, 2dim, H/2, W/2]

        u0 = self.up0(d1)                       # [B, dim, H, W]
        s0g = self.skip0_gate(s0)
        d0 = torch.cat([u0, s0g], dim=1)        # [B, 2dim, H, W]
        d0 = self.dec0(d0)                      # [B, dim, H, W]

        # Refinement + Output
        y = self.refine(d0) + d0                # global residual refinement
        y = torch.sigmoid(self.out_conv(y))     # [B,3,H,W] in [0,1]
        return y
