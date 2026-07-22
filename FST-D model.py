import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F


def warp_image(image2, flow):
    if image2.dim() != 4 or flow.dim() != 4:
        raise ValueError(f"warp_image expects 4D tensors, got image2={image2.shape}, flow={flow.shape}")

    B, C, H, W = image2.size()

    base_grid = F.affine_grid(
        torch.eye(2, 3, device=image2.device, dtype=image2.dtype).unsqueeze(0).expand(B, -1, -1),
        size=(B, C, H, W),
        align_corners=True
    )

    flow_x = flow[:, 0, :, :]
    flow_y = flow[:, 1, :, :]

    norm_flow_x = 2.0 * flow_x / (W - 1)
    norm_flow_y = 2.0 * flow_y / (H - 1)

    flow_grid = torch.stack([norm_flow_x, norm_flow_y], dim=-1)
    target_grid = base_grid + flow_grid

    warped_image = F.grid_sample(
        image2,
        target_grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )
    return warped_image


def flow_smoothness_l1(flow: torch.Tensor) -> torch.Tensor:
    if flow.dim() != 4 or flow.size(1) != 2:
        raise ValueError(f"flow_smoothness_l1 expects (B,2,H,W), got {tuple(flow.shape)}")

    du_dx = flow[:, 0, :, 1:] - flow[:, 0, :, :-1]
    du_dy = flow[:, 0, 1:, :] - flow[:, 0, :-1, :]
    dv_dx = flow[:, 1, :, 1:] - flow[:, 1, :, :-1]
    dv_dy = flow[:, 1, 1:, :] - flow[:, 1, :-1, :]

    return du_dx.abs().mean() + du_dy.abs().mean() + dv_dx.abs().mean() + dv_dy.abs().mean()


class CorrelationLayer(nn.Module):

    def __init__(self, max_displacement=12, stride_1=1, stride_2=1):
        super(CorrelationLayer, self).__init__()
        self.max_displacement = max_displacement
        self.stride_1 = stride_1
        self.stride_2 = stride_2
        self.D = 2 * self.max_displacement + 1
        self.D_squared = self.D * self.D

    def forward(self, input1, input2):
        assert input1.dim() == 4 and input2.dim() == 4
        B, C, H, W = input1.size()

        corr_volume = input1.new_zeros(B, self.D_squared, H, W)

        padded_input2 = F.pad(input2, (
            self.max_displacement, self.max_displacement,
            self.max_displacement, self.max_displacement
        ))

        for i in range(self.D):
            for j in range(self.D):
                displacement_h = i - self.max_displacement
                displacement_w = j - self.max_displacement

                shifted_input2 = padded_input2[:, :,
                                 self.max_displacement + displacement_h: self.max_displacement + displacement_h + H,
                                 self.max_displacement + displacement_w: self.max_displacement + displacement_w + W]

                correlation = (input1 * shifted_input2).sum(dim=1)
                corr_volume[:, i * self.D + j, :, :] = correlation

        return corr_volume


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

    H_padded, W_padded = H + pad_h, W + pad_w
    num_windows_h = H_padded // window_size
    num_windows_w = W_padded // window_size

    x = x.view(B, num_windows_h, window_size, num_windows_w, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (H, W, H_padded, W_padded)


def window_reverse(windows, window_size, H_padded, W_padded, H, W):
    B = int(windows.shape[0] / ((H_padded // window_size) * (W_padded // window_size)))

    x = windows.view(B, H_padded // window_size, W_padded // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H_padded, W_padded, -1)
    x = x[:, :H, :W, :].contiguous()
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        self.relative_position_index = relative_coords.sum(-1)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        if x.dim() != 3:
            raise ValueError(f"Expected 3D tensor as input, got {x.dim()}D tensor")

        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7,
                 shift_size=0, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (
            slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (
            slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows, _ = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows, (_, _, H_padded, W_padded) = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H_padded, W_padded, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)
        return x


class ProgressiveUpsampleHead(nn.Module):

    def __init__(self, input_resolution, dim, out_channels):
        super().__init__()
        self.input_resolution = input_resolution
        hidden_dim = max(dim // 2, out_channels * 4)
        mid_dim = max(hidden_dim // 2, out_channels * 2)

        self.up_stage1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GELU(),
        )
        self.up_stage2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GELU(),
        )
        self.refine_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden_dim, mid_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(mid_dim, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, x):
        h, w = self.input_resolution
        b, l, c = x.shape
        assert l == h * w, "input features have wrong size"

        x = x.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        x = self.up_stage1(x)
        x = self.up_stage2(x)
        x = self.refine_head(x)

        x = F.softsign(x)
        return x * 15.0


class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads,
                 window_size, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                 downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer)
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class BasicLayerUp(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads,
                 window_size, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                 upsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer)
            for i in range(depth)])

        if upsample is not None:
            self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=128, patch_size=4, in_chans=1, embed_dim=48, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
        pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
        x = F.pad(x, (0, pad_w, 0, pad_h))
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class SwinTransformerSys(nn.Module):
    def __init__(self, img_size=128, patch_size=4, in_chans=2, num_classes=2,
                 embed_dim=96, depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24], window_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first",
                 extra_in_chans=0, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample
        self.extra_in_chans = int(extra_in_chans)

        self.half_embed_dim = embed_dim // 2

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=1,
            embed_dim=self.half_embed_dim, norm_layer=norm_layer if self.patch_norm else None)

        if self.extra_in_chans > 0:
            self.extra_patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=self.extra_in_chans,
                embed_dim=self.half_embed_dim, norm_layer=norm_layer if self.patch_norm else None)
        else:
            self.extra_patch_embed = None

        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.correlation_layer = CorrelationLayer(max_displacement=12)
        corr_out_channels = self.correlation_layer.D_squared

        fusion_in_channels = embed_dim + corr_out_channels
        if self.extra_patch_embed is not None:
            fusion_in_channels += self.half_embed_dim

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fusion_in_channels, embed_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GELU(),
        )

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer), patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer], num_heads=num_heads[i_layer], window_size=window_size,
                mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(
                2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                int(embed_dim * 2 ** (self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()
            if i_layer == 0:
                layer_up = PatchExpand(
                    input_resolution=(
                        patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                        patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up = BasicLayerUp(
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    input_resolution=(
                        patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                        patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                    depth=depths[(self.num_layers - 1 - i_layer)], num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                    window_size=window_size, mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                        depths[:(self.num_layers - 1 - i_layer) + 1])],
                    norm_layer=norm_layer, upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                    use_checkpoint=use_checkpoint)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)

        if self.final_upsample == "expand_first":
            self.up = ProgressiveUpsampleHead(
                input_resolution=(img_size // patch_size, img_size // patch_size),
                dim=embed_dim, out_channels=self.num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if (isinstance(m, nn.Linear) and m.bias is not None):
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x_downsample = []

        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)

        x = self.norm(x)
        return x, x_downsample

    def forward_up_features(self, x, x_downsample):
        for idx, layer_up in enumerate(self.layers_up):
            if idx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x, x_downsample[3 - idx]], -1)
                x = self.concat_back_dim[idx](x)
                x = layer_up(x)

        x = self.norm_up(x)
        return x

    def upX4(self, x):
        if self.final_upsample == "expand_first":
            x = self.up(x)
        return x

    def forward(self, img1, img2, extra_features=None):
        f1_embed = self.patch_embed(img1)
        f2_embed = self.patch_embed(img2)

        B, L, C_half = f1_embed.shape
        H, W = self.patches_resolution

        f1_4d = f1_embed.transpose(1, 2).view(B, C_half, H, W)
        f2_4d = f2_embed.transpose(1, 2).view(B, C_half, H, W)

        corr_volume = self.correlation_layer(f1_4d, f2_4d)

        fused_parts = [f1_4d, f2_4d]
        if self.extra_patch_embed is not None:
            if extra_features is None:
                raise ValueError("extra_features is required when extra_in_chans > 0")
            if extra_features.shape[1] != self.extra_in_chans:
                raise ValueError(
                    f"Expected extra_features with {self.extra_in_chans} channels, got {extra_features.shape[1]}")
            extra_embed = self.extra_patch_embed(extra_features)
            extra_embed_4d = extra_embed.transpose(1, 2).view(B, self.half_embed_dim, H, W)
            fused_parts.append(extra_embed_4d)
        fused_parts.append(corr_volume)

        fused_features_2d = torch.cat(fused_parts, dim=1)
        x = self.fusion_conv(fused_features_2d)
        x = x.flatten(2).transpose(1, 2)

        x, x_downsample = self.forward_features(x)
        x = self.forward_up_features(x, x_downsample)
        x = self.upX4(x)
        return x


class StackedSwinSys(nn.Module):
    def __init__(self, BaseSwinSys, img_size=128, patch_size=4, embed_dim=96,
                 depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], num_classes=2, **kwargs):
        super().__init__()

        self.net1 = BaseSwinSys(
            img_size=img_size, patch_size=patch_size, in_chans=2,
            embed_dim=embed_dim, depths=depths, num_heads=num_heads,
            num_classes=num_classes, **kwargs
        )

        self.net2 = BaseSwinSys(
            img_size=img_size, patch_size=patch_size, in_chans=2,
            embed_dim=embed_dim, depths=depths, num_heads=num_heads,
            num_classes=num_classes, extra_in_chans=3, **kwargs
        )

    def forward(self, img1, img2, detach_w1_for_refine=False, refine_detach_alpha=None):
        assert img1.shape[1] == 1 and img2.shape[1] == 1

        w1 = self.net1(img1, img2)

        if refine_detach_alpha is not None:
            alpha = float(refine_detach_alpha)
            alpha = min(max(alpha, 0.0), 1.0)
            w1_for_refine = alpha * w1.detach() + (1.0 - alpha) * w1
        else:
            w1_for_refine = w1.detach() if detach_w1_for_refine else w1

        warped_img2 = warp_image(img2, w1_for_refine)
        residual_img = img1 - warped_img2
        refine_context = torch.cat([w1_for_refine, residual_img], dim=1)

        dw = self.net2(img1, warped_img2, extra_features=refine_context)

        w_final = w1 + dw
        return w_final, w1, dw