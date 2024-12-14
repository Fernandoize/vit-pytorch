import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    """
    FFN一般由两个Linear + 1个RELU或GEL U
    """
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        # 是否需要将多个头进行合并
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        # Layer Normalization 通常用于每个子层的连接（如自注意力子层和前馈神经网络子层），以帮助模型收敛并稳定训练。
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        #  dropout 层，用于防止过拟合。
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        # 负责将每个头的输出结合成最终的输出
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # 对输入进行层归一化，确保训练稳定
        x = self.norm(x)
        # 输出分为三部分，分别对应 Q、K 和 V
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # 使用 rearrange 将每个 Q、K、V 的形状调整为 (batch_size, heads, sequence_length, dimension_per_head)。
        # 其中 b 代表 batch size，n 代表序列长度，h 代表注意力头的数量，d 代表每个头的维度。
        # 假设q,k,v的shape为(1, 32, 512), 则变换为(1, 8, 32, 64), 实际上先将512拆分为head * dim_head, 再交换head和sequence长度的位置，便于后续计算
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        # 计算注意力分数，即将查询 Q 与键 K 的转置相乘。乘以缩放因子 self.scale 来防止点积过大。
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # 对注意力分数应用 softmax 函数，得到注意力权重。
        attn = self.attend(dots)
        attn = self.dropout(attn)
        # 用得到的注意力权重 attn 与值 V 相乘，得到加权的输出特征。
        out = torch.matmul(attn, v)
        # 将输出的形状调整回(batch_size, sequence_length, hidden_dim)。
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            # 残差连接使得模型在学习过程中可以更容易地保留一部分原始信息，即使在深层的网络中，有助于训练更深的神经网络。
            x = attn(x) + x
            # FFN最开始有一次LayerNorm
            x = ff(x) + x
        # 最后再来一次layer_norm
        return self.norm(x)

class ViT(nn.Module):

    #  patch_size 代表每个小块的大小，num_classes 分类数, dim 特征的维度，depth: encoder的层数，heads 注意力头的数量
    # mlp_dim mlp的维度，dim_head, 注意力层的维度
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        # 计算图像总的小块数量 num_patches 和每个小块的维度 patch_dim。
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        # 池化方式
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # h 为图像高度，w为宽度，p1为patch高度，p2为patch宽度，c为channel数
        # 将patch从(3, 16, 16)转化为16 * 16 * 3, 也就是将patch打平
        #  将每个小块的特征从patch_dim 转换为 dim，并进行归一化
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        # self.pos_embedding: 为小块和 CLS token 创建一个可学习的位置嵌入，用于在 Transformer 中引入位置信息。
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # self.dropout: 将 dropout 应用到位置嵌入和小块嵌入的输出上，防止过拟合。
        self.dropout = nn.Dropout(emb_dropout)

        # transformer模块
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # 池化 + 线性层
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        # x = self.to_patch_embedding(img): 将输入图像分块并嵌入到相应的维度。
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # 复制 CLS token，以适应批量大小。
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        # 将 CLS token 添加到嵌入序列的开头。
        x = torch.cat((cls_tokens, x), dim=1)
        # 将位置嵌入添加到小块嵌入中
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)
        # 根据池化方式选择输出（CLS token 或者小块特征的平均值）。
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        # 将输出送入分类头，完成最终的分类
        return self.mlp_head(x)
