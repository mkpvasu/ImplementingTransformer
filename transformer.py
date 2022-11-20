# Importing necessary libraries
import os
import torch
import torch.nn as nn
import torch.functional as F
import matplotlib.pyplot as plt
import PIL.Image as PImg
import torchvision.transforms as transform
from einops import reduce, rearrange, repeat
from einops.layers.torch import Reduce, Rearrange
from torchsummary import summary

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

blackCat = PImg.open('./blackcat.jpg')
blackCatFig = plt.imshow(blackCat)
# plt.show()

# Resize img to 128 * 128 img
imgSize = 128
resize = transform.Compose([transform.Resize((imgSize, imgSize)), transform.ToTensor()])
x = resize(blackCat)
x = x.unsqueeze(0)

# Size for each patch = 16
patchSize = 16
imgPatches = rearrange(x, 'b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patchSize, s2=patchSize)
# print(imgPatches.size())


class PatchEmbedding(nn.Module):
    def __init__(self, in_c: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = imgSize):
        super().__init__()
        self.patch_size = patch_size
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos = nn.Parameter(torch.randn((img_size//patch_size)**2 + 1, emb_size))
        self.projection = nn.Sequential(
            nn.Conv2d(in_c, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e')
        )

    def forward(self, patchedImg: torch.Tensor) -> torch.Tensor:
        b, _, _, _ = patchedImg.shape
        patchedImg = self.projection(patchedImg)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        patchedImg = torch.cat([cls_tokens, patchedImg], dim=1)
        patchedImg += self.pos
        return patchedImg


print(PatchEmbedding()(x).shape)
