import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Reduce
from timm.models import register_model
from torch import einsum, nn

from .resnet import BasicBlock


class ConvAttn(nn.Module):
    def __init__(self, block, in_planes, out_planes):
        super(ConvAttn, self).__init__()
        self.block = block
        self.in_planes = in_planes
        self.projection = nn.Sequential(
            nn.Linear(in_planes, out_planes), nn.ReLU()
        )  # nn.Linear(in_planes, out_planes)

        self.scale = out_planes ** -0.5

    def forward(self, x, cls):
        out = self.block(x)
        k = rearrange(out, "... c h w -> ... c (h w)")
        q = self.projection(cls)  # b1c
        attn = einsum("bc,bcn->bn", q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        cls = einsum("bn,bnc->bc", attn, k.transpose(-1, -2))
        return out, cls


class Repeat(nn.Module):
    def __init__(self, pattern, planes):
        super(Repeat, self).__init__()
        self.pattern = pattern
        self.cls = nn.Parameter(torch.randn(1, planes))

    def forward(self, x):
        b, *_ = x.shape
        cls = repeat(self.cls, self.pattern, b=b)
        return cls


class SepLinear(nn.Module):
    def __init__(self, in_planes, out_planes, norm):
        super(SepLinear, self).__init__()
        self.linear1 = nn.Linear(in_planes, out_planes)
        self.linear2 = nn.Linear(in_planes, out_planes)
        self.norm = norm

    def forward(self, x1, x2):
        if self.norm:
            x1 = F.normalize(x1, dim=-1)
            x2 = F.normalize(x2, dim=-1)
        out1 = self.linear1(x1)
        out2 = self.linear2(x2)
        return out1 + out2


class CatLinear(nn.Module):
    def __init__(self, in_planes, out_planes, norm):
        super(CatLinear, self).__init__()
        self.linear = nn.Linear(in_planes * 2, out_planes)
        self.norm = norm

    def forward(self, x1, x2):
        if self.norm:
            x1 = F.normalize(x1, dim=-1)
            x2 = F.normalize(x2, dim=-1)
        x = torch.cat([x1, x2], dim=-1)
        return self.linear(x)


class UseOneLinear(nn.Module):
    def __init__(self, in_planes, out_planes, norm, use_idx=0):
        super(UseOneLinear, self).__init__()
        self.linear = nn.Linear(in_planes, out_planes)
        self.norm = norm
        self.use_idx = use_idx

    def forward(self, x1, x2):
        x = x1 if self.use_idx == 0 else x2
        if self.norm:
            x = F.normalize(x, dim=-1)
        return self.linear(x)


class AttnRes(nn.Module):
    def __init__(
        self,
        block,
        num_blocks,
        num_classes=10,
        token_type="avg",
        classifier_type="sep",
        norm=False,
    ):
        super(AttnRes, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = self._make_token(token_type)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = self._make_classifier(
            512 * block.expansion, num_classes, classifier_type, norm
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        initial_planes = self.in_planes
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return ConvAttn(nn.Sequential(*layers), initial_planes, self.in_planes)

    def _make_classifier(self, in_planes, num_classes, classifier_type, norm):
        if classifier_type == "sep":
            return SepLinear(in_planes, num_classes, norm)
        elif classifier_type == "cat":
            return CatLinear(in_planes, num_classes, norm)
        elif classifier_type == "cls":
            return UseOneLinear(in_planes, num_classes, norm, use_idx=1)

    def _make_token(self, token_type):
        if token_type == "avg":
            return Reduce("b c h w -> b c", "mean")
        elif token_type == "learnable":
            self.cls = nn.Parameter(torch.randn(1, self.in_planes))
            return Repeat("() c -> b c", self.in_planes)
        else:
            raise ValueError("Unknown pooling type: {}".format(token_type))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        cls = self.pool(x)
        x, cls = self.layer1(x, cls)
        x, cls = self.layer2(x, cls)
        x, cls = self.layer3(x, cls)
        x, cls = self.layer4(x, cls)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        out = self.linear(x, cls)
        return out


@register_model
def attnres18(pretrained=False, **kwargs):
    return AttnRes(BasicBlock, [2, 2, 2, 2], **kwargs)


@register_model
def sepattnres18(pretrained=False, **kwargs):
    return AttnRes(BasicBlock, [2, 2, 2, 2], classifier_type="sep", **kwargs)


@register_model
def catattnres18(pretrained=False, **kwargs):
    return AttnRes(BasicBlock, [2, 2, 2, 2], classifier_type="cat", **kwargs)


@register_model
def clsattnres18(pretrained=False, **kwargs):
    return AttnRes(BasicBlock, [2, 2, 2, 2], classifier_type="cls", **kwargs)


@register_model
def sepattnres18_norm(pretrained=False, **kwargs):
    return AttnRes(BasicBlock, [2, 2, 2, 2], classifier_type="sep", norm=True, **kwargs)


@register_model
def catattnres18_norm(pretrained=False, **kwargs):
    return AttnRes(BasicBlock, [2, 2, 2, 2], classifier_type="cat", norm=True, **kwargs)


@register_model
def clsattnres18_norm(pretrained=False, **kwargs):
    return AttnRes(BasicBlock, [2, 2, 2, 2], classifier_type="cls", norm=True, **kwargs)


@register_model
def seprestoken18(pretrained=False, **kwargs):
    return AttnRes(
        BasicBlock, [2, 2, 2, 2], token_type="learnable", classifier_type="sep", **kwargs
    )


@register_model
def catrestoken18(pretrained=False, **kwargs):
    return AttnRes(
        BasicBlock, [2, 2, 2, 2], token_type="learnable", classifier_type="cat", **kwargs
    )


@register_model
def clsrestoken18(pretrained=False, **kwargs):
    return AttnRes(
        BasicBlock, [2, 2, 2, 2], token_type="learnable", classifier_type="cls", **kwargs
    )


@register_model
def seprestoken18_norm(pretrained=False, **kwargs):
    return AttnRes(
        BasicBlock,
        [2, 2, 2, 2],
        token_type="learnable",
        classifier_type="sep",
        norm=True,
        **kwargs
    )


@register_model
def catrestoken18_norm(pretrained=False, **kwargs):
    return AttnRes(
        BasicBlock,
        [2, 2, 2, 2],
        token_type="learnable",
        classifier_type="cat",
        norm=True,
        **kwargs
    )


@register_model
def clsrestoken18_norm(pretrained=False, **kwargs):
    return AttnRes(
        BasicBlock,
        [2, 2, 2, 2],
        token_type="learnable",
        classifier_type="cls",
        norm=True,
        **kwargs
    )
