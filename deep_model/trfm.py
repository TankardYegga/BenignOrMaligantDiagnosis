# -*- encoding: utf-8 -*-
"""
@File    : densenet_w_trfm.py
@Time    : 3/7/2022 9:11 AM
@Author  : Liuwen Zou
@Email   : levinforward@163.com
@Software: PyCharm
"""
import torch
import torch.nn as nn


class WindowAttention(nn.Module):

    @staticmethod
    def weight_init(m):
        # 1. 根据网络层的不同定义不同的初始化方式  
        if isinstance(m, nn.Linear):
          nn.init.xavier_normal_(m.weight)
          nn.init.constant_(m.bias, 0)
        # 也可以判断是否为conv2d，使用相应的初始化方式 
        elif isinstance(m, nn.Conv2d):
           nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        # 是否为批归一化层
        elif isinstance(m, nn.BatchNorm2d):
          nn.init.constant_(m.weight, 1)
          nn.init.constant_(m.bias, 0)


    def __init__(self, dim, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)
        self.apply(self.weight_init)


    def forward(self, x):

        batch_size, seq_length, embed_dim = x.shape

        # [batch_size, seq_length, embed_dim * 3]
        qkv = self.qkv(x)
        # [batch_size, seq_length, 3, embed_dim]
        qkv = qkv.reshape(batch_size, seq_length, 3, embed_dim)
        # [batch_size, seq_length, 3, num_heads, embed_dim // num_heads]
        qkv = qkv.reshape(batch_size, seq_length, 3, self.num_heads, embed_dim // self.num_heads)
        # [3, batch_size, num_heads, seq_length, embed_dim // num_heads]
        qkv = qkv.permute(2, 0, 3, 1, 4)
        # batch_size, num_heads, seq_length, embed_dim // num_heads]
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        # [batch_size, num_heads, seq_length, embed_dim // num_heads] @ batch_size, num_heads,  embed_dim // num_heads, seq_length]
        # = [ batch_size, num_heads, seq_length, seq_length]
        attn_map = ( q @ k.transpose(-2, -1) )
        attn_score = self.softmax(attn_map)
        attn_score = self.attn_drop(attn_score)

        # get_attentioned_value
        # [ batch_size, num_heads, seq_length, seq_length] @ [batch_size, num_heads, seq_length, embed_dim // num_heads]
        # = [ batch_size, num_heads, seq_length, embed_dim // num_heads ]
        x = attn_score @ v
        # [ batch_size, seq_length, num_heads, embed_dim // num_heads ]
        x = x.transpose(1, 2)
        # [ batch_size, seq_length, embed_dim]
        x = x.reshape(batch_size, seq_length, embed_dim)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SmallWindowAttention(nn.Module):

    def __init__(self, dim, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x):

        batch_size, seq_length, embed_dim = x.shape

        # [batch_size, seq_length, embed_dim * 2]
        qkv = self.qkv(x)
        # [batch_size, seq_length, 2, embed_dim]
        qkv = qkv.reshape(batch_size, seq_length, 2, embed_dim)
        # [batch_size, seq_length, 2, num_heads, embed_dim // num_heads]
        qkv = qkv.reshape(batch_size, seq_length, 2, self.num_heads, embed_dim // self.num_heads)
        # [2, batch_size, num_heads, seq_length, embed_dim // num_heads]
        qkv = qkv.permute(2, 0, 3, 1, 4)
        # batch_size, num_heads, seq_length, embed_dim // num_heads]
        q, v = qkv.unbind(0)

        q = q * self.scale
        # [batch_size, num_heads, seq_length, embed_dim // num_heads] @ batch_size, num_heads,  embed_dim // num_heads, seq_length]
        # = [ batch_size, num_heads, seq_length, seq_length]
        attn_map = ( q @ q.transpose(-2, -1) )
        attn_score = self.softmax(attn_map)
        attn_score = self.attn_drop(attn_score)

        # get_attentioned_value
        # [ batch_size, num_heads, seq_length, seq_length] @ [batch_size, num_heads, seq_length, embed_dim // num_heads]
        # = [ batch_size, num_heads, seq_length, embed_dim // num_heads ]
        x = attn_score @ v
        # [ batch_size, seq_length, num_heads, embed_dim // num_heads ]
        x = x.transpose(1, 2)
        # [ batch_size, seq_length, embed_dim]
        x = x.reshape(batch_size, seq_length, embed_dim)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


if __name__ == '__main__':
    # x = torch.randn((25))
    # y = torch.randn((25))
    # z = torch.randn((25))
    # t = torch.randn((25))

    # stack_inputs = torch.stack((x, y, z, t), axis=0).unsqueeze(0)
    # print(stack_inputs.shape)

    # model = WindowAttention(dim=25, num_heads=5)
    # res = model.forward(stack_inputs)
    # print(res.shape)
    # print(stack_inputs)
    # print(res)

    x = torch.randn((25))
    y = torch.randn((25))
    z = torch.randn((25))
    t = torch.randn((25))

    stack_inputs = torch.stack((x, y, z, t), axis=0).unsqueeze(0)
    print(stack_inputs.shape)

    model = SmallWindowAttention(dim=25, num_heads=5)
    res = model.forward(stack_inputs)
    print(res.shape)
    print(stack_inputs)
    print(res)