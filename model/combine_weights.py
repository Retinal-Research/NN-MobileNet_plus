import os
import torch


mapping = [
    ("features.7.out.8.bn.weight",         "stages.2.0.conv3_1x1.bn.weight"),
    ("features.7.out.8.bn.bias",           "stages.2.0.conv3_1x1.bn.bias"),
    ("features.7.out.8.bn.running_mean",   "stages.2.0.conv3_1x1.bn.running_mean"),
    ("features.7.out.8.bn.running_var",    "stages.2.0.conv3_1x1.bn.running_var"),
]

mapping += [
    # local conv kxk
    ("features.7.out.9.conv1.0.weight",    "stages.2.1.conv_kxk.conv.weight"),
    ("features.7.out.9.conv1.1.weight",    "stages.2.1.conv_kxk.bn.weight"),
    ("features.7.out.9.conv1.1.bias",      "stages.2.1.conv_kxk.bn.bias"),
    ("features.7.out.9.conv1.1.running_mean","stages.2.1.conv_kxk.bn.running_mean"),
    ("features.7.out.9.conv1.1.running_var","stages.2.1.conv_kxk.bn.running_var"),

    # 1x1 提升到 dim（注意：timm 只有 weight，没 BN）
    ("features.7.out.9.conv2.0.weight",    "stages.2.1.conv_1x1.weight"),
    # transformer block 0
    ("features.7.out.9.transformer.layers.0.0.norm.weight", "stages.2.1.transformer.0.norm1.weight"),
    ("features.7.out.9.transformer.layers.0.0.norm.bias",   "stages.2.1.transformer.0.norm1.bias"),
    ("features.7.out.9.transformer.layers.0.0.fn.to_qkv.weight", "stages.2.1.transformer.0.attn.qkv.weight"),
    ("features.7.out.9.transformer.layers.0.0.fn.to_qkv.bias",   "stages.2.1.transformer.0.attn.qkv.bias"),
    ("features.7.out.9.transformer.layers.0.0.fn.to_out.0.weight","stages.2.1.transformer.0.attn.proj.weight"),
    ("features.7.out.9.transformer.layers.0.0.fn.to_out.0.bias",  "stages.2.1.transformer.0.attn.proj.bias"),
    ("features.7.out.9.transformer.layers.0.1.norm.weight",      "stages.2.1.transformer.0.norm2.weight"),
    ("features.7.out.9.transformer.layers.0.1.norm.bias",        "stages.2.1.transformer.0.norm2.bias"),
    ("features.7.out.9.transformer.layers.0.1.fn.net.0.weight",  "stages.2.1.transformer.0.mlp.fc1.weight"),
    ("features.7.out.9.transformer.layers.0.1.fn.net.0.bias",    "stages.2.1.transformer.0.mlp.fc1.bias"),
    ("features.7.out.9.transformer.layers.0.1.fn.net.3.weight",  "stages.2.1.transformer.0.mlp.fc2.weight"),
    ("features.7.out.9.transformer.layers.0.1.fn.net.3.bias",    "stages.2.1.transformer.0.mlp.fc2.bias"),

    # transformer block 1
    ("features.7.out.9.transformer.layers.1.0.norm.weight", "stages.2.1.transformer.1.norm1.weight"),
    ("features.7.out.9.transformer.layers.1.0.norm.bias",   "stages.2.1.transformer.1.norm1.bias"),
    ("features.7.out.9.transformer.layers.1.0.fn.to_qkv.weight", "stages.2.1.transformer.1.attn.qkv.weight"),
    ("features.7.out.9.transformer.layers.1.0.fn.to_qkv.bias",   "stages.2.1.transformer.1.attn.qkv.bias"),
    ("features.7.out.9.transformer.layers.1.0.fn.to_out.0.weight","stages.2.1.transformer.1.attn.proj.weight"),
    ("features.7.out.9.transformer.layers.1.0.fn.to_out.0.bias",  "stages.2.1.transformer.1.attn.proj.bias"),
    ("features.7.out.9.transformer.layers.1.1.norm.weight",      "stages.2.1.transformer.1.norm2.weight"),
    ("features.7.out.9.transformer.layers.1.1.norm.bias",        "stages.2.1.transformer.1.norm2.bias"),
    ("features.7.out.9.transformer.layers.1.1.fn.net.0.weight",  "stages.2.1.transformer.1.mlp.fc1.weight"),
    ("features.7.out.9.transformer.layers.1.1.fn.net.0.bias",    "stages.2.1.transformer.1.mlp.fc1.bias"),
    ("features.7.out.9.transformer.layers.1.1.fn.net.3.weight",  "stages.2.1.transformer.1.mlp.fc2.weight"),
    ("features.7.out.9.transformer.layers.1.1.fn.net.3.bias",    "stages.2.1.transformer.1.mlp.fc2.bias"),

    # proj + fusion
    ("features.7.out.9.conv3.0.weight",    "stages.2.1.conv_proj.conv.weight"),
    ("features.7.out.9.conv3.1.weight",    "stages.2.1.conv_proj.bn.weight"),
    ("features.7.out.9.conv3.1.bias",      "stages.2.1.conv_proj.bn.bias"),
    ("features.7.out.9.conv3.1.running_mean","stages.2.1.conv_proj.bn.running_mean"),
    ("features.7.out.9.conv3.1.running_var","stages.2.1.conv_proj.bn.running_var"),

    ("features.7.out.9.conv4.0.weight",    "stages.2.1.conv_fusion.conv.weight"),
    ("features.7.out.9.conv4.1.weight",    "stages.2.1.conv_fusion.bn.weight"),
    ("features.7.out.9.conv4.1.bias",      "stages.2.1.conv_fusion.bn.bias"),
    ("features.7.out.9.conv4.1.running_mean","stages.2.1.conv_fusion.bn.running_mean"),
    ("features.7.out.9.conv4.1.running_var","stages.2.1.conv_fusion.bn.running_var"),
]

mapping += [
    ("features.8.out.0.conv.0.weight",     "stages.3.0.conv1_1x1.conv.weight"),
    ("features.8.out.0.conv.1.weight",     "stages.3.0.conv1_1x1.bn.weight"),
    ("features.8.out.0.conv.1.bias",       "stages.3.0.conv1_1x1.bn.bias"),
    ("features.8.out.0.conv.1.running_mean","stages.3.0.conv1_1x1.bn.running_mean"),
    ("features.8.out.0.conv.1.running_var","stages.3.0.conv1_1x1.bn.running_var"),

    ("features.8.out.0.conv.3.weight",     "stages.3.0.conv2_kxk.conv.weight"),
    ("features.8.out.0.conv.4.weight",     "stages.3.0.conv2_kxk.bn.weight"),
    ("features.8.out.0.conv.4.bias",       "stages.3.0.conv2_kxk.bn.bias"),
    ("features.8.out.0.conv.4.running_mean","stages.3.0.conv2_kxk.bn.running_mean"),
    ("features.8.out.0.conv.4.running_var","stages.3.0.conv2_kxk.bn.running_var"),

    ("features.8.out.0.conv.6.weight",     "stages.3.0.conv3_1x1.conv.weight"),
    ("features.8.out.0.conv.7.weight",     "stages.3.0.conv3_1x1.bn.weight"),
    ("features.8.out.0.conv.7.bias",       "stages.3.0.conv3_1x1.bn.bias"),
    ("features.8.out.0.conv.7.running_mean","stages.3.0.conv3_1x1.bn.running_mean"),
    ("features.8.out.0.conv.7.running_var","stages.3.0.conv3_1x1.bn.running_var"),
]


mapping += [
    # local conv kxk
    ("features.8.out.1.conv1.0.weight",    "stages.3.1.conv_kxk.conv.weight"),
    ("features.8.out.1.conv1.1.weight",    "stages.3.1.conv_kxk.bn.weight"),
    ("features.8.out.1.conv1.1.bias",      "stages.3.1.conv_kxk.bn.bias"),
    ("features.8.out.1.conv1.1.running_mean","stages.3.1.conv_kxk.bn.running_mean"),
    ("features.8.out.1.conv1.1.running_var","stages.3.1.conv_kxk.bn.running_var"),

    # 1x1 -> dim（timm 只有 weight）
    ("features.8.out.1.conv2.0.weight",    "stages.3.1.conv_1x1.weight"),

    # transformer blocks (0..3)
    # block 0
    ("features.8.out.1.transformer.layers.0.0.norm.weight", "stages.3.1.transformer.0.norm1.weight"),
    ("features.8.out.1.transformer.layers.0.0.norm.bias",   "stages.3.1.transformer.0.norm1.bias"),
    ("features.8.out.1.transformer.layers.0.0.fn.to_qkv.weight", "stages.3.1.transformer.0.attn.qkv.weight"),
    ("features.8.out.1.transformer.layers.0.0.fn.to_qkv.bias",   "stages.3.1.transformer.0.attn.qkv.bias"),
    ("features.8.out.1.transformer.layers.0.0.fn.to_out.0.weight","stages.3.1.transformer.0.attn.proj.weight"),
    ("features.8.out.1.transformer.layers.0.0.fn.to_out.0.bias",  "stages.3.1.transformer.0.attn.proj.bias"),
    ("features.8.out.1.transformer.layers.0.1.norm.weight",      "stages.3.1.transformer.0.norm2.weight"),
    ("features.8.out.1.transformer.layers.0.1.norm.bias",        "stages.3.1.transformer.0.norm2.bias"),
    ("features.8.out.1.transformer.layers.0.1.fn.net.0.weight",  "stages.3.1.transformer.0.mlp.fc1.weight"),
    ("features.8.out.1.transformer.layers.0.1.fn.net.0.bias",    "stages.3.1.transformer.0.mlp.fc1.bias"),
    ("features.8.out.1.transformer.layers.0.1.fn.net.3.weight",  "stages.3.1.transformer.0.mlp.fc2.weight"),
    ("features.8.out.1.transformer.layers.0.1.fn.net.3.bias",    "stages.3.1.transformer.0.mlp.fc2.bias"),
    # block 1
    ("features.8.out.1.transformer.layers.1.0.norm.weight", "stages.3.1.transformer.1.norm1.weight"),
    ("features.8.out.1.transformer.layers.1.0.norm.bias",   "stages.3.1.transformer.1.norm1.bias"),
    ("features.8.out.1.transformer.layers.1.0.fn.to_qkv.weight", "stages.3.1.transformer.1.attn.qkv.weight"),
    ("features.8.out.1.transformer.layers.1.0.fn.to_qkv.bias",   "stages.3.1.transformer.1.attn.qkv.bias"),
    ("features.8.out.1.transformer.layers.1.0.fn.to_out.0.weight","stages.3.1.transformer.1.attn.proj.weight"),
    ("features.8.out.1.transformer.layers.1.0.fn.to_out.0.bias",  "stages.3.1.transformer.1.attn.proj.bias"),
    ("features.8.out.1.transformer.layers.1.1.norm.weight",      "stages.3.1.transformer.1.norm2.weight"),
    ("features.8.out.1.transformer.layers.1.1.norm.bias",        "stages.3.1.transformer.1.norm2.bias"),
    ("features.8.out.1.transformer.layers.1.1.fn.net.0.weight",  "stages.3.1.transformer.1.mlp.fc1.weight"),
    ("features.8.out.1.transformer.layers.1.1.fn.net.0.bias",    "stages.3.1.transformer.1.mlp.fc1.bias"),
    ("features.8.out.1.transformer.layers.1.1.fn.net.3.weight",  "stages.3.1.transformer.1.mlp.fc2.weight"),
    ("features.8.out.1.transformer.layers.1.1.fn.net.3.bias",    "stages.3.1.transformer.1.mlp.fc2.bias"),
    # block 2
    ("features.8.out.1.transformer.layers.2.0.norm.weight", "stages.3.1.transformer.2.norm1.weight"),
    ("features.8.out.1.transformer.layers.2.0.norm.bias",   "stages.3.1.transformer.2.norm1.bias"),
    ("features.8.out.1.transformer.layers.2.0.fn.to_qkv.weight", "stages.3.1.transformer.2.attn.qkv.weight"),
    ("features.8.out.1.transformer.layers.2.0.fn.to_qkv.bias",   "stages.3.1.transformer.2.attn.qkv.bias"),
    ("features.8.out.1.transformer.layers.2.0.fn.to_out.0.weight","stages.3.1.transformer.2.attn.proj.weight"),
    ("features.8.out.1.transformer.layers.2.0.fn.to_out.0.bias",  "stages.3.1.transformer.2.attn.proj.bias"),
    ("features.8.out.1.transformer.layers.2.1.norm.weight",      "stages.3.1.transformer.2.norm2.weight"),
    ("features.8.out.1.transformer.layers.2.1.norm.bias",        "stages.3.1.transformer.2.norm2.bias"),
    ("features.8.out.1.transformer.layers.2.1.fn.net.0.weight",  "stages.3.1.transformer.2.mlp.fc1.weight"),
    ("features.8.out.1.transformer.layers.2.1.fn.net.0.bias",    "stages.3.1.transformer.2.mlp.fc1.bias"),
    ("features.8.out.1.transformer.layers.2.1.fn.net.3.weight",  "stages.3.1.transformer.2.mlp.fc2.weight"),
    ("features.8.out.1.transformer.layers.2.1.fn.net.3.bias",    "stages.3.1.transformer.2.mlp.fc2.bias"),
    # block 3
    ("features.8.out.1.transformer.layers.3.0.norm.weight", "stages.3.1.transformer.3.norm1.weight"),
    ("features.8.out.1.transformer.layers.3.0.norm.bias",   "stages.3.1.transformer.3.norm1.bias"),
    ("features.8.out.1.transformer.layers.3.0.fn.to_qkv.weight", "stages.3.1.transformer.3.attn.qkv.weight"),
    ("features.8.out.1.transformer.layers.3.0.fn.to_qkv.bias",   "stages.3.1.transformer.3.attn.qkv.bias"),
    ("features.8.out.1.transformer.layers.3.0.fn.to_out.0.weight","stages.3.1.transformer.3.attn.proj.weight"),
    ("features.8.out.1.transformer.layers.3.0.fn.to_out.0.bias",  "stages.3.1.transformer.3.attn.proj.bias"),
    ("features.8.out.1.transformer.layers.3.1.norm.weight",      "stages.3.1.transformer.3.norm2.weight"),
    ("features.8.out.1.transformer.layers.3.1.norm.bias",        "stages.3.1.transformer.3.norm2.bias"),
    ("features.8.out.1.transformer.layers.3.1.fn.net.0.weight",  "stages.3.1.transformer.3.mlp.fc1.weight"),
    ("features.8.out.1.transformer.layers.3.1.fn.net.0.bias",    "stages.3.1.transformer.3.mlp.fc1.bias"),
    ("features.8.out.1.transformer.layers.3.1.fn.net.3.weight",  "stages.3.1.transformer.3.mlp.fc2.weight"),
    ("features.8.out.1.transformer.layers.3.1.fn.net.3.bias",    "stages.3.1.transformer.3.mlp.fc2.bias"),

    # proj + fusion
    ("features.8.out.1.conv3.0.weight",    "stages.3.1.conv_proj.conv.weight"),
    ("features.8.out.1.conv3.1.weight",    "stages.3.1.conv_proj.bn.weight"),
    ("features.8.out.1.conv3.1.bias",      "stages.3.1.conv_proj.bn.bias"),
    ("features.8.out.1.conv3.1.running_mean","stages.3.1.conv_proj.bn.running_mean"),
    ("features.8.out.1.conv3.1.running_var","stages.3.1.conv_proj.bn.running_var"),

    ("features.8.out.1.conv4.0.weight",    "stages.3.1.conv_fusion.conv.weight"),
    ("features.8.out.1.conv4.1.weight",    "stages.3.1.conv_fusion.bn.weight"),
    ("features.8.out.1.conv4.1.bias",      "stages.3.1.conv_fusion.bn.bias"),
    ("features.8.out.1.conv4.1.running_mean","stages.3.1.conv_fusion.bn.running_mean"),
    ("features.8.out.1.conv4.1.running_var","stages.3.1.conv_fusion.bn.running_var"),
]

mapping += [
    ("features.9.out.0.conv.0.weight",     "stages.4.0.conv1_1x1.conv.weight"),
    ("features.9.out.0.conv.1.weight",     "stages.4.0.conv1_1x1.bn.weight"),
    ("features.9.out.0.conv.1.bias",       "stages.4.0.conv1_1x1.bn.bias"),
    ("features.9.out.0.conv.1.running_mean","stages.4.0.conv1_1x1.bn.running_mean"),
    ("features.9.out.0.conv.1.running_var","stages.4.0.conv1_1x1.bn.running_var"),

    ("features.9.out.0.conv.3.weight",     "stages.4.0.conv2_kxk.conv.weight"),
    ("features.9.out.0.conv.4.weight",     "stages.4.0.conv2_kxk.bn.weight"),
    ("features.9.out.0.conv.4.bias",       "stages.4.0.conv2_kxk.bn.bias"),
    ("features.9.out.0.conv.4.running_mean","stages.4.0.conv2_kxk.bn.running_mean"),
    ("features.9.out.0.conv.4.running_var","stages.4.0.conv2_kxk.bn.running_var"),

    ("features.9.out.0.conv.6.weight",     "stages.4.0.conv3_1x1.conv.weight"),
    ("features.9.out.0.conv.7.weight",     "stages.4.0.conv3_1x1.bn.weight"),
    ("features.9.out.0.conv.7.bias",       "stages.4.0.conv3_1x1.bn.bias"),
    ("features.9.out.0.conv.7.running_mean","stages.4.0.conv3_1x1.bn.running_mean"),
    ("features.9.out.0.conv.7.running_var","stages.4.0.conv3_1x1.bn.running_var"),
]


mapping += [
    # local conv kxk
    ("features.9.out.1.conv1.0.weight",    "stages.4.1.conv_kxk.conv.weight"),
    ("features.9.out.1.conv1.1.weight",    "stages.4.1.conv_kxk.bn.weight"),
    ("features.9.out.1.conv1.1.bias",      "stages.4.1.conv_kxk.bn.bias"),
    ("features.9.out.1.conv1.1.running_mean","stages.4.1.conv_kxk.bn.running_mean"),
    ("features.9.out.1.conv1.1.running_var","stages.4.1.conv_kxk.bn.running_var"),

    # 1x1 -> dim（timm 只有 weight）
    ("features.9.out.1.conv2.0.weight",    "stages.4.1.conv_1x1.weight"),

    # transformer blocks (0..2)
    # block 0
    ("features.9.out.1.transformer.layers.0.0.norm.weight", "stages.4.1.transformer.0.norm1.weight"),
    ("features.9.out.1.transformer.layers.0.0.norm.bias",   "stages.4.1.transformer.0.norm1.bias"),
    ("features.9.out.1.transformer.layers.0.0.fn.to_qkv.weight", "stages.4.1.transformer.0.attn.qkv.weight"),
    ("features.9.out.1.transformer.layers.0.0.fn.to_qkv.bias",   "stages.4.1.transformer.0.attn.qkv.bias"),
    ("features.9.out.1.transformer.layers.0.0.fn.to_out.0.weight","stages.4.1.transformer.0.attn.proj.weight"),
    ("features.9.out.1.transformer.layers.0.0.fn.to_out.0.bias",  "stages.4.1.transformer.0.attn.proj.bias"),
    ("features.9.out.1.transformer.layers.0.1.norm.weight",      "stages.4.1.transformer.0.norm2.weight"),
    ("features.9.out.1.transformer.layers.0.1.norm.bias",        "stages.4.1.transformer.0.norm2.bias"),
    ("features.9.out.1.transformer.layers.0.1.fn.net.0.weight",  "stages.4.1.transformer.0.mlp.fc1.weight"),
    ("features.9.out.1.transformer.layers.0.1.fn.net.0.bias",    "stages.4.1.transformer.0.mlp.fc1.bias"),
    ("features.9.out.1.transformer.layers.0.1.fn.net.3.weight",  "stages.4.1.transformer.0.mlp.fc2.weight"),
    ("features.9.out.1.transformer.layers.0.1.fn.net.3.bias",    "stages.4.1.transformer.0.mlp.fc2.bias"),
    # block 1
    ("features.9.out.1.transformer.layers.1.0.norm.weight", "stages.4.1.transformer.1.norm1.weight"),
    ("features.9.out.1.transformer.layers.1.0.norm.bias",   "stages.4.1.transformer.1.norm1.bias"),
    ("features.9.out.1.transformer.layers.1.0.fn.to_qkv.weight", "stages.4.1.transformer.1.attn.qkv.weight"),
    ("features.9.out.1.transformer.layers.1.0.fn.to_qkv.bias",   "stages.4.1.transformer.1.attn.qkv.bias"),
    ("features.9.out.1.transformer.layers.1.0.fn.to_out.0.weight","stages.4.1.transformer.1.attn.proj.weight"),
    ("features.9.out.1.transformer.layers.1.0.fn.to_out.0.bias",  "stages.4.1.transformer.1.attn.proj.bias"),
    ("features.9.out.1.transformer.layers.1.1.norm.weight",      "stages.4.1.transformer.1.norm2.weight"),
    ("features.9.out.1.transformer.layers.1.1.norm.bias",        "stages.4.1.transformer.1.norm2.bias"),
    ("features.9.out.1.transformer.layers.1.1.fn.net.0.weight",  "stages.4.1.transformer.1.mlp.fc1.weight"),
    ("features.9.out.1.transformer.layers.1.1.fn.net.0.bias",    "stages.4.1.transformer.1.mlp.fc1.bias"),
    ("features.9.out.1.transformer.layers.1.1.fn.net.3.weight",  "stages.4.1.transformer.1.mlp.fc2.weight"),
    ("features.9.out.1.transformer.layers.1.1.fn.net.3.bias",    "stages.4.1.transformer.1.mlp.fc2.bias"),
    # block 2
    ("features.9.out.1.transformer.layers.2.0.norm.weight", "stages.4.1.transformer.2.norm1.weight"),
    ("features.9.out.1.transformer.layers.2.0.norm.bias",   "stages.4.1.transformer.2.norm1.bias"),
    ("features.9.out.1.transformer.layers.2.0.fn.to_qkv.weight", "stages.4.1.transformer.2.attn.qkv.weight"),
    ("features.9.out.1.transformer.layers.2.0.fn.to_qkv.bias",   "stages.4.1.transformer.2.attn.qkv.bias"),
    ("features.9.out.1.transformer.layers.2.0.fn.to_out.0.weight","stages.4.1.transformer.2.attn.proj.weight"),
    ("features.9.out.1.transformer.layers.2.0.fn.to_out.0.bias",  "stages.4.1.transformer.2.attn.proj.bias"),
    ("features.9.out.1.transformer.layers.2.1.norm.weight",      "stages.4.1.transformer.2.norm2.weight"),
    ("features.9.out.1.transformer.layers.2.1.norm.bias",        "stages.4.1.transformer.2.norm2.bias"),
    ("features.9.out.1.transformer.layers.2.1.fn.net.0.weight",  "stages.4.1.transformer.2.mlp.fc1.weight"),
    ("features.9.out.1.transformer.layers.2.1.fn.net.0.bias",    "stages.4.1.transformer.2.mlp.fc1.bias"),
    ("features.9.out.1.transformer.layers.2.1.fn.net.3.weight",  "stages.4.1.transformer.2.mlp.fc2.weight"),
    ("features.9.out.1.transformer.layers.2.1.fn.net.3.bias",    "stages.4.1.transformer.2.mlp.fc2.bias"),

    # proj + fusion
    ("features.9.out.1.conv3.0.weight",    "stages.4.1.conv_proj.conv.weight"),
    ("features.9.out.1.conv3.1.weight",    "stages.4.1.conv_proj.bn.weight"),
    ("features.9.out.1.conv3.1.bias",      "stages.4.1.conv_proj.bn.bias"),
    ("features.9.out.1.conv3.1.running_mean","stages.4.1.conv_proj.bn.running_mean"),
    ("features.9.out.1.conv3.1.running_var","stages.4.1.conv_proj.bn.running_var"),

    ("features.9.out.1.conv4.0.weight",    "stages.4.1.conv_fusion.conv.weight"),
    ("features.9.out.1.conv4.1.weight",    "stages.4.1.conv_fusion.bn.weight"),
    ("features.9.out.1.conv4.1.bias",      "stages.4.1.conv_fusion.bn.bias"),
    ("features.9.out.1.conv4.1.running_mean","stages.4.1.conv_fusion.bn.running_mean"),
    ("features.9.out.1.conv4.1.running_var","stages.4.1.conv_fusion.bn.running_var"),
]

mapping += [
    ("features.10.0.weight", "final_conv.conv.weight"),
    ("features.10.1.weight", "final_conv.bn.weight"),
    ("features.10.1.bias",   "final_conv.bn.bias"),
    ("features.10.1.running_mean", "final_conv.bn.running_mean"),
    ("features.10.1.running_var",  "final_conv.bn.running_var"),
]

def inspect_checkpoint(ckpt_path: str, show_first: int = 30, filter_substr: str | None = None):
    """
    Quick sanity-check for a weights file.

    - Loads .pth / .pt (cpu)
    - Handles both {"state_dict": ...} and plain state_dict
    - Prints count, sample keys, and tensor shapes
    - Optionally filter by substring to only show matching keys

    Args
    ----
    ckpt_path : str
        Path to checkpoint file
    show_first : int
        How many entries to preview (after optional filtering)
    filter_substr : str | None
        Only show keys containing this substring (e.g., "transformer" / "features.7")
    """
    assert os.path.isfile(ckpt_path), f"File not found: {ckpt_path}"
    obj = torch.load(ckpt_path, map_location="cpu")

    # unwrap common containers
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        sd = obj["state_dict"]
        wrapper = True
    elif isinstance(obj, dict):
        # if it's already a state_dict-like mapping
        sd = {k: v for k, v in obj.items() if torch.is_tensor(v)}
        wrapper = False
    else:
        raise TypeError(f"Unsupported checkpoint format: {type(obj)}")

    keys = list(sd.keys())
    if filter_substr:
        keys = [k for k in keys if filter_substr in k]

    print(f"✅ Loaded: {ckpt_path}")
    print(f" - Wrapped under 'state_dict': {wrapper}")
    print(f" - Total tensors: {len(sd)} (showing {min(show_first, len(keys))})")
    if filter_substr:
        print(f" - Filter: '{filter_substr}'  -> matched {len(keys)} tensors")

    # parameter count (only filtered if filter_substr provided)
    def _numel(t): return t.numel() if torch.is_tensor(t) else 0
    total_params = sum(_numel(sd[k]) for k in keys)
    print(f" - Total elements (matched): {total_params:,}")

    for i, k in enumerate(keys[:show_first], 1):
        v = sd[k]
        shape = tuple(v.shape) if torch.is_tensor(v) else ()
        dtype = getattr(v, "dtype", None)
        print(f"{i:3d}. {k:60s} {shape} {str(dtype)}")


def check_load_compat(model: torch.nn.Module, ckpt_path: str, strict: bool = False):
    """
    Try loading weights into `model` and report:
      - missing keys (in model but not in ckpt)
      - unexpected keys (in ckpt but not in model)
      - shape mismatches (caught by manual compare)

    Returns (missing_keys, unexpected_keys, shape_mismatch)
    """
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    model_sd = model.state_dict()
    shape_mismatch = []

    # manual shape check on intersecting keys
    for k in set(sd.keys()).intersection(model_sd.keys()):
        if torch.is_tensor(sd[k]) and torch.is_tensor(model_sd[k]):
            if sd[k].shape != model_sd[k].shape:
                shape_mismatch.append((k, tuple(sd[k].shape), tuple(model_sd[k].shape)))

    missing, unexpected = model.load_state_dict(sd, strict=strict)

    print(f"🔍 Load check (strict={strict}):")
    print(f" - Missing keys:    {len(missing)}")
    print(f" - Unexpected keys: {len(unexpected)}")
    print(f" - Shape mismatch:  {len(shape_mismatch)}")
    if shape_mismatch:
        print("   (first 20)")
        for k, s_src, s_dst in shape_mismatch[:20]:
            print(f"   * {k}: ckpt{list(s_src)} -> model{list(s_dst)}")

    return missing, unexpected, shape_mismatch


def load_cnn_part(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)  # 有的 ckpt 存在 "state_dict" 里

    new_state = {}
    for k, v in state_dict.items():
        if k in model.state_dict():
            if k.startswith("features.6.out.7.weight") or k.startswith("features.7.out.7.weight"):
                continue
            if k.startswith("features.8"):
                break  # 到 features.7 停止
            new_state[k] = v
    # 打印最终加载的 key
    print("\n--- Loaded Keys ---")
    for k in new_state.keys():
        print(k)
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print(f"✅ Loaded {len(new_state)} params (up to features.7)")
    print(f"   missing: {len(missing)}, unexpected: {len(unexpected)}")

    return model


import torch

def load_from_mapping(model, src_state, mapping, strict=False):
    """按映射关系把 src_state 里的参数拷到 model 上。"""
    dst = model.state_dict()
    copied, skipped = 0, []

    for m_key, s_key in mapping:
        if m_key not in dst:
            skipped.append((m_key, s_key, "dst-missing"))
            continue
        if s_key not in src_state:
            skipped.append((m_key, s_key, "src-missing"))
            continue
        if dst[m_key].shape != src_state[s_key].shape:
            skipped.append((m_key, s_key, f"shape {tuple(src_state[s_key].shape)} -> {tuple(dst[m_key].shape)}"))
            continue
        dst[m_key].copy_(src_state[s_key].to(dst[m_key].dtype))
        copied += 1

    model.load_state_dict(dst, strict=strict)
    print(f"✅ copied {copied} tensors from mapping; skipped {len(skipped)}")
    if skipped:
        for m_key, s_key, reason in skipped[:12]:
            print("  - skip:", m_key, "<-", s_key, "|", reason)
        if len(skipped) > 12:
            print(f"  ... (+{len(skipped)-12} more)")
    return model

import torch

def save_merged_weights(model, out_path="merged_weights.pth"):
    """
    把当前模型(已偷好权重)的 state_dict 存成 .pth
    """
    sd = model.state_dict()
    torch.save({"state_dict": sd}, out_path)
    print(f"✅ saved {len(sd)} tensors to: {out_path}")

def inspect_pth_keys(pth_path, prefix=None, limit=None, to_txt=None):
    """
    查看 .pth 里有哪些 key 和 shape
    - prefix: 只看某个前缀（比如 'features.7.'），默认看全部
    - limit: 只打印前 N 个
    - to_txt: 同时把列表写到文本文件
    """
    obj = torch.load(pth_path, map_location="cpu")
    sd = obj.get("state_dict", obj)

    items = [(k, tuple(v.shape)) for k, v in sd.items()]
    if prefix:
        items = [(k,s) for k,s in items if k.startswith(prefix)]

    total = len(items)
    if limit:
        items_print = items[:limit]
    else:
        items_print = items

    for k, s in items_print:
        print(f"{k:60s} {s}")
    print(f"\n📦 total tensors{' with prefix '+prefix if prefix else ''}: {total}")

    if to_txt:
        with open(to_txt, "w") as f:
            for k, s in items:
                f.write(f"{k}\t{s}\n")
        print(f"📝 also wrote list to: {to_txt}")


from timm import create_model
from resnext import ReXNetV1
inspect_checkpoint("/scratch/xinli38/nn-mobilenet++/rexnetv1_1.0.pth", show_first=100)
model = ReXNetV1(width_mult=1.0)
# check_load_compat(model, "/scratch/xinli38/nn-mobilenet++/rexnetv1_1.0.pth", strict=False)
print("\n权重文件里的参数:")
for k, v in model.state_dict().items():
    print(k, v.shape)


# chenged_modek = load_cnn_part(model,"/scratch/xinli38/nn-mobilenet++/rexnetv1_1.0.pth")
# timm_model = create_model("mobilevit_xs", pretrained=True)

# src_state = timm_model.state_dict()

# # 2) 把映射好的权重喂到你的模型
# load_from_mapping(chenged_modek, src_state, mapping, strict=False)

# save_merged_weights(model, "/scratch/xinli38/nn-mobilenet++/pretrained_weights/merged_pretrained.pth")

# # 2) 看全部 key/shape（只打印前 50 个）
# inspect_checkpoint("/scratch/xinli38/nn-mobilenet++/pretrained_weights/merged_pretrained.pth", show_first=500)

# 3) 只看 ViT 这段，比如 features.7. 开头，并导出到文本
# inspect_pth_keys("/scratch/xinli38/nn-mobilenet++/pretrained_weights/merged_pretrained.pth", prefix="*", to_txt="feat7_keys.txt")


import torch
ckpt = torch.load("/scratch/xinli38/nn-mobilenet++/pretrained_weights/merged_pretrained.pth", map_location="cpu")

# 查看所有 key
print(list(ckpt.keys())[:20])   # 先打印前 20 个 key 看看

# 假设你要看 "features.3.out.0.weight"
key = "features.7.out.8.bn.weightt"
if key in ckpt:
    tensor = ckpt[key]
    print(f"{key} -> shape: {tensor.shape}")
    print("values:", tensor.flatten()[:20])  # 打印前 20 个值
else:
    print(f"{key} not found in checkpoint")





























