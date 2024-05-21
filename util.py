# from math import ceil
import torch.nn as nn
# import re
# from custom_op.conv_avg import Conv2dAvg
# from custom_op.conv_svd_with_var import Conv2dSVD_with_var
# from custom_op.conv_hosvd_with_var import Conv2dHOSVD_with_var
# import torch



def get_all_conv_with_name(model):
    conv_layers = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.modules.conv.Conv2d):
            conv_layers[name] = mod
    return conv_layers
    

def get_memory_usage(process):
    mem_info = process.memory_info()
    return mem_info.rss

# def get_active_conv_with_name(model):
#     total_conv_layer = get_all_conv_with_name(model)
#     if model.num_of_finetune == "all" or model.num_of_finetune > len(total_conv_layer):
#         return total_conv_layer
#     elif model.num_of_finetune == None or model.num_of_finetune == 0:
#         return -1 # Không có conv layer nào được finetuned
#     else:
#         active_conv_layers = dict(list(total_conv_layer.items())[-model.num_of_finetune:]) # Chỉ áp dụng filter vào num_of_finetune conv2d layer cuối
#         return active_conv_layers