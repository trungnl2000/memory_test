import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torchvision.models import mobilenet_v2
from torch.utils.data import DataLoader, Subset
from util import get_all_conv_with_name
from custom_op.register import register_HOSVD_with_var
import psutil
import os
from util import get_memory_usage

torch.manual_seed(233)
# Specify the batch size
batch_size = 1

# Load the pretrained MobileNetV2 model
model = mobilenet_v2(pretrained=True)

# Modify the last layer for CIFAR-10 (10 classes)
model.classifier[1] = nn.Linear(model.last_channel, 10)

# Define a simple transformation to match MobileNetV2's expected input
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the CIFAR-10 dataset and select a single image
dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
subset_indices = list(range(batch_size))
single_batch_dataset = Subset(dataset, subset_indices)  # Use only the first image for simplicity
dataloader = DataLoader(single_batch_dataset, batch_size=batch_size, shuffle=True)

# Set up the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Move model to GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
model.to(device)

num_of_finetune = 1

all_convolution_layers = get_all_conv_with_name(model)

finetuned_conv_layers = dict(list(all_convolution_layers.items())[-num_of_finetune:])

filter_cfgs = {"cfgs": finetuned_conv_layers, "type": "conv", "SVD_var": 0.8}

register_HOSVD_with_var(model, filter_cfgs)



###### Freeze
for name, mod in model.named_modules():
    if len(list(mod.children())) == 0 and name not in finetuned_conv_layers.keys() and name != '':
        mod.eval()
        for param in mod.parameters():
            param.requires_grad = False # Freeze layer
    elif name in finetuned_conv_layers.keys(): # Khi duyệt đến layer sẽ được finetune => break. Vì đằng sau các conv2d layer này còn có thể có các lớp fc, gì gì đó mà vẫn cần finetune
        break

# Training loop (single iteration for demonstration)
model.train()
for inputs, labels in dataloader:
    inputs, labels = inputs.to(device), labels.to(device)

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    # Backward pass and optimization
    optimizer.zero_grad()

    # Get the process ID of the current script
    pid = os.getpid()
    process = psutil.Process(pid)

    loss.backward()
    optimizer.step()

    # Print loss
    # print(f'Training loss: {loss.item()}')
    print("________________HOSVD__________________")
    print(f"Memory usage: {get_memory_usage(process) / (1024 ** 2)} MB")

print('Training step completed.')
