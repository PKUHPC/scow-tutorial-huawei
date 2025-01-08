import torch
from torch import nn
import torch_npu
import torch.distributed as dist
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet18, ResNet18_Weights
import time
import os
from datetime import timedelta
import torch.multiprocessing as mp

torch.manual_seed(0)
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29500'
# 数据预处理
train_transforms = transforms.Compose([
    transforms.Resize(40),
    transforms.RandomResizedCrop(32, scale=(0.64, 1.0), ratio=(1.0, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

# 加载数据集
train_dataset = datasets.CIFAR10(root='./cifar', train=True, download=True, transform=train_transforms)
val_dataset = datasets.CIFAR10(root='./cifar', train=False, download=True, transform=val_transforms)

def ddp_setup(rank, world_size):
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)

def main_worker(rank, world_size, batch_size, device_ids):
    """
    rank: 当前进程的 rank
    world_size: 总进程数
    batch_size: 全局 batch size
    device_ids: 可用的设备 ID 列表
    """
    ddp_setup(rank, world_size)

    # 设置设备
    device_id = device_ids[rank]  # 根据 rank 获取对应的设备 ID
    torch_npu.npu.set_device(device_id)
    print(f"Process {rank} is using device npu:{device_id}")

    total_batch_size = batch_size
    total_workers = world_size

    batch_size = int(total_batch_size / world_size)
    workers = int((total_workers + world_size - 1) / world_size)

    # 使用 ResNet18 模型
    model = resnet18(weights=None, num_classes=10)

    loc = f'npu:{device_id}'
    model = model.to(loc)
    criterion = nn.CrossEntropyLoss().to(loc)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    test_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, sampler=test_sampler, drop_last=True)

    model = nn.parallel.DistributedDataParallel(model, device_ids=[device_id])

    for epoch in range(5):
        print(f"Epoch {epoch+1} start")
        train_sampler.set_epoch(epoch)
        average_loss, average_load_time, average_train_time = train(train_loader, model, criterion, optimizer, epoch, device_id)

        # 验证
        val_start_time = time.time()
        accuracy_dict = accuracy(model, val_loader, loc)
        val_end_time = time.time()
        average_val_time = timedelta(seconds=val_end_time - val_start_time)

        # 输出信息
        print(f"loss: {average_loss:.4f} | test accuracy: {accuracy_dict:.2f}% | load_time: {average_load_time} | train_time: {average_train_time} | val_time: {average_val_time}")

        # 保存模型
        if rank == 0:  # 只在主进程中保存模型
            os.makedirs('./models', exist_ok=True)
            torch.save(model.state_dict(), f'./models/resnet18_epoch_{epoch+1}.pth')


def train(train_loader, model, criterion, optimizer, epoch, gpu):
    model.train()
    train_ls = []
    load_time = []
    train_time = []

    for i, (images, target) in enumerate(train_loader):
        loc = f'npu:{gpu}'
        
        # 加载数据
        start_load = time.time()
        images, target = images.to(loc, non_blocking=True), target.to(loc, non_blocking=True)
        end_load = time.time()
        load_time.append(end_load - start_load)

        # 前向传播和反向传播
        start_train = time.time()
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        end_train = time.time()
        train_time.append(end_train - start_train)

        train_ls.append(loss.item())

    average_loss = sum(train_ls) / len(train_ls)
    average_load_time = timedelta(seconds=sum(load_time))
    average_train_time = timedelta(seconds=sum(train_time))

    return average_loss, average_load_time, average_train_time

def accuracy(model, data_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return 100 * correct / total

def main():
    world_size = torch_npu.npu.device_count()
    batch_size = 512
    device_ids = [0, 1]  # 强制指定设备为 npu:1 和 npu:2
    mp.spawn(main_worker, args=(world_size, batch_size, device_ids), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
