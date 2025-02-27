import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

def weighted_bce_loss(output, target, pos_weight=10):
    loss = -(pos_weight * target * torch.log(output + 1e-8) + (1 - target) * torch.log(1 - output + 1e-8))
    return torch.mean(loss)

def train(model, train_loader, optimizer, scheduler, epoch, num_epochs):
    model.train()
    total_loss = 0
    for images, masks in train_loader:
        images, masks = images.cuda(), masks.cuda()
        optimizer.zero_grad()
        outputs = model(images)
        loss = weighted_bce_loss(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    scheduler.step(avg_loss)
    return avg_loss