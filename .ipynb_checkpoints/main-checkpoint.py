import os
import torch
from model import UNetFPN
from train import train
from eval import evaluate_model_partial, plot_metrics
from dataloader import create_dataloaders
from test import test
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

def main():
    # 设置
    model = UNetFPN().cuda()
    optimizer = Adam(model.parameters(), lr=0.0005)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    train_loader, test_loader = create_dataloaders()
    num_epochs = 100
    output_dir = "output_images"
    os.makedirs(output_dir, exist_ok=True)

    # 训练循环
    loss_history = []
    f1_scores_history = []
    iou_scores_history = []
    best_f1 = -float('inf')
    best_iou = 0.0
    best_model_path = os.path.join(output_dir, 'best_model.pth')

    for epoch in range(num_epochs):
        avg_loss = train(model, train_loader, optimizer, scheduler, epoch, num_epochs)
        val_f1, val_iou = evaluate_model_partial(model, test_loader)
        loss_history.append(avg_loss)
        f1_scores_history.append(val_f1)
        iou_scores_history.append(val_iou)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Validation F1: {val_f1:.4f}, Validation IoU: {val_iou:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_iou = val_iou
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model with F1 Score: {best_f1:.4f} at epoch {epoch + 1}")

    print(f"\nTraining completed. Best F1 Score: {best_f1:.4f}, Corresponding IoU: {best_iou:.4f}")
    plot_metrics(loss_history, f1_scores_history, iou_scores_history, output_dir)

    # 测试最佳模型
    test(model, test_loader, best_model_path)

if __name__ == "__main__":
    main()