import torch
import numpy as np
from sklearn.metrics import f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def evaluate_model_partial(model, loader):
    model.eval()
    f1_scores = []
    iou_scores = []
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.cuda(), masks.cuda()
            outputs = model(images)
            preds = (outputs > 0.5).float()
            f1 = f1_score(masks.cpu().numpy().flatten(), preds.cpu().numpy().flatten())
            intersection = (preds * masks).sum().item()
            union = (preds + masks).sum().item() - intersection
            iou = intersection / (union + 1e-8)
            f1_scores.append(f1)
            iou_scores.append(iou)
    return np.mean(f1_scores), np.mean(iou_scores)

def plot_metrics(loss_history, f1_history, iou_history, output_dir):
    epochs = range(1, len(loss_history) + 1)
    
    plt.figure()
    plt.plot(epochs, loss_history, 'b-', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss.png'))
    plt.close()

    plt.figure()
    plt.plot(epochs, f1_history, 'g-', label='Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Validation F1 Score over Epochs')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'f1_score.png'))
    plt.close()

    plt.figure()
    plt.plot(epochs, iou_history, 'r-', label='Validation IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.title('Validation IoU over Epochs')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'iou.png'))
    plt.close()