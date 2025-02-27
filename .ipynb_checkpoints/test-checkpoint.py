import torch
from eval import evaluate_model_partial

def test(model, test_loader, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    f1, iou = evaluate_model_partial(model, test_loader)
    print(f"Test F1 Score: {f1:.4f}, Test IoU: {iou:.4f}")
    return f1, iou