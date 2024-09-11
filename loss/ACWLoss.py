import torch
import torch.nn as nn
import torch.nn.functional as F


class ACWLoss(nn.Module):
    def __init__(self, quan, alpha):
        super().__init__()
        self.alpha = alpha
        self.beta = 2 - alpha
        self.quantity = quan

    def forward(self, inputs, targets):
        class_num = inputs.size(1)
        inputs = inputs.permute(0, 2, 3, 1).reshape(-1, class_num)
        targets = targets.reshape(-1, 1)
        batch_size = inputs.size(0)

        predicted_values = F.softmax(inputs, 1)

        # 1 hot encoding
        y_zerohot = torch.zeros(batch_size, class_num).scatter_(1, targets.view(batch_size, 1).data.cpu(), 1).cuda()

        th = torch.quantile(predicted_values[y_zerohot > 0].clone(), self.quantity, dim=0)
        mask_h = (predicted_values > th)
        mask_l = ~mask_h
        if (mask_h * y_zerohot).sum():
            la = (-torch.log(predicted_values + 1e-7) * mask_h * y_zerohot).sum() / ((mask_h * y_zerohot).sum())
        else:
            la = 0
        if (mask_l * y_zerohot).sum():
            lb = (-torch.log(predicted_values + 1e-7) * mask_l * y_zerohot).sum() / ((mask_l * y_zerohot).sum())
        else:
            lb = 0

        loss = la * self.alpha + lb * self.beta
        return loss
