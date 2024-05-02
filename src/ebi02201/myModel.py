import torch
import torch.nn as nn
from transformers import AutoModel
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=1.5, reduction='mean'): # gamma 2.0 is very strict with missing labels
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss
        
class SmoothFocalLoss(FocalLoss):
    def __init__(self, alpha=0.25, gamma=2.0, smoothing=0.1, reduction='mean'):
        super(SmoothFocalLoss, self).__init__(alpha, gamma, reduction)
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        with torch.no_grad():
            targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return super(SmoothFocalLoss, self).forward(inputs, targets)

class EntityModel(nn.Module):
    def __init__(self, model_name, num_labels, dropout_rate=0.3, pos_weights=None, loss_type='bce', loss_params=None):
        """
        Initialize the EntityModel.
        
        Args:
            model_name (str): Name of the model to be used from transformers.
            num_labels (int): Number of labels for the classification task.
            dropout_rate (float): Dropout rate for regularization.
            pos_weights (Tensor, optional): Weights for positive samples in case of imbalanced classes.
            loss_type (str): Type of the loss function ('bce', 'focal', or 'smooth_focal').
            loss_params (dict, optional): Parameters for the loss function. 
                                        E.g., for FocalLoss, provide {'alpha': 0.25, 'gamma': 2.0}.
        """
        super(EntityModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.model_dropout = nn.Dropout(dropout_rate)
        self.num_labels = num_labels
        self.label_classifier = nn.Linear(768, self.num_labels)
        
        # Set up the loss function
        self.loss_type = loss_type
        if loss_params is None:
            loss_params = {}
        if loss_type == 'focal':
            self.loss_fct = FocalLoss(**loss_params)
        elif loss_type == 'smooth_focal':
            self.loss_fct = SmoothFocalLoss(**loss_params)
        else:
            if pos_weights is not None:
                self.loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
            else:
                self.loss_fct = nn.BCEWithLogitsLoss()

    def forward(self, batch):
        x = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])[0]
        x = self.model_dropout(x)
        logits = self.label_classifier(x)

        probabilities = torch.sigmoid(logits)

        if 'labels' in batch:
            loss = self.loss_fct(logits.view(-1, self.num_labels), batch['labels'].view(-1, self.num_labels))
            return probabilities, loss
        return probabilities
