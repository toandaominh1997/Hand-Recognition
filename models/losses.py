import torch.nn as nn


class ArcLoss(nn.Module):
    def __init__(self, margin, p):
        super(ArcLoss, self).__init__()
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=p)

    def forward(self, inputs, target):
        anchor = inputs['anchor']
        positive = inputs['positive']
        negative = inputs['negative']
        return self.triplet_loss(anchor, positive, negative) 
