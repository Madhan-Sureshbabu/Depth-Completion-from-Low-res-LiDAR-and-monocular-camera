import os
import torch

def save_checkpoint(state,epoch,output_dir) :
	checkpoint_filename = os.path.join(output_dir,'checkpoint-'+str(epoch)+'.pth.tar')
	torch.save(state,checkpoint_filename)

class MaskedMSELoss(torch.nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = (diff**2).mean()
        return self.loss

