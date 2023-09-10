import torchmetrics.functional as F

def gradient_loss(y_true,y_pred):
    """ Gradient magnitude loss 
    
    Parameters:
        y_true (Tensor): target image data
        y_pred (Tensor): predicted image data
    
    Returns:
        gradient loss term
    """

    dx_true, dy_true = F.image.image_gradients(y_true)
    dx_pred, dy_pred = F.image.image_gradients(y_pred)
    return ((dx_true-dx_pred)**2).sum(dim=[1,2,3]).mean(dim=[0]) + ((dy_true-dy_pred)**2).sum(dim=[1,2,3]).mean(dim=[0])