# cornell eaval
# python3 evaluate.py --network /home/ericlab/gr_convnet/robotic-grasping/logs/allpwlayer_coernell_Grconv/epoch_20_iou_0.98 --dataset cornell --dataset-path /home/ericlab/cornell_grasp --iou-eval
# cornel train 
# python3 train_network.py --dataset cornell --dataset-path /home/ericlab/cornell_grasp --description training_cornell
# python3 train_network.py --dataset jacquard --dataset-path /home/ericlab/jacquard_dataset --description training_jacquard --use-dropout 0 --input-size 300
import torch.nn as nn
import torch.nn.functional as F


class GraspModel(nn.Module):
    """
    An abstract model for grasp network in a common format.
    """

    def __init__(self):
        super(GraspModel, self).__init__()

    def forward(self, x_in):
        raise NotImplementedError()

    def compute_loss(self, xc, yc):
        y_pos, y_cos, y_sin, y_width = yc
        pos_pred, cos_pred, sin_pred, width_pred = self(xc)

        p_loss = F.smooth_l1_loss(pos_pred, y_pos)
        cos_loss = F.smooth_l1_loss(cos_pred, y_cos)
        sin_loss = F.smooth_l1_loss(sin_pred, y_sin)
        width_loss = F.smooth_l1_loss(width_pred, y_width)

        return {
            'loss': p_loss + cos_loss + sin_loss + width_loss,
            'losses': {
                'p_loss': p_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss
            },
            'pred': {
                'pos': pos_pred,
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred
            }
        }

    def predict(self, xc):
        pos_pred, cos_pred, sin_pred, width_pred = self(xc)
        return {
            'pos': pos_pred,
            'cos': cos_pred,
            'sin': sin_pred,
            'width': width_pred
        }

class ResidualBlock(nn.Module):
    """
    A residual block with dropout option
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x_in):
        x = self.bn1(self.conv1(x_in))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        return x + x_in

class ResidualBlockMish(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResidualBlockMish, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act = nn.Mish()
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x_in):
        x = self.bn1(self.conv1(x_in))
        x = self.act(x)
        x = self.bn2(self.conv2(x))
        return x + x_in

    
class ResidualBlock_sepa(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResidualBlock_sepa, self).__init__()
        
        self.pw1 = nn.Conv2d(in_channels, 64, 1,1)
        self.conv1 = nn.Conv2d(64, 64, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.act = nn.Mish(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pw2 = nn.Conv2d(64,out_channels,1,1)

    def forward(self, x_in):
        x = self.pw1(x_in)
        x = self.bn1(self.conv1(x))
        x = self.act(x)
        x = self.bn2(self.conv2(x))
        x = self.pw2(x)

        return x + x_in


# class SELayer(nn.Module):
#     def __init__(self, inp, oup, reduction=4):
#         super(SELayer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#                 nn.Linear(oup, _make_divisible(inp // reduction, 8)),
#                 nn.Mish(inplace=True),
#                 nn.Linear(_make_divisible(inp // reduction, 8), oup),
#                 nn.Sigmoid()
#         )

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y


# def conv_3x3_bn(inp, oup, stride):

#     return nn.Sequential(
#         nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
#         nn.BatchNorm2d(oup),
#         nn.Mish()
#     )

# def conv_1x1_bn(inp, oup):
#     return nn.Sequential(
#         nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
#         nn.BatchNorm2d(oup),
#         ()
#     )

# class SEResidualBlock(nn.Module):

#     def __init__(self, in_channels, out_channels, kernel_size=3):
#         super(ResidualBlock2, self).__init__()
#         self.pw1 = nn.Conv2d(in_channels, 64, 1)

#         self.conv1 = nn.Conv2d(64, 64, kernel_size, padding=1)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.act = nn.Mish()
#         self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
#         self.bn2 = nn.BatchNorm2d(in_channels)
#         self.pw2 = nn.Conv2d(64,128,1)

#     def forward(self, x_in):
#         x = self.pw1(x_in)
#         x = self.bn1(self.conv1(x))
#         x = self.act(x)
#         x = self.bn2(self.conv2(x))
#         x = self.pw2(x)

#         return x + x_in
    
# class SEResidualBlock2(nn.Module):

#     def __init__(self, in_channels, out_channels, kernel_size=3):
#         super(ResidualBlock2, self).__init__()
#         self.pw1 = nn.Conv2d(in_channels, 64, 1)

#         self.conv1 = nn.Conv2d(64, 64, kernel_size, padding=1)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.act = nn.Mish()
#         self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
#         self.bn2 = nn.BatchNorm2d(in_channels)
#         self.pw2 = nn.Conv2d(64,128,1)

#     def forward(self, x_in):
#         x = self.pw1(x_in)
#         x = self.bn1(self.conv1(x))
#         x = self.act(x)
#         x = self.bn2(self.conv2(x))
#         x = self.pw2(x)

#         return x + x_in





