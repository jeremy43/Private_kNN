from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
import torchvision

__all__ = ['ResNet50', 'ResNet101', 'ResNet50M','ResNet50MA2']

class ResNet50(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet50, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.classifier = nn.Linear(2048, num_classes)
        self.feat_dim = 2048 # feature dimension

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        y = self.classifier(f)
        if not self.training:
            #print('return feature')
            return f,y
        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        elif self.loss == {'cent'}:
            return y, f
        elif self.loss == {'ring'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

class ResNet101(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet101, self).__init__()
        self.loss = loss
        resnet101 = torchvision.models.resnet101(pretrained=True)
        self.base = nn.Sequential(*list(resnet101.children())[:-2])
        self.classifier = nn.Linear(2048, num_classes)
        self.feat_dim = 2048 # feature dimension

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        if not self.training:
            return f
        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        elif self.loss == {'cent'}:
            return y, f
        elif self.loss == {'ring'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

class ResNet50M(nn.Module):
    """ResNet50 + mid-level features.

    Reference:
    Yu et al. The Devil is in the Middle: Exploiting Mid-level Representations for
    Cross-Domain Instance Matching. arXiv:1711.08106.
    """
    def __init__(self, num_classes=0, loss={'xent'}, **kwargs):
        super(ResNet50M, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        base = nn.Sequential(*list(resnet50.children())[:-2])
        self.layers1 = nn.Sequential(base[0], base[1], base[2])
        self.layers2 = nn.Sequential(base[3], base[4])
        self.layers3 = base[5]
        self.layers4 = base[6]
        self.layers5a = base[7][0]
        self.layers5b = base[7][1]
        self.layers5c = base[7][2]
        self.fc_fuse = nn.Sequential(nn.Linear(4096, 1024), nn.BatchNorm1d(1024), nn.ReLU())
        self.classifier = nn.Linear(3072, num_classes)
        self.feat_dim = 3072 # feature dimension

    def forward(self, x):
        x1 = self.layers1(x)
        x2 = self.layers2(x1)
        x3 = self.layers3(x2)
        x4 = self.layers4(x3)
        x5a = self.layers5a(x4)
        x5b = self.layers5b(x5a)
        x5c = self.layers5c(x5b)

        x5a_feat = F.avg_pool2d(x5a, x5a.size()[2:]).view(x5a.size(0), x5a.size(1))
        x5b_feat = F.avg_pool2d(x5b, x5b.size()[2:]).view(x5b.size(0), x5b.size(1))
        x5c_feat = F.avg_pool2d(x5c, x5c.size()[2:]).view(x5c.size(0), x5c.size(1))

        midfeat = torch.cat((x5a_feat, x5b_feat), dim=1)
        midfeat = self.fc_fuse(midfeat)

        combofeat = torch.cat((x5c_feat, midfeat), dim=1)
        prelogits = self.classifier(combofeat)
        if not self.training:
            #print('evaluate for resnet50m', prelogits.shape)
            #return x5c_feat,prelogits

            return combofeat,prelogits

        if self.loss == {'xent'}:
            #print('prelogitss shape', prelogits.shape)
            return prelogits
        elif self.loss == {'xent', 'htri'}:
            return prelogits, combofeat
        elif self.loss == {'cent'}:
            return prelogits, combofeat
        elif self.loss == {'ring'}:
            return prelogits, combofeat
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

class ResNet50MT(nn.Module):
    """ResNet50 + mid-level features + Teachers

    ResNet50M model for teachers
    """
    def __init__(self, num_classes=0, rot_mat=0, loss={'xent'}, **kwargs):
        super(ResNet50MT, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        base = nn.Sequential(*list(resnet50.children())[:-2])
        self.layers1 = nn.Sequential(base[0], base[1], base[2])
        self.layers2 = nn.Sequential(base[3], base[4])
        self.layers3 = base[5]
        self.layers4 = base[6]
        self.layers5a = base[7][0]
        self.layers5b = base[7][1]
        self.layers5c = base[7][2]
        self.fc_fuse = nn.Sequential(nn.Linear(4096, 1024), nn.BatchNorm1d(1024), nn.ReLU())
        #self.embedding = nn.Embedding.from_pretrained(rot_mat) rot_mat.shape[1]
        self.embedding = nn.Parameter(rot_mat, requires_grad=False)
        self.classifier = nn.Linear(rot_mat.shape[1], num_classes)
        self.feat_dim = rot_mat.shape[1] # feature dimension

    def forward(self, x):
        x1 = self.layers1(x)
        x2 = self.layers2(x1)
        x3 = self.layers3(x2)
        x4 = self.layers4(x3)
        x5a = self.layers5a(x4)
        x5b = self.layers5b(x5a)
        x5c = self.layers5c(x5b)

        x5a_feat = F.avg_pool2d(x5a, x5a.size()[2:]).view(x5a.size(0), x5a.size(1))
        x5b_feat = F.avg_pool2d(x5b, x5b.size()[2:]).view(x5b.size(0), x5b.size(1))
        x5c_feat = F.avg_pool2d(x5c, x5c.size()[2:]).view(x5c.size(0), x5c.size(1))

        midfeat = torch.cat((x5a_feat, x5b_feat), dim=1)
        midfeat = self.fc_fuse(midfeat)

        combofeat = torch.cat((x5c_feat, midfeat), dim=1)
        combofeat_rot = torch.mm(combofeat, self.embedding)
        #torch.mm(torch.mm(combofeat,rot_mat),rot_mat.t())
        prelogits = self.classifier(combofeat_rot)

        if not self.training:
            return combofeat_rot,prelogits

        if self.loss == {'xent'}:
            return prelogits
        elif self.loss == {'xent', 'htri'}:
            return prelogits, combofeat_rot
        elif self.loss == {'cent'}:
            return prelogits, combofeat_rot
        elif self.loss == {'ring'}:
            return prelogits, combofeat_rot
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class ResNet50MTR(nn.Module):
    """ResNet50 + mid-level features + Teachers

    ResNet50M model for teachers
    """
    def __init__(self, num_classes=0, rot_mat=0, loss={'xent'}, **kwargs):
        super(ResNet50MTR, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        base = nn.Sequential(*list(resnet50.children())[:-2])
        self.layers1 = nn.Sequential(base[0], base[1], base[2])
        self.layers2 = nn.Sequential(base[3], base[4])
        self.layers3 = base[5]
        self.layers4 = base[6]
        self.layers5a = base[7][0]
        self.layers5b = base[7][1]
        self.layers5c = base[7][2]
        self.fc_fuse = nn.Sequential(nn.Linear(4096, 1024), nn.BatchNorm1d(1024), nn.ReLU())
        #self.embedding = nn.Embedding.from_pretrained(rot_mat) rot_mat.shape[1]
        self.embedding = nn.Parameter(torch.mm(rot_mat, rot_mat.t()), requires_grad=False)
        self.classifier = nn.Linear(3072, num_classes)
        self.feat_dim = 3072 # feature dimension

    def forward(self, x):
        x1 = self.layers1(x)
        x2 = self.layers2(x1)
        x3 = self.layers3(x2)
        x4 = self.layers4(x3)
        x5a = self.layers5a(x4)
        x5b = self.layers5b(x5a)
        x5c = self.layers5c(x5b)

        x5a_feat = F.avg_pool2d(x5a, x5a.size()[2:]).view(x5a.size(0), x5a.size(1))
        x5b_feat = F.avg_pool2d(x5b, x5b.size()[2:]).view(x5b.size(0), x5b.size(1))
        x5c_feat = F.avg_pool2d(x5c, x5c.size()[2:]).view(x5c.size(0), x5c.size(1))

        midfeat = torch.cat((x5a_feat, x5b_feat), dim=1)
        midfeat = self.fc_fuse(midfeat)

        combofeat = torch.cat((x5c_feat, midfeat), dim=1)
        combofeat_rot = torch.mm(combofeat, self.embedding)
        #torch.mm(torch.mm(combofeat,rot_mat),rot_mat.t())
        prelogits = self.classifier(combofeat_rot)

        if not self.training:
            return combofeat_rot,prelogits

        if self.loss == {'xent'}:
            return prelogits
        elif self.loss == {'xent', 'htri'}:
            return prelogits, combofeat_rot
        elif self.loss == {'cent'}:
            return prelogits, combofeat_rot
        elif self.loss == {'ring'}:
            return prelogits, combofeat_rot
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class ResNet50MA(nn.Module):
    """ResNet50 + attribute.
    Add one linear layer for attribute classification along with identity classification.
    Multi-task setting
    """
    def __init__(self, num_classesI=0, num_classesA=0, loss={'xent'}, **kwargs):
        super(ResNet50MA, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        base = nn.Sequential(*list(resnet50.children())[:-2])
        self.layers1 = nn.Sequential(base[0], base[1], base[2])
        self.layers2 = nn.Sequential(base[3], base[4])
        self.layers3 = base[5]
        self.layers4 = base[6]
        self.layers5a = base[7][0]
        self.layers5b = base[7][1]
        self.layers5c = base[7][2]
        self.fc_fuse = nn.Sequential(nn.Linear(4096, 1024), nn.BatchNorm1d(1024), nn.ReLU())
        self.classifierI = nn.Linear(3072, num_classesI)
        self.classifierA = nn.Linear(3072, num_classesA)
        self.feat_dim = 3072 # feature dimension

    def forward(self, x):
        x1 = self.layers1(x)
        x2 = self.layers2(x1)
        x3 = self.layers3(x2)
        x4 = self.layers4(x3)
        x5a = self.layers5a(x4)
        x5b = self.layers5b(x5a)
        x5c = self.layers5c(x5b)

        x5a_feat = F.avg_pool2d(x5a, x5a.size()[2:]).view(x5a.size(0), x5a.size(1))
        x5b_feat = F.avg_pool2d(x5b, x5b.size()[2:]).view(x5b.size(0), x5b.size(1))
        x5c_feat = F.avg_pool2d(x5c, x5c.size()[2:]).view(x5c.size(0), x5c.size(1))

        midfeat = torch.cat((x5a_feat, x5b_feat), dim=1)
        midfeat = self.fc_fuse(midfeat)

        combofeat = torch.cat((x5c_feat, midfeat), dim=1)



        prelogitsA = self.classifierA(combofeat)
        if not self.training:
            return combofeat,prelogitsA

        prelogitsI = self.classifierI(combofeat)
        
        if self.loss == {'xent'}:
            return prelogitsI
        elif self.loss == {'xent','att'}:
            return prelogitsI, prelogitsA
        elif self.loss == {'xent', 'htri'}:
            return prelogits, combofeat
        elif self.loss == {'cent'}:
            return prelogits, combofeat
        elif self.loss == {'ring'}:
            return prelogits, combofeat
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

class ResNet50MA1(nn.Module):
    """ResNet50 + attribute.
    Add one linear layer for attribute classification along with identity classification.
    Multi-task setting
    """
    def __init__(self, num_classesI=0, num_classesA=0, loss={'xent'}, **kwargs):
        super(ResNet50MA1, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        base = nn.Sequential(*list(resnet50.children())[:-2])
        self.layers1 = nn.Sequential(base[0], base[1], base[2])
        self.layers2 = nn.Sequential(base[3], base[4])
        self.layers3 = base[5]
        self.layers4 = base[6]
        self.layers5a = base[7][0]
        self.layers5b = base[7][1]
        self.layers5c = base[7][2]
        self.fc_fuse = nn.Sequential(nn.Linear(4096, 1024), nn.BatchNorm1d(1024), nn.ReLU())
        self.fc_i1 = nn.Sequential(nn.Linear(3072,1024))
        self.fc_a1 = nn.Sequential(nn.Linear(1024,512))
        self.fc_a2 = nn.Sequential(nn.Linear(512,256))
        self.classifierI = nn.Linear(1024, num_classesI)
        self.classifierA = nn.Linear(256, num_classesA)
        self.feat_dim = 3072 # feature dimension

    def forward(self, x):
        x1 = self.layers1(x)
        x2 = self.layers2(x1)
        x3 = self.layers3(x2)
        x4 = self.layers4(x3)
        x5a = self.layers5a(x4)
        x5b = self.layers5b(x5a)
        x5c = self.layers5c(x5b)

        x5a_feat = F.avg_pool2d(x5a, x5a.size()[2:]).view(x5a.size(0), x5a.size(1))
        x5b_feat = F.avg_pool2d(x5b, x5b.size()[2:]).view(x5b.size(0), x5b.size(1))
        x5c_feat = F.avg_pool2d(x5c, x5c.size()[2:]).view(x5c.size(0), x5c.size(1))

        midfeat = torch.cat((x5a_feat, x5b_feat), dim=1)
        midfeat = self.fc_fuse(midfeat)

        combofeat = torch.cat((x5c_feat, midfeat), dim=1)
        feat_i1 = self.fc_i1(combofeat)
        feat_a1 = self.fc_a1(feat_i1)
        feat_a2 = self.fc_a2(feat_a1)
        prelogitsA = self.classifierA(feat_a2)
        if not self.training:
            return feat_i1,prelogitsA
        prelogitsI = self.classifierI(feat_i1)

        
        if self.loss == {'xent'}:
            return prelogitsI
        elif self.loss == {'xent','att'}:
            return prelogitsI, prelogitsA
        elif self.loss == {'xent', 'htri'}:
            return prelogits, feat_i1
        elif self.loss == {'cent'}:
            return prelogits, feat_i1
        elif self.loss == {'ring'}:
            return prelogits, feat_i1
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class ResNet50MA2(nn.Module):
    """ResNet50 + attribute.
    Add one linear layer for attribute classification along with identity classification.
    Multi-task setting
    """
    def __init__(self, num_classesI=0, num_classesA=0, loss={'xent'}, **kwargs):
        super(ResNet50MA2, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        base = nn.Sequential(*list(resnet50.children())[:-2])
        self.layers1 = nn.Sequential(base[0], base[1], base[2])
        self.layers2 = nn.Sequential(base[3], base[4])
        self.layers3 = base[5]
        self.layers4 = base[6]
        self.layers5a = base[7][0]
        self.layers5b = base[7][1]
        self.layers5c = base[7][2]
        self.fc_fuse = nn.Sequential(nn.Linear(4096, 1024), nn.BatchNorm1d(1024), nn.ReLU())
        self.fc_i1 = nn.Sequential(nn.Linear(3072,512))
        self.fc_a1 = nn.Sequential(nn.Linear(3072,128))
        self.classifierI = nn.Linear(512, num_classesI)
        self.classifierA = nn.Linear(128, num_classesA)
        self.feat_dim = 3072 # feature dimension

    def forward(self, x):
        x1 = self.layers1(x)
        x2 = self.layers2(x1)
        x3 = self.layers3(x2)
        x4 = self.layers4(x3)
        x5a = self.layers5a(x4)
        x5b = self.layers5b(x5a)
        x5c = self.layers5c(x5b)

        x5a_feat = F.avg_pool2d(x5a, x5a.size()[2:]).view(x5a.size(0), x5a.size(1))
        x5b_feat = F.avg_pool2d(x5b, x5b.size()[2:]).view(x5b.size(0), x5b.size(1))
        x5c_feat = F.avg_pool2d(x5c, x5c.size()[2:]).view(x5c.size(0), x5c.size(1))

        midfeat = torch.cat((x5a_feat, x5b_feat), dim=1)
        midfeat = self.fc_fuse(midfeat)

        combofeat = torch.cat((x5c_feat, midfeat), dim=1)
        feat_i1 = self.fc_i1(combofeat)
        feat_a1 = self.fc_a1(combofeat)
        prelogitsA = self.classifierA(feat_a1)

        if not self.training:
            return feat_i1,prelogitsA
        prelogitsI = self.classifierI(feat_i1)
        if self.loss == {'xent'}:
            #print('ok with resnet', prelogitsA.shape)
            return prelogitsA
            #return prelogitsI
        elif self.loss == {'xent','att'}:
            return prelogitsI, prelogitsA
        elif self.loss == {'xent', 'htri'}:
            return prelogits, feat_i1
        elif self.loss == {'cent'}:
            return prelogits, feat_i1
        elif self.loss == {'ring'}:
            return prelogits, feat_i1
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))
