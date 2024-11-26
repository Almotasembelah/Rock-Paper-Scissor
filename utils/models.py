import torch.nn as nn
import torch.nn.functional as F
import torch

class CNNModel(nn.Module):
  def __init__(self, in_channels, out_channels, n_filters):
    super(CNNModel, self).__init__()

    self.conv1 = nn.Conv2d(in_channels=in_channels,
                           out_channels=n_filters,
                           kernel_size=3,
                           stride=1,
                           padding=1)

    self.conv2 = nn.Conv2d(in_channels=n_filters,
                           out_channels=n_filters//4,
                           kernel_size=5,
                           stride=1,
                           padding=1)

    self.conv3 = nn.Conv2d(in_channels=n_filters//4,
                           out_channels=n_filters//8,
                           kernel_size=5,
                           stride=1)

    self.fc1 = nn.Linear((n_filters//8)*27*27, 1200)
    self.fc2 = nn.Linear(1200, 600)
    self.fc3 = nn.Linear(600, out_channels)

    self.relu = nn.ReLU()
    self.maxpool = nn.MaxPool2d(kernel_size=2)
    self.avgpool = nn.AvgPool2d(kernel_size=2)
    self.dropout2d = nn.Dropout2d(0.4)
    self.dropout = nn.Dropout(0.5)

  def featurizer(self, x):
    #3@128*128 -> n_filters@128*128 -> n_filters@64*64
    x = self.conv1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    # n_filters@64*64 -> n_filters@62*62 -> n_filters@31*31
    x = self.conv2(x)
    x = self.relu(x)
    x = self.avgpool(x)
    x = self.dropout2d(x)

    # n_filters@31X31 -> n_filters@27X27
    x = self.conv3(x)
    x = self.relu(x)
    x = self.dropout2d(x)

    x = torch.flatten(x, 1)

    return x

  def classifier(self, x):
    x = self.fc1(x)
    x = self.relu(x)
    x = self.dropout(x)

    x = self.fc2(x)
    x = self.relu(x)

    x = self.fc3(x)
    return x

  def forward(self, x):
    x = self.featurizer(x)
    x = self.classifier(x)
    return x


class CNNModelV01(nn.Module):
  def __init__(self, in_channels, out_channels, n_filters):
    super(CNNModelV01, self).__init__()

    self.conv1 = nn.Conv2d(in_channels=3,
                            out_channels=n_filters,
                            kernel_size=3,
                            stride=1,
                            padding=1)

    self.conv2 = nn.Conv2d(in_channels=n_filters,
                            out_channels=2*n_filters,
                            kernel_size=3,
                            stride=1,
                            padding=1)

    self.conv3 = nn.Conv2d(in_channels=n_filters*2,
                            out_channels=n_filters*2,
                            kernel_size=3,
                            stride=1,
                            padding=1)

    self.conv3 = nn.Conv2d(in_channels=2*n_filters,
                            out_channels=n_filters*2,
                            kernel_size=3,
                            stride=1,
                          padding=1)

    self.conv4 = nn.Conv2d(in_channels=n_filters*2,
                            out_channels=n_filters*2,
                            kernel_size=3,
                            stride=1,
                            padding=1)


    self.fc1 = nn.Linear((n_filters*2)*8*8, 1500)

    self.fc2 = nn.Linear(1500, 500)

    self.fc3 = nn.Linear(500, out_channels)

    self.relu = nn.ReLU()
    self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    self.dropout2d_1 = nn.Dropout2d(0.3)
    self.dropout2d_2 = nn.Dropout2d(0.4)
    self.dropout2d_3 = nn.Dropout2d(0.5)
    self.dropout = nn.Dropout(0.5)
    self.batchnorm = nn.BatchNorm2d(n_filters)

  def featurizer(self, x):

    #3@64X64 -> n_filters@64X64
    x = self.conv1(x)
    x = self.relu(x)
#     x = self.batchnorm(x)
#     x = self.maxpool(x)
    x = self.dropout2d_1(x)

    #n_filters@64X64 -> 2*n_filters@64X64 -> 2*n_filters@32X32
    x = self.conv2(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.dropout2d_1(x)
#     ----------------------------------------------------------------
    #2*n_filters@32X32 -> 2*n_filters@32X32-> 2*n_filters@16X16
    x = self.conv3(x)
    x = self.relu(x)
#     g = self.batchnorm(g)
    x = self.maxpool(x)

    x = self.dropout2d_2(x)

    #2*n_filters@16X16 -> 2*n_filters@8X8
    x = self.conv4(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = torch.flatten(x, 1)
    return x

  def classifier(self, x):
    x = self.fc1(x)
    x = self.relu(x)
    x = self.dropout(x)
    x = self.fc2(x)
    x = self.relu(x)
    x = self.dropout(x)
    x = self.fc3(x)
    return x

  def forward(self, x):
    x = self.featurizer(x)
    x = self.classifier(x)
    return x



class CNNModelV02(nn.Module):
    def __init__(self, in_channels, out_channels, n_filters):
        super(CNNModelV02, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=n_filters,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        
        self.conv1_1 = nn.Conv2d(in_channels=n_filters,
                               out_channels=n_filters,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        
        self.conv2 = nn.Conv2d(in_channels=1,
                               out_channels=n_filters,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        
        self.conv2_1 = nn.Conv2d(in_channels=n_filters,
                               out_channels=n_filters,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        
        self.conv3_1 = nn.Conv2d(in_channels=1,
                               out_channels=n_filters,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        
        self.conv3_2 = nn.Conv2d(in_channels=n_filters,
                               out_channels=n_filters,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        
        self.conv4 = nn.Conv2d(in_channels=n_filters,
                              out_channels=4*n_filters,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        
        self.fc1 = nn.Linear((n_filters)*16*16, 1500)
        self.fc1_1 = nn.Linear(1500, 500)
        
        self.fc2 = nn.Linear((n_filters)*16*16, 1500)
        self.fc2_1 = nn.Linear(1500, 500)
        
        self.fc3 = nn.Linear((n_filters)*16*16, 1500)
        self.fc3_1 = nn.Linear(1500, 500)

        self.fc4 = nn.Linear((n_filters*4)*16*16, 1500)
        self.fc4_1 = nn.Linear(1500, 500)
        
        self.fc10 = nn.Linear(500, out_channels)
        
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.dropout2d_l = nn.Dropout2d(0.4)
        self.dropout2d_u = nn.Dropout2d(0.3)
        self.dropout2d_v = nn.Dropout2d(0.4)
        self.dropout = nn.Dropout(0.5)
        self.batchnorm = nn.BatchNorm2d(n_filters)
    
    def featurizer(self, x):
        l = x[:, :1, :, :]
        u = x[:, 1:2, :, :]
        v = x[:, 2:3, :, :]
        
        #1@64X64 -> n_filters@64X64 -> n_filters@32X32
        l = self.conv1(l)
        l = self.relu(l)
        # l = self.batchnorm(l)
        l = self.maxpool(l)
        l = self.dropout2d_l(l)
        
        #n_filters@32X32 -> n_filters@16X16
        l1 = self.conv1_1(l)
        l1 = self.relu(l1)
        l1 = self.maxpool(l1)

        # l = l + l1
        
        #     ----------------------------------------------------------------
        #1@64X64 -> n_filters@64X64-> n_filters@32X32
        u = self.conv2(u)
        u = self.relu(u)
        # u = self.batchnorm(u)
        u = self.maxpool(u)
        
        u = self.dropout2d_u(u)
        
        #2*n_filters@32X32 -> n_filters@16X16
        u1 = self.conv2_1(u)
        u1 = self.relu(u1)
        u1 = self.maxpool(u1)
        # u = u + u1
        #     ---------------------------------------------------------------
        #1@64X64 -> n_filters@64X64 -> n_filters@32X32
        v = self.conv3_1(v)
        v = self.relu(v)
        # v = self.batchnorm(v)
        v = self.maxpool(v)
        v = self.dropout2d_v(v)
        
        #n_filters@32X32 -> n_filters@16X16
        v1 = self.conv3_2(v)
        v1 = self.relu(v1)
        v1 = self.maxpool(v1)
        # v = v + v1

        #n_filters@32X32
        luv = l + u + v

        # n_filters@15X15 -> 4*n_filters@7X7
        luv = self.conv4(luv)
        luv = self.maxpool(luv)

        l = torch.flatten(l1, 1)
        v = torch.flatten(v1, 1)
        u = torch.flatten(u1, 1)
        luv = torch.flatten(luv, 1)
        return l, u, v, luv

    def classifier(self, l, u, v, luv):
        luv = self.fc4(luv)
        luv = self.relu(luv)
        luv = self.dropout(luv)
        luv = self.fc4_1(luv)

        l = self.fc1(l)
        l = self.relu(l)
        l = self.dropout(l)
        l = self.fc1_1(l)
        l = self.relu(l)
        
        u = self.fc2(u)
        u = self.relu(u)
        u = self.dropout(u)
        u = self.fc2_1(u)
        u = self.relu(u)
        
        v = self.fc3(v)
        v = self.relu(v)
        v = self.dropout(v)
        v = self.fc3_1(v)
        v = self.relu(v)
        
        z = torch.concat([l + u + v + luv], 1)

        z = self.fc10(z)
        return z

    def forward(self, x):
        l, u, v, luv = self.featurizer(x)
        z = self.classifier(l, u, v, luv)
        return z