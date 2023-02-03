import torch
import torch.nn as nn
import torch.nn.functional as F


# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#         self.l1 = nn.Linear(100, 128)
#         self.bn1 = nn.BatchNorm1d(128)
#         self.drop1 = nn.Dropout(0.5)
#         self.l2 = nn.Linear(128, 256)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.drop2 = nn.Dropout(0.5)
#         self.l3 = nn.Linear(256, 512)
#         self.bn3 = nn.BatchNorm1d(512)
#         self.drop3 = nn.Dropout(0.5)
#         self.l4 = nn.Linear(512, 1024)
#         self.bn4 = nn.BatchNorm1d(1024)
#         self.drop4 = nn.Dropout(0.5)
#         self.l5 = nn.Linear(1024, 128 * 128)

#     def forward(self, x):
#         x = F.relu(((self.l1(x))))
#         x = F.relu(((self.l2(x))))
#         x = F.relu(((self.l3(x))))
#         x = F.relu((self.drop4(self.l4(x))))
#         x = torch.sigmoid(self.l5(x))
#         return x.view(-1, 128, 128)


# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.fc1 = nn.Linear(128 * 128, 1024)
#         self.fc2 = nn.Linear(1024, 512)
#         self.fc3 = nn.Linear(512, 256)
#         self.fc4 = nn.Linear(256, 128)
#         self.fc5 = nn.Linear(128, 1)

#     def forward(self, x):
#         x = x.view(-1, 128 * 128)
#         x = nn.LeakyReLU(0.2)(self.fc1(x))
#         x = nn.LeakyReLU(0.2)(self.fc2(x))
#         x = nn.LeakyReLU(0.2)(self.fc3(x))
#         x = nn.LeakyReLU(0.2)(self.fc4(x))
#         x = nn.Sigmoid()(self.fc5(x))
#         return x


class Generator_256(nn.Module):
    def __init__(self, transpose=True):
        super(Generator_256, self).__init__()

        # Layer 1: Fully connected
        # self.fc1 = nn.Linear(100, 128 * 16 * 16)
        self.fc1 = nn.Linear(100, 128 * 32 * 32)

        if transpose == True:

            # Layer 2: Convolutional
            self.conv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)

            # Layer 3: Convolutional
            self.conv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)

            # layer 4: Convolutional
            self.conv3 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)

        else:
            self.conv1 = nn.Conv2d(128, 64, kernel_size=4, stride=2, padding=1)
            self.conv1 = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            )(self.conv1)

            self.conv2 = nn.Conv2d(64, 32, kernel_size=4, stride=2, padding=1)
            self.conv2 = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            )(self.conv2)

            self.conv3 = nn.Conv2d(32, 1, kernel_size=4, stride=2, padding=1)
            self.conv3 = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            )(self.conv3)

        # BatchNorm
        self.batchnorm256 = nn.BatchNorm2d(256)
        self.batchnorm128 = nn.BatchNorm2d(128)
        self.batchnorm64 = nn.BatchNorm2d(64)
        self.batchnorm32 = nn.BatchNorm2d(32)
        self.batchnorm3 = nn.BatchNorm2d(16)

        # Dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Layer 1
        x = self.fc1(x)
        x = x.view(-1, 128, 32, 32)
        x = self.batchnorm128(x)
        x = F.leaky_relu(x, 0.2)
        x = self.dropout(x)

        # Layer 2
        x = self.conv1(x)
        x = self.batchnorm64(x)
        x = F.leaky_relu(x, 0.2)
        x = self.dropout(x)

        # Layer 3
        x = self.conv2(x)
        x = self.batchnorm32(x)
        x = F.leaky_relu(x, 0.2)
        x = self.dropout(x)

        # Layer 4
        x = self.conv3(x)
        x = torch.tanh(x)

        return x


class Discriminator_256(nn.Module):
    def __init__(self):
        super(Discriminator_256, self).__init__()

        # Layer 1: Convolutional # input 256x256 output 128x128
        self.conv1 = nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1)
        # Layer 2: Convolutional # input 128x128 output 64x64
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
        # Layer 3: Convolutional # input 64x64 output 64x64
        self.conv3 = nn.Conv2d(32, 50, kernel_size=3, stride=1, padding=1)
        # Layer 4: Convolutional # input 64x64 output 32x32
        self.conv4 = nn.Conv2d(50, 100, kernel_size=4, stride=2, padding=1)
        # Layer 5: Convolutional # input 32x32 output 16x16
        self.conv5 = nn.Conv2d(100, 128, kernel_size=4, stride=2, padding=1)
        #  Layer 6: Convolutional input 16x16 output 8x8
        self.conv6 = nn.Conv2d(128, 200, kernel_size=4, stride=2, padding=1)
        # Layer 7: Convolutional # input 8x8 output 4x4
        self.conv7 = nn.Conv2d(200, 256, kernel_size=4, stride=2, padding=1)

        # Layer 8: Fully connected
        self.fc1 = nn.Linear(256 * 4 * 4, 300 * 32)
        self.fc2 = nn.Linear(300 * 32, 1)

        # BatchNorm
        self.batchnorm16 = nn.BatchNorm2d(16)
        self.batchnorm32 = nn.BatchNorm2d(32)
        self.batchnorm50 = nn.BatchNorm2d(50)
        # self.batchnorm64 = nn.BatchNorm2d(64)
        self.batchnorm100 = nn.BatchNorm2d(100)
        self.batchnorm128 = nn.BatchNorm2d(128)
        self.batchnorm200 = nn.BatchNorm2d(200)
        self.batchnorm256 = nn.BatchNorm2d(256)
        # self.batchnorm5 = nn.BatchNorm2d(512)

    def forward(self, x):
        # Layer 1
        x = self.conv1(x)
        x = self.batchnorm16(x)
        x = F.leaky_relu(x, 0.2)
        # Layer 2
        x = self.conv2(x)
        x = self.batchnorm32(x)
        x = F.leaky_relu(x, 0.2)
        # Layer 3
        x = self.conv3(x)
        x = self.batchnorm50(x)
        x = F.leaky_relu(x, 0.2)
        # Layer 4
        x = self.conv4(x)
        x = self.batchnorm100(x)
        x = F.leaky_relu(x, 0.2)
        # Layer 5
        x = self.conv5(x)
        x = self.batchnorm128(x)
        x = F.leaky_relu(x, 0.2)
        # Layer 6
        x = self.conv6(x)
        x = self.batchnorm200(x)
        x = F.leaky_relu(x, 0.2)
        # Layer 7
        x = self.conv7(x)
        x = self.batchnorm256(x)
        x = F.leaky_relu(x, 0.2)
        # Layer 8
        x = x.view(-1, 256 * 4 * 4)
        x = self.fc1(x)
        x = F.leaky_relu(x, 0.2)
        # Layer 9
        x = self.fc2(x)
        x = torch.sigmoid(x)

        return x


class Generator_128(nn.Module):
    def __init__(self, transpose=True):
        super(Generator_128, self).__init__()

        self.transpose = transpose

        # layer 1: Linear
        self.fc1 = nn.Linear(100, 128 * 16 * 16)

        if self.transpose == True:

            # Layer 2: Convolutional # input 16x16 output 32X32
            self.conv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)

            # Layer 3: Convolutional # input 32X32 output 32X32
            self.conv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1)

            # Layer 4: Convolutional # input 32X32 output 64x64
            self.conv3 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)

            # # Layer 4: Convolutional # input 32X32 output 64x64
            # self.conv4 = nn.ConvTranspose2d(20, 16, kernel_size=4, stride=2, padding=1)

            # Layer 6: Convolutional # input 64x64 output 128x128
            self.conv4 = nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1)

            # layer 7: Convolutional # input 64x64 output 128x128
            # self.conv5 = nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1)

        else:
            self.conv1 = nn.Conv2d(128, 64, kernel_size=4, stride=2, padding=1)
            self.conv2 = nn.Conv2d(64, 32, kernel_size=4, stride=2, padding=1)
            self.conv3 = nn.Conv2d(32, 16, kernel_size=4, stride=2, padding=1)
            self.conv4 = nn.Conv2d(16, 1, kernel_size=4, stride=2, padding=1)

            self.sample = nn.Upsample(
                scale_factor=4, mode="bilinear", align_corners=True
            )
            self.sample2 = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            )

        # BatchNorm
        self.batchnorm1D = nn.BatchNorm1d(128 * 16 * 16)

        self.batchnorm512 = nn.BatchNorm2d(512)
        self.batchnorm256 = nn.BatchNorm2d(256)
        self.batchnorm200 = nn.BatchNorm2d(200)
        self.batchnorm128 = nn.BatchNorm2d(128)
        self.batchnorm64 = nn.BatchNorm2d(64)
        self.batchnorm32 = nn.BatchNorm2d(32)
        self.batchnorm16 = nn.BatchNorm2d(16)
        self.batchnorm8 = nn.BatchNorm2d(8)

        # Dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):

        if self.transpose == True:
            # Layer 1
            x = self.fc1(x)
            x = self.batchnorm1D(x)
            x = x.view(-1, 128, 16, 16)
            x = F.leaky_relu(x, 0.2)
            x = self.dropout(x)

            # Layer 2
            x = self.conv1(x)
            x = self.batchnorm64(x)
            x = F.leaky_relu(x, 0.2)
            x = self.dropout(x)

            # Layer 3
            x = self.conv2(x)
            x = self.batchnorm32(x)
            x = F.leaky_relu(x, 0.2)
            x = self.dropout(x)

            # Layer 4
            x = self.conv3(x)
            x = self.batchnorm16(x)
            x = F.leaky_relu(x, 0.2)

            # # Layer 5
            # x = self.conv4(x)
            # x = self.batchnorm8(x)
            # x = F.leaky_relu(x, 0.2)

            # Layer 6
            x = self.conv4(x)
            x = torch.tanh(x)
            return x

        else:
            # Layer 1
            x = self.fc1(x)
            x = x.view(-1, 128, 8, 8)
            x = F.leaky_relu(x, 0.2)
            x = self.dropout(x)

            # Layer 2
            x = self.sample(x)
            x = self.conv1(x)
            x = self.batchnorm64(x)
            x = F.leaky_relu(x, 0.2)
            x = self.dropout(x)

            # Layer 3
            x = self.sample(x)
            x = self.conv2(x)
            x = self.batchnorm32(x)
            x = F.leaky_relu(x, 0.2)
            x = self.dropout(x)

            # Layer 4
            x = self.sample(x)
            x = self.conv3(x)
            x = self.batchnorm16(x)
            x = F.leaky_relu(x, 0.2)

            # Layer 5
            x = self.sample(x)
            x = self.conv4(x)
            x = torch.tanh(x)
            return x


class Discriminator_128(nn.Module):
    def __init__(self, spectral_norm=False, n_conv=5):
        super(Discriminator_128, self).__init__()
        self.sn = spectral_norm

        if self.sn == False:
            # Layer: Convolutional # input 128x128 output 128x128
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
            # Layer: Convolutional  # input 128x128 output 64x64
            self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
            # Layer: Convolutional  # input 64x64 output 64x64
            self.conv3 = nn.Conv2d(32, 50, kernel_size=3, stride=1, padding=1)
            # Layer: Convolutional  # input 64x64 output 32x32
            self.conv4 = nn.Conv2d(50, 100, kernel_size=4, stride=2, padding=1)
            # Layer: Convolutional # input 32x32 output 32x32
            self.conv5 = nn.Conv2d(100, 128, kernel_size=3, stride=1, padding=1)
            # Layer: Convolutional # input 32x32 output 16x16
            self.conv6 = nn.Conv2d(128, 200, kernel_size=4, stride=2, padding=1)
            # Layer: Convolutional  # input 16x16 output 8x8
            self.conv7 = nn.Conv2d(200, 256, kernel_size=4, stride=2, padding=1)
            # Layer: Convolutional # input 8x8 output 8x8
            self.conv8 = nn.Conv2d(256, 300, kernel_size=3, stride=1, padding=1)
            # Layer: Convolutional # input 8x8 output 4x4
            self.conv9 = nn.Conv2d(300, 400, kernel_size=4, stride=2, padding=1)
            # Layer: Convolutional # input 4x4 output 4x4
            self.conv10 = nn.Conv2d(400, 512, kernel_size=3, stride=1, padding=1)
            

            # Layer 8: Fully connected
            self.fc1 = nn.Linear(512 * 4 * 4, 1)
            # Layer 9: Fully connected
            # self.fc2 = nn.Linear(128 * 8 * 8, 1)

        else:
            # Adapted from https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
            sn_fn = torch.nn.utils.spectral_norm
            self.conv1 = sn_fn(torch.nn.Conv2d(1, 64, 3, stride=1, padding=(1, 1)))
            self.conv2 = sn_fn(torch.nn.Conv2d(64, 64, 4, stride=2, padding=(1, 1)))
            self.conv3 = sn_fn(torch.nn.Conv2d(64, 128, 3, stride=1, padding=(1, 1)))
            self.conv4 = sn_fn(torch.nn.Conv2d(128, 128, 4, stride=2, padding=(1, 1)))
            self.conv5 = sn_fn(torch.nn.Conv2d(128, 256, 3, stride=1, padding=(1, 1)))
            self.conv6 = sn_fn(torch.nn.Conv2d(256, 256, 4, stride=2, padding=(1, 1)))
            self.conv7 = sn_fn(torch.nn.Conv2d(256, 512, 3, stride=1, padding=(1, 1)))
            self.fc = sn_fn(torch.nn.Linear(16 * 16 * 512, 1))
            self.act = torch.nn.LeakyReLU(0.1)

        # BatchNorm
        self.batchnorm8 = nn.BatchNorm2d(8)
        self.batchnorm16 = nn.BatchNorm2d(16)
        self.batchnorm32 = nn.BatchNorm2d(32)
        self.batchnorm50 = nn.BatchNorm2d(50)
        self.batchnorm64 = nn.BatchNorm2d(64)
        self.batchnorm100 = nn.BatchNorm2d(100)
        self.batchnorm128 = nn.BatchNorm2d(128)
        self.batchnorm200 = nn.BatchNorm2d(200)
        self.batchnorm256 = nn.BatchNorm2d(256)
        self.batchnorm300 = nn.BatchNorm2d(300)
        self.batchnorm400 = nn.BatchNorm2d(400)
        self.batchnorm512 = nn.BatchNorm2d(512)

    def forward(self, x):
        if self.sn == False:

            # Layer 1
            x = self.conv1(x)
            x = F.leaky_relu(x, 0.2)

            # Layer 2
            x = self.conv2(x)
            x = self.batchnorm32(x)
            x = F.leaky_relu(x, 0.2)

            # Layer 3
            x = self.conv3(x)
            x = self.batchnorm50(x)
            x = F.leaky_relu(x, 0.2)

            # Layer 4
            x = self.conv4(x)
            x = self.batchnorm100(x)
            x = F.leaky_relu(x, 0.2)

            # Layer 5
            x = self.conv5(x)
            x = self.batchnorm128(x)
            x = F.leaky_relu(x, 0.2)

            # Layer 6
            x = self.conv6(x)
            x = self.batchnorm200(x)
            x = F.leaky_relu(x, 0.2)
            
            # Layer 6
            x = self.conv7(x)
            x = self.batchnorm256(x)
            x = F.leaky_relu(x, 0.2)

            # Layer 7
            x = self.conv8(x)
            x = self.batchnorm300(x)
            x = F.leaky_relu(x, 0.2)

            # Layer 8
            x = self.conv9(x)
            x = self.batchnorm400(x)
            x = F.leaky_relu(x, 0.2)
            
            x = self.conv10(x)
            x = self.batchnorm512(x)
            x = F.leaky_relu(x, 0.2)

            # Layer 8
            x = x.view(-1, 512 * 4 * 4)
            x = self.fc1(x)
            x = torch.sigmoid(x)

            return x
        else:

            m = self.act(self.conv1(x))
            m = self.act(self.conv2(m))
            m = self.act(self.conv3(m))
            m = self.act(self.conv4(m))
            m = self.act(self.conv5(m))
            m = self.act(self.conv6(m))
            m = self.act(self.conv7(m))
            m = self.fc(m.view(-1, 16 * 16 * 512))

            return torch.sigmoid(m)
