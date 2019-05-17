import torch.nn as nn
import torch.nn.functional as F


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()

        self.dense1 = nn.Linear(410 * 308, 256 * 13)
        self.drop5 = nn.Dropout(0.2)

        self.dense2 = nn.Linear(256 * 13, 256 * 13)
        self.drop6 = nn.Dropout(0.2)

        self.dense3 = nn.Linear(256 * 13, 2)

    def forward(self, x):
        # flatten layer
        x = x.view(x.size(0), -1)

        x = F.elu(self.dense1(x))
        x = self.drop5(x)

        x = F.relu(self.dense2(x))
        x = self.drop6(x)

        x = self.dense3(x)

        return x
