
import torch.nn as nn

class DeepQNetwork(nn.Module):
    def __init__(self):
        super(DeepQNetwork, self).__init__()

        # Input: 17 Features (16 Lidar Rays + 1 Velocity)
        # Output: 2 Actions (0: Do Nothing, 1: Flap)
        self.fc = nn.Sequential(
            nn.Linear(17, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2)
        )
        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        output = self.fc(input)
        return output