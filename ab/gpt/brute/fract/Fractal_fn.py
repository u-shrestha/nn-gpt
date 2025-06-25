import torch
import torch.nn as nn
import numpy as np

class FractalUnit(nn.Module):
    def __init__(self, in_channels, out_channels,
                 in_depth, out_depth,
                 in_width, out_width,
                 num_columns, loc_drop_prob, dropout_prob):
        super().__init__()
        self.unit = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )

    def forward(self, x):
        return self.unit(x)


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.N = prm['N']
        self.num_columns = prm['num_columns']
        self.glob_drop_ratio = 0.5
        self.loc_drop_prob = 0.15
        self.glob_num_columns = np.random.randint(0, self.num_columns, size=(1,))
        dropout = prm['dropout']

        channels = [64 * (2 ** (i if i != 4 else i - 1)) for i in range(self.N)]
        dropout_probs = [min(0.5, dropout + i * 0.1) for i in range(self.N)]

        in_channels = in_shape[1]
        in_depth = in_shape[2]
        in_width = in_shape[3]

        self.features = nn.Sequential()

        for i in range(self.N):
            out_channels = channels[i]
            out_depth = max(1, in_depth // 2)
            out_width = max(1, in_width // 2)

            self.features.add_module(f"unit{i + 1}", FractalUnit(
                in_channels=in_channels,
                out_channels=out_channels,
                in_depth=in_depth,
                out_depth=out_depth,
                in_width=in_width,
                out_width=out_width,
                num_columns=self.num_columns,
                loc_drop_prob=self.loc_drop_prob,
                dropout_prob=dropout_probs[i]
            ))

            in_channels = out_channels
            in_depth = out_depth
            in_width = out_width

        with torch.no_grad():
            dummy_input = torch.zeros(1, *in_shape[1:]).to(device)
            dummy_output = self.features(dummy_input)
            in_features = dummy_output.view(1, -1).shape[1]

        self.output = nn.Linear(in_features, out_shape[0])
        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.output(x)

    def train_setup(self, prm):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        self.train()
        for x, y in train_data:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            pred = self(x)
            loss = self.criterion(pred, y)
            loss.backward()
            self.optimizer.step()


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}
