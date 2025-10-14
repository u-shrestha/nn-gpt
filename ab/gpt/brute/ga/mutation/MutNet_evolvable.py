#MutNet_evolvable.py:
import torch
import torch.nn as nn
import random

SEARCH_SPACE = {
    'conv1_filters': [32, 64, 96],
    'conv1_kernel': [7, 9, 11],
    'conv1_stride': [3, 4],
    'conv2_filters': [128, 192, 256],
    'conv2_kernel': [3, 5],
    'conv3_filters': [256, 384, 440],
    'conv4_filters': [256, 384],
    'conv5_filters': [192, 256],
    'fc1_neurons': [2048, 3072, 4096],
    'fc2_neurons': [2048, 3072, 4096],
    'lr': [0.001, 0.005, 0.01],
    'momentum': [0.85, 0.9, 0.95],
    'dropout': [0.4, 0.5, 0.6],
    # Binary inclusion flags (used at compile time only)
    'include_conv1': [0, 1],
    'include_conv2': [0, 1],
    'include_conv3': [0, 1],
    'include_conv4': [0, 1],
    'include_conv5': [0, 1],
    # Pooling types (compile-time only)
    'pooling_type1': ['MaxPool2d', 'AvgPool2d', 'AdaptiveMaxPool2d'],
    'pooling_type2': ['MaxPool2d', 'AvgPool2d', 'AdaptiveMaxPool2d'],
    'pooling_type3': ['MaxPool2d', 'AvgPool2d', 'AdaptiveMaxPool2d'],
    'activation_type': ['ReLU', 'LeakyReLU', 'ELU'],
    # NEW: BatchNorm toggle
    'use_batchnorm': [0, 1],
}

def create_random_chromosome():
    chromosome = {}
    for key, values in SEARCH_SPACE.items():
        chromosome[key] = random.choice(values)
    # Ensure at least one conv layer is included
    if not any(chromosome.get(f'include_conv{i}', 0) for i in range(1, 6)):
        chromosome['include_conv1'] = 1
    return chromosome

def supported_hyperparameters():
    # ONLY train-time hyperparameters
    return {'lr', 'momentum', 'dropout'}

def get_activation(activation_type):
    if activation_type == 'ReLU':
        return nn.ReLU(inplace=True)
    elif activation_type == 'LeakyReLU':
        return nn.LeakyReLU(inplace=True)
    elif activation_type == 'ELU':
        return nn.ELU(inplace=True)
    else:
        return nn.ReLU(inplace=True)

def get_activation_code(activation_type):
    if activation_type == 'ReLU':
        return "nn.ReLU(inplace=True)"
    elif activation_type == 'LeakyReLU':
        return "nn.LeakyReLU(inplace=True)"
    elif activation_type == 'ELU':
        return "nn.ELU(inplace=True)"
    else:
        return "nn.ReLU(inplace=True)"

def get_pooling_code(pooling_type, layer_idx):
    if pooling_type == 'MaxPool2d':
        return "nn.MaxPool2d(kernel_size=2, stride=2)"
    elif pooling_type == 'AvgPool2d':
        return "nn.AvgPool2d(kernel_size=2, stride=2)"
    elif pooling_type == 'AdaptiveMaxPool2d':
        return "nn.AdaptiveMaxPool2d(output_size=(6, 6))"
    else:
        return "nn.MaxPool2d(kernel_size=2, stride=2)"

def generate_model_code_string(chromosome: dict, in_shape: tuple = (3, 32, 32), out_shape: tuple = (10,)) -> str:
    lines = [
        "import torch",
        "import torch.nn as nn",
        "",
        "",
        "def supported_hyperparameters():",
        "    return {'lr', 'momentum', 'dropout'}",
        "",
        "",
        "class Net(nn.Module):",
        "    def train_setup(self, prm):",
        "        self.to(self.device)",
        "        self.criteria = (nn.CrossEntropyLoss().to(self.device),)",
        "        self.optimizer = torch.optim.SGD(",
        "            self.parameters(),",
        "            lr=prm['lr'],",
        "            momentum=prm['momentum']",
        "        )",
        "",
        "    def learn(self, train_data):",
        "        self.train()",
        "        for inputs, labels in train_data:",
        "            inputs, labels = inputs.to(self.device), labels.to(self.device)",
        "            self.optimizer.zero_grad()",
        "            outputs = self(inputs)",
        "            loss = self.criteria[0](outputs, labels)",
        "            loss.backward()",
        "            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)",
        "            self.optimizer.step()",
        "",
        "    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:",
        "        super().__init__()",
        "        self.device = device",
        "        layers = []",
        f"        in_channels = {in_shape[0]}",
    ]

    activation_code = get_activation_code(chromosome['activation_type'])
    use_bn = chromosome.get('use_batchnorm', 0)

    # Layer 1
    if chromosome['include_conv1']:
        lines += [
            f"        layers += [",
            f"            nn.Conv2d(in_channels, {chromosome['conv1_filters']}, kernel_size={chromosome['conv1_kernel']}, stride={chromosome['conv1_stride']}, padding=2),",
        ]
        if use_bn:
            lines.append(f"            nn.BatchNorm2d({chromosome['conv1_filters']}),")
        lines += [
            f"            {activation_code},",
            f"        ]",
            f"        layers.append({get_pooling_code(chromosome['pooling_type1'], 1)})",
            f"        in_channels = {chromosome['conv1_filters']}",
        ]

    # Layer 2
    if chromosome['include_conv2']:
        lines += [
            f"        layers += [",
            f"            nn.Conv2d(in_channels, {chromosome['conv2_filters']}, kernel_size={chromosome['conv2_kernel']}, padding=2),",
        ]
        if use_bn:
            lines.append(f"            nn.BatchNorm2d({chromosome['conv2_filters']}),")
        lines += [
            f"            {activation_code},",
            f"        ]",
            f"        layers.append({get_pooling_code(chromosome['pooling_type2'], 2)})",
            f"        in_channels = {chromosome['conv2_filters']}",
        ]

    # Layer 3
    if chromosome['include_conv3']:
        lines += [
            f"        layers += [",
            f"            nn.Conv2d(in_channels, {chromosome['conv3_filters']}, kernel_size=3, padding=1),",
        ]
        if use_bn:
            lines.append(f"            nn.BatchNorm2d({chromosome['conv3_filters']}),")
        lines += [
            f"            {activation_code},",
            f"        ]",
            f"        in_channels = {chromosome['conv3_filters']}",
        ]

    # Layer 4
    if chromosome['include_conv4']:
        lines += [
            f"        layers += [",
            f"            nn.Conv2d(in_channels, {chromosome['conv4_filters']}, kernel_size=3, padding=1),",
        ]
        if use_bn:
            lines.append(f"            nn.BatchNorm2d({chromosome['conv4_filters']}),")
        lines += [
            f"            {activation_code},",
            f"        ]",
            f"        in_channels = {chromosome['conv4_filters']}",
        ]

    # Layer 5
    if chromosome['include_conv5']:
        lines += [
            f"        layers += [",
            f"            nn.Conv2d(in_channels, {chromosome['conv5_filters']}, kernel_size=3, padding=1),",
        ]
        if use_bn:
            lines.append(f"            nn.BatchNorm2d({chromosome['conv5_filters']}),")
        lines += [
            f"            {activation_code},",
            f"        ]",
            f"        layers.append({get_pooling_code(chromosome['pooling_type3'], 3)})",
            f"        in_channels = {chromosome['conv5_filters']}",
        ]

    # Fallback (should not occur due to guard in create_random_chromosome)
    if not any(chromosome.get(f'include_conv{i}', 0) for i in range(1, 6)):
        lines += [
            "        layers = [",
            "            nn.AdaptiveAvgPool2d((6, 6)),",
            "            nn.Flatten()",
            "        ]",
            f"        in_channels = {in_shape[0]}",
        ]

    lines += [
        "        self.features = nn.Sequential(*layers)",
        "        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))",
        f"        classifier_input_features = in_channels * 6 * 6",
        "        self.classifier = nn.Sequential(",
        f"            nn.Dropout(p=prm['dropout']),",
        f"            nn.Linear(classifier_input_features, {chromosome['fc1_neurons']}),",
        f"            {activation_code},",
        f"            nn.Dropout(p=prm['dropout']),",
        f"            nn.Linear({chromosome['fc1_neurons']}, {chromosome['fc2_neurons']}),",
        f"            {activation_code},",
        f"            nn.Linear({chromosome['fc2_neurons']}, out_shape[0]),",
        "        )",
        "",
        "    def forward(self, x: torch.Tensor) -> torch.Tensor:",
        "        x = self.features(x)",
        "        x = self.avgpool(x)",
        "        x = torch.flatten(x, 1)",
        "        x = self.classifier(x)",
        "        return x",
        "",
        f"# Chromosome used to generate this model:",
        f"# {chromosome}",
    ]

    return "\n".join(lines)

# Runtime model (for GA evaluation only)
class Net(nn.Module):
    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'],
            momentum=prm['momentum']
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def __init__(self, in_shape: tuple, out_shape: tuple, chromosome: dict, device: torch.device):
        super().__init__()
        self.device = device
        if not any(chromosome.get(f'include_conv{i}', 0) for i in range(1, 6)):
            chromosome['include_conv1'] = 1

        layers = []
        in_channels = in_shape[0]
        activation_fn = get_activation(chromosome['activation_type'])
        use_bn = chromosome.get('use_batchnorm', 0)

        if chromosome['include_conv1']:
            layers.append(nn.Conv2d(in_channels, chromosome['conv1_filters'],
                                    kernel_size=chromosome['conv1_kernel'],
                                    stride=chromosome['conv1_stride'], padding=2))
            if use_bn:
                layers.append(nn.BatchNorm2d(chromosome['conv1_filters']))
            layers.append(activation_fn)
            if chromosome['pooling_type1'] == 'AdaptiveMaxPool2d':
                layers.append(nn.AdaptiveMaxPool2d(output_size=(6, 6)))
            else:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2) if chromosome['pooling_type1'] == 'MaxPool2d'
                              else nn.AvgPool2d(kernel_size=2, stride=2))
            in_channels = chromosome['conv1_filters']

        if chromosome['include_conv2']:
            layers.append(nn.Conv2d(in_channels, chromosome['conv2_filters'],
                                    kernel_size=chromosome['conv2_kernel'], padding=2))
            if use_bn:
                layers.append(nn.BatchNorm2d(chromosome['conv2_filters']))
            layers.append(activation_fn)
            if chromosome['pooling_type2'] == 'AdaptiveMaxPool2d':
                layers.append(nn.AdaptiveMaxPool2d(output_size=(6, 6)))
            else:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2) if chromosome['pooling_type2'] == 'MaxPool2d'
                              else nn.AvgPool2d(kernel_size=2, stride=2))
            in_channels = chromosome['conv2_filters']

        if chromosome['include_conv3']:
            layers.append(nn.Conv2d(in_channels, chromosome['conv3_filters'],
                                    kernel_size=3, padding=1))
            if use_bn:
                layers.append(nn.BatchNorm2d(chromosome['conv3_filters']))
            layers.append(activation_fn)
            in_channels = chromosome['conv3_filters']

        if chromosome['include_conv4']:
            layers.append(nn.Conv2d(in_channels, chromosome['conv4_filters'],
                                    kernel_size=3, padding=1))
            if use_bn:
                layers.append(nn.BatchNorm2d(chromosome['conv4_filters']))
            layers.append(activation_fn)
            in_channels = chromosome['conv4_filters']

        if chromosome['include_conv5']:
            layers.append(nn.Conv2d(in_channels, chromosome['conv5_filters'],
                                    kernel_size=3, padding=1))
            if use_bn:
                layers.append(nn.BatchNorm2d(chromosome['conv5_filters']))
            layers.append(activation_fn)
            if chromosome['pooling_type3'] == 'AdaptiveMaxPool2d':
                layers.append(nn.AdaptiveMaxPool2d(output_size=(6, 6)))
            else:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2) if chromosome['pooling_type3'] == 'MaxPool2d'
                              else nn.AvgPool2d(kernel_size=2, stride=2))
            in_channels = chromosome['conv5_filters']

        if not layers:
            layers = [nn.AdaptiveAvgPool2d((6, 6)), nn.Flatten()]
            in_channels = in_shape[0]

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(p=chromosome['dropout']),
            nn.Linear(in_channels * 6 * 6, chromosome['fc1_neurons']),
            activation_fn,
            nn.Dropout(p=chromosome['dropout']),
            nn.Linear(chromosome['fc1_neurons'], chromosome['fc2_neurons']),
            activation_fn,
            nn.Linear(chromosome['fc2_neurons'], out_shape[0]),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x