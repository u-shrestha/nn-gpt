from ab.gpt.brute.ga.mutation.MutNet_evolvable import generate_model_code_string

# Define chromosome that mimics AlexNet
alexnet_chromosome = {
    'conv1_filters': 64,   # AlexNet often 96, but 64 is in search space usually (standard is 64 or 96)
    'conv1_kernel': 11,
    'conv1_stride': 4,
    'conv2_filters': 192,
    'conv2_kernel': 5,
    'conv3_filters': 384,
    'conv4_filters': 256, 
    'conv5_filters': 256,
    'fc1_neurons': 4096,
    'fc2_neurons': 4096,
    'lr': 0.01,
    'momentum': 0.9,
    'dropout': 0.5,
    'include_conv1': 1,
    'include_conv2': 1,
    'include_conv3': 1,
    'include_conv4': 1,
    'include_conv5': 1,
    'pooling_type1': 'MaxPool2d',
    'pooling_type2': 'MaxPool2d',
    'pooling_type3': 'MaxPool2d',
    'activation_type': 'ReLU',
    'use_batchnorm': 0 # AlexNet didn't use BN originally
}

code = generate_model_code_string(
    alexnet_chromosome, 
    in_shape=(3, 227, 227), # AlexNet standard input
    out_shape=(10,)
)

# Fix imports if needed or adjust logic (MutNet_evolvable generates imports)
with open('ab/gpt/brute/ga/modular/alexnet_mut.py', 'w') as f:
    f.write(code)

print("Generated ab/gpt/brute/ga/modular/alexnet_mut.py")
