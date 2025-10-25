import torch
import argparse
import yaml
from torch.nn import functional as F
import importlib
import sys
sys.path.append('/root/autodl-tmp/IR')
print(sys.path)
from time import time

def prepare_model(opts):
    ''' Prepare the model by loading it from the specified module. '''
    net = importlib.import_module('models.' + opts.model_name)
    model = net.Video_Backbone()
    return model

def count_parameters(model):
    ''' Count the number of parameters in the model. '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_flops_and_macs(model, input_shape, device):
    ''' Count FLOPs and MACs using thop. '''
    from thop import profile
    input_tensor = torch.randn(input_shape).to(device)  # Move input tensor to the specified device
    t1 = time()
    macs, params = profile(model, inputs=(input_tensor,), verbose=False)
    t2 = time()
    print((t2-t1)/6)
    flops = 2 * macs  # 1 MAC = 2 FLOPs
    return flops, macs, params

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='', help='The name of this experiment')
    parser.add_argument('--model_name', type=str, default='RNN_Swin_5', help='The name of adopted model')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to use (e.g., "cuda:0" or "cpu")')
    opts = parser.parse_args()

    # Initialize the model
    model = prepare_model(opts).to(opts.device)

    # Count the number of parameters
    num_params = count_parameters(model)
    print(f"Number of parameters: {num_params}")

    # Count the number of FLOPs and MACs
    input_shape = (1, 6, 3, 256, 256)  # Example input shape (batch_size, frames, channels, height, width)
    flops, macs, params = count_flops_and_macs(model, input_shape, opts.device)
    gflops = flops / 1e9 / 6 # Convert FLOPs to GFLOPs
    gmacs = macs / 1e9 / 6 # Convert MACs to GMACs
    params = params / 1e6
    print(f"Number of GFLOPs: {gflops:.2f} GFLOPs")
    print(f"Number of GMACs: {gmacs:.2f} GMACs")
    print(f"Number of parameters (using thop): {params}")