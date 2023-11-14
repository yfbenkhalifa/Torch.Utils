from ast import Is
from numpy import isin
import torch.nn as nn
import numpy as np
import collections.abc


class FactoryInterface():
    def create(self):
        pass

class CNNFactory(FactoryInterface):
    def __init__(self, input_shape: [int, int], cnn_layer : nn.Module, ffn_layer : nn.Module):
        """Input Parameters"""
        self.input_shape = input_shape
        self.input_channels = cnn_layer[0].in_channels
        self.output_shape : [int, int]
        self.output_channels : int
        
        self.cnn_layer = cnn_layer
        self.ffn_layer = ffn_layer
        """Network parameters"""
        self.channels = []
        self.kernels = []
        self.paddings = []
        self.strides = []
        
        """Output parameters"""
        self.cnn_output_shape : [int, int]
        self.fnn_input_shape : int
        
        for module in cnn_layer:
            if not isinstance(module, nn.Module):
                raise TypeError("cnn_layer must be a list of nn.Module")
            if hasattr(module, 'kernel_size'):
                self.kernels.append(module.kernel_size)
            if hasattr(module, 'padding'):
                self.paddings.append(module.padding)
            if hasattr(module, 'stride'):
                self.strides.append(module.stride)
                
            # Extract the output channels
            if isinstance(module, nn.Conv2d):
                self.output_channels = module.out_channels
                
    def create(self):
        ffn_input_layer = self.create_flatten_layer()
        return nn.Sequential(
            self.cnn_layer,
            nn.Flatten(),
            ffn_input_layer,
            self.ffn_layer
        )
                
    def create_flatten_layer(self):
        ffn_input_shape = self.compute_ffn_input_shape()
        return nn.Linear(np.int32(ffn_input_shape), self.ffn_layer[0].in_features)
        
                
    def compute_ffn_input_shape(self):
        cnn_output_shape = self.compute_cnn_output_shape()
        return cnn_output_shape[0] * cnn_output_shape[1] * self.output_channels
    
    def compute_cnn_output_shape(self):
        output_shape = self.input_shape.copy()
        for module in self.cnn_layer:
            output_shape = self.compute_layer_output_shape(output_shape, module)
        return output_shape
        
    def validate(self):
        return len(self.kernels) == len(self.paddings) == len(self.strides)
        
    def compute_layer_output_shape(self, input_shape, layer) -> [int, int]:
        output_shape = input_shape
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.MaxPool2d):
            kernel_size = layer.kernel_size if isinstance(layer.kernel_size, collections.abc.Sequence) else [layer.kernel_size, layer.kernel_size]
            padding = layer.padding if isinstance(layer.padding, collections.abc.Sequence) else [layer.padding, layer.padding]
            stride = layer.stride if isinstance(layer.stride, collections.abc.Sequence) else [layer.stride, layer.stride]

            output_shape[0] = np.floor(
                ((output_shape[0] + 2*padding[0] - (kernel_size[0]-1)-1) / stride[0]) + 1
            )
            output_shape[1] = np.floor(
                ((output_shape[1] + 2*padding[1] - (kernel_size[1]-1)-1) / stride[1]) + 1
            )
        if isinstance(layer, nn.AvgPool2d):
            output_shape[0] = np.floor(
                ((output_shape[0] + 2*padding[0] - (kernel_size[0])) / stride[0]) + 1
            )
            output_shape[1] = np.floor(
                ((output_shape[1] + 2*padding[1] - (kernel_size[1])) / stride[1]) + 1
            )
            
        return output_shape
    