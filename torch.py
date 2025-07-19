#!/usr/bin/env python3
"""
Torch stub module for tiny_village project.
This provides minimal torch-like functionality without requiring the full PyTorch installation.
"""

import random


class MockTensor:
    """Simple mock tensor class that behaves like a list/array."""
    
    def __init__(self, data, shape=None):
        if isinstance(data, (list, tuple)):
            self.data = data
            self.shape = shape or self._calculate_shape(data)
        else:
            self.data = data
            self.shape = shape or ()
    
    def _calculate_shape(self, data):
        """Calculate shape of nested list."""
        if not isinstance(data, (list, tuple)):
            return ()
        shape = [len(data)]
        if data and isinstance(data[0], (list, tuple)):
            shape.extend(self._calculate_shape(data[0]))
        return tuple(shape)
    
    def __repr__(self):
        return f"MockTensor({self.data})"
    
    def size(self):
        return self.shape
    
    def to(self, device):
        """Mock device transfer."""
        return self


def rand(*size):
    """
    Generate random tensor-like values.
    
    Previously this function always returned 0, which masked issues where randomness is expected.
    Now it returns actual random values to ensure tests fail when functions don't work properly.
    
    Args:
        *size: Dimensions of the tensor to generate
        
    Returns:
        MockTensor: Random values between 0 and 1
    """
    if len(size) == 0:
        return random.random()
    elif len(size) == 1:
        data = [random.random() for _ in range(size[0])]
        return MockTensor(data, size)
    elif len(size) == 2:
        data = [[random.random() for _ in range(size[1])] for _ in range(size[0])]
        return MockTensor(data, size)
    elif len(size) == 3:
        data = [[[random.random() for _ in range(size[2])] for _ in range(size[1])] for _ in range(size[0])]
        return MockTensor(data, size)
    else:
        # For higher dimensions, just return a flat list with the right length
        total_size = 1
        for dim in size:
            total_size *= dim
        data = [random.random() for _ in range(total_size)]
        return MockTensor(data, size)


def randn(*size):
    """Generate random tensor with normal distribution."""
    if len(size) == 0:
        return random.gauss(0, 1)
    elif len(size) == 1:
        data = [random.gauss(0, 1) for _ in range(size[0])]
        return MockTensor(data, size)
    elif len(size) == 2:
        data = [[random.gauss(0, 1) for _ in range(size[1])] for _ in range(size[0])]
        return MockTensor(data, size)
    elif len(size) == 3:
        data = [[[random.gauss(0, 1) for _ in range(size[2])] for _ in range(size[1])] for _ in range(size[0])]
        return MockTensor(data, size)
    else:
        # For higher dimensions, just return a flat list with the right length
        total_size = 1
        for dim in size:
            total_size *= dim
        data = [random.gauss(0, 1) for _ in range(total_size)]
        return MockTensor(data, size)


def zeros(*size):
    """Generate zero tensor."""
    if len(size) == 0:
        return 0.0
    elif len(size) == 1:
        data = [0.0] * size[0]
        return MockTensor(data, size)
    elif len(size) == 2:
        data = [[0.0] * size[1] for _ in range(size[0])]
        return MockTensor(data, size)
    elif len(size) == 3:
        data = [[[0.0] * size[2] for _ in range(size[1])] for _ in range(size[0])]
        return MockTensor(data, size)
    else:
        # For higher dimensions, just return zeros
        total_size = 1
        for dim in size:
            total_size *= dim
        data = [0.0] * total_size
        return MockTensor(data, size)


def ones(*size):
    """Generate ones tensor."""
    if len(size) == 0:
        return 1.0
    elif len(size) == 1:
        data = [1.0] * size[0]
        return MockTensor(data, size)
    elif len(size) == 2:
        data = [[1.0] * size[1] for _ in range(size[0])]
        return MockTensor(data, size)
    elif len(size) == 3:
        data = [[[1.0] * size[2] for _ in range(size[1])] for _ in range(size[0])]
        return MockTensor(data, size)
    else:
        # For higher dimensions, just return ones
        total_size = 1
        for dim in size:
            total_size *= dim
        data = [1.0] * total_size
        return MockTensor(data, size)


def tensor(data, dtype=None):
    """Create tensor from data."""
    return MockTensor(data)


def cat(tensors, dim=0):
    """Concatenate tensors."""
    # Simple concatenation mock
    if not tensors:
        return MockTensor([])
    
    # For simplicity, just combine the data
    combined_data = []
    for t in tensors:
        if hasattr(t, 'data'):
            combined_data.extend(t.data if isinstance(t.data, list) else [t.data])
        else:
            combined_data.append(t)
    
    return MockTensor(combined_data)


def clamp(input, min=None, max=None):
    """Clamp tensor values."""
    if hasattr(input, 'data'):
        if isinstance(input.data, list):
            clamped = []
            for item in input.data:
                if isinstance(item, list):
                    clamped.append([max(min or float('-inf'), min(max or float('inf'), x)) for x in item])
                else:
                    clamped.append(max(min or float('-inf'), min(max or float('inf'), item)))
            return MockTensor(clamped, input.shape)
        else:
            return max(min or float('-inf'), min(max or float('inf'), input.data))
    else:
        return max(min or float('-inf'), min(max or float('inf'), input))


def matmul(a, b):
    """Matrix multiplication mock."""
    # Simple mock - just return a MockTensor
    return MockTensor([[1.0, 2.0], [3.0, 4.0]])


def arange(start, end=None, step=1):
    """Generate range of values."""
    if end is None:
        data = list(range(start))
    else:
        data = list(range(start, end, step))
    return MockTensor(data)


# Device management
def device(device_str):
    """Mock device function."""
    return device_str


# Mock CUDA functions
class cuda:
    @staticmethod
    def is_available():
        return False


def manual_seed(seed):
    """Set random seed."""
    random.seed(seed)


# Stub classes for compatibility
class Graph:
    """Mock Graph class."""
    pass


def eq(a, b):
    """Element-wise equality comparison."""
    return np.equal(a, b)


# Mock for classes and modules that might be used
class nn:
    """Mock nn module."""
    class Module:
        pass
    
    class Linear:
        def __init__(self, *args, **kwargs):
            pass
    
    class LSTM:
        def __init__(self, *args, **kwargs):
            pass
    
    class MSELoss:
        def __init__(self, *args, **kwargs):
            pass
            
        def __call__(self, *args, **kwargs):
            return 0.0
    
    class MultiheadAttention:
        def __init__(self, *args, **kwargs):
            pass
    
    class GRU:
        def __init__(self, *args, **kwargs):
            pass


class optim:
    """Mock optim module."""
    class Adam:
        def __init__(self, *args, **kwargs):
            pass
        
        def zero_grad(self):
            pass
        
        def step(self):
            pass


# Add additional mock functionality as needed
class no_grad:
    """Mock no_grad context manager."""
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass