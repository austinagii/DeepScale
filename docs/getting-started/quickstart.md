# Quick Start

This guide will help you get started with DeepScale in minutes.

## Basic Usage

```python
import deepscale

# Initialize DeepScale configuration
config = deepscale.Config()

# Your training code here
```

## Configuration

DeepScale uses YAML configuration files for easy setup:

```yaml
# config.yaml
training:
  epochs: 10
  batch_size: 32
  learning_rate: 0.001

storage:
  backend: "file_system"
  path: "./checkpoints"
```

## Running Training

```python
from deepscale import DeepScale

# Load configuration
ds = DeepScale.from_config("config.yaml")

# Start training
ds.train(model, dataloader)
```

## Next Steps

- Learn more about [Configuration](../user-guide/configuration.md)
- Explore [Training](../user-guide/training.md) options
- Check out [Storage](../user-guide/storage.md) backends