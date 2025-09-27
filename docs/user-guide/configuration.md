# Configuration

DeepScale provides flexible configuration management through YAML files and environment variables.

## Configuration Files

### Basic Structure

```yaml
# deepscale.yaml
training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001

storage:
  backend: "file_system"
  path: "./checkpoints"

monitoring:
  wandb:
    project: "my-project"
    enabled: true
```

### Environment Variables

You can override configuration values using environment variables:

```bash
export DEEPSCALE_TRAINING_EPOCHS=50
export DEEPSCALE_STORAGE_PATH="/data/checkpoints"
```

## Configuration API

```python
from deepscale.config import Config

# Load from file
config = Config.from_file("deepscale.yaml")

# Access values
print(config.training.epochs)
print(config.storage.backend)

# Override values
config.training.learning_rate = 0.01
```

## Advanced Configuration

### Conditional Configuration

```yaml
training:
  epochs: !ENV ${EPOCHS:100}  # Default to 100 if EPOCHS not set
  batch_size: !ENV ${BATCH_SIZE:32}
```

### Multiple Environments

```yaml
# config/development.yaml
training:
  epochs: 10
  debug: true

# config/production.yaml
training:
  epochs: 1000
  debug: false
```