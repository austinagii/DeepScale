# Training

Learn how to use DeepScale for training your machine learning models.

## Basic Training

```python
from deepscale import DeepScale
import torch

# Initialize DeepScale
ds = DeepScale()

# Your model and data
model = YourModel()
dataloader = YourDataLoader()

# Start training
ds.train(model, dataloader, epochs=100)
```

## Distributed Training

DeepScale supports distributed training across multiple GPUs:

```python
# Multi-GPU training
ds = DeepScale(
    distributed=True,
    num_gpus=4
)

ds.train(model, dataloader)
```

## Checkpointing

Automatic checkpointing during training:

```python
ds = DeepScale(
    checkpoint_every=10,  # Save every 10 epochs
    checkpoint_path="./checkpoints"
)

# Resume from checkpoint
ds.resume_from_checkpoint("./checkpoints/epoch_50.pt")
```

## Custom Training Loops

For more control, use custom training loops:

```python
from deepscale.training import TrainingLoop

class CustomTrainingLoop(TrainingLoop):
    def training_step(self, batch, batch_idx):
        # Your custom training logic
        loss = self.model(batch)
        return loss

    def validation_step(self, batch, batch_idx):
        # Your custom validation logic
        return validation_loss

# Use custom loop
ds = DeepScale(training_loop=CustomTrainingLoop())
```