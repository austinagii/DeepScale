# Storage

DeepScale provides flexible storage backends for checkpoints and artifacts.

## Storage Backends

### File System Storage

Store checkpoints on local or network file systems:

```yaml
storage:
  backend: "file_system"
  path: "./checkpoints"
  compression: "gzip"
```

### Cloud Storage

#### Azure Blob Storage

```yaml
storage:
  backend: "azure_blob"
  container: "my-checkpoints"
  connection_string: !ENV ${AZURE_STORAGE_CONNECTION_STRING}
```

#### S3-Compatible Storage

```yaml
storage:
  backend: "s3"
  bucket: "my-checkpoints"
  region: "us-west-2"
  access_key: !ENV ${AWS_ACCESS_KEY_ID}
  secret_key: !ENV ${AWS_SECRET_ACCESS_KEY}
```

## Storage API

```python
from deepscale.storage import get_storage_client

# Get storage client
storage = get_storage_client("file_system", path="./checkpoints")

# Save checkpoint
checkpoint_data = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "epoch": 42
}
storage.save_checkpoint("epoch_42", checkpoint_data)

# Load checkpoint
loaded_checkpoint = storage.load_checkpoint("epoch_42")
```

## Advanced Features

### Compression

Enable compression to reduce storage size:

```python
storage = get_storage_client(
    "file_system",
    path="./checkpoints",
    compression="gzip"
)
```

### Versioning

Automatic versioning of checkpoints:

```python
# Save with automatic versioning
storage.save_checkpoint("model", data, version="auto")

# Load specific version
data = storage.load_checkpoint("model", version="v1.2.3")
```