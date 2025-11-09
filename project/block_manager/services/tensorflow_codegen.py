"""

TensorFlow/Keras Code Generation Service
Generates tf.keras.Model code from architecture graphs
"""

from typing import List, Dict, Any, Optional, Tuple
from collections import deque


def generate_tensorflow_code(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    project_name: str = "GeneratedModel"
) -> Dict[str, str]:
    """
    Generate complete TensorFlow/Keras code including model, training, and data loading.
    
    Args:
        nodes: List of node dictionaries from architecture
        edges: List of edge dictionaries defining connections
        project_name: Name for the generated model class
        
    Returns:
        Dictionary with keys: 'model', 'train', 'dataset', 'config'
    """
    # Topologically sort nodes
    sorted_nodes = topological_sort(nodes, edges)
    
    # Generate different components
    model_code = generate_model_class(sorted_nodes, edges, project_name)
    train_code = generate_training_script(project_name)
    dataset_code = generate_dataset_class(nodes)
    config_code = generate_config_file(nodes)
    
    return {
        'model': model_code,
        'train': train_code,
        'dataset': dataset_code,
        'config': config_code
    }


def topological_sort(nodes: List[Dict], edges: List[Dict]) -> List[Dict]:
    """Sort nodes in topological order based on edges"""
    node_map = {node['id']: node for node in nodes}
    
    # Build adjacency list
    graph = {node['id']: [] for node in nodes}
    in_degree = {node['id']: 0 for node in nodes}
    
    for edge in edges:
        source = edge.get('source')
        target = edge.get('target')
        if source in graph and target in graph:
            graph[source].append(target)
            in_degree[target] += 1
    
    # Kahn's algorithm
    queue = deque([node_id for node_id, degree in in_degree.items() if degree == 0])
    sorted_ids = []
    
    while queue:
        node_id = queue.popleft()
        sorted_ids.append(node_id)
        
        for neighbor in graph[node_id]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    # Return nodes in sorted order
    return [node_map[node_id] for node_id in sorted_ids if node_id in node_map]


def generate_model_class(
    nodes: List[Dict],
    edges: List[Dict],
    project_name: str
) -> str:
    """Generate tf.keras.Model subclass"""
    
    class_name = to_class_name(project_name)
    
    # Generate layer initializations
    layer_inits = []
    for idx, node in enumerate(nodes):
        layer_code = generate_layer_init(node, idx)
        if layer_code:
            layer_inits.append(layer_code)
    
    # Generate forward pass (call method)
    forward_pass = generate_forward_pass(nodes, edges)
    
    # Assemble the model class
    code = f'''import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class {class_name}(keras.Model):
    """Generated TensorFlow model using tf.keras"""
    
    def __init__(self):
        super({class_name}, self).__init__()
        
        # Initialize layers
'''
    
    # Add layer initializations
    for init in layer_inits:
        code += f'        {init}\n'
    
    code += '''
    def call(self, inputs, training=None):
        """Forward pass through the model"""
'''
    
    # Add forward pass logic
    for line in forward_pass:
        code += f'        {line}\n'
    
    code += '''
        return x

def create_model():
    """Create and return the model instance"""
    model = {class_name}()
    return model

if __name__ == '__main__':
    model = create_model()
    print(model.summary())
'''.format(class_name=class_name)
    
    return code


def generate_layer_init(node: Dict, idx: int) -> Optional[str]:
    """Generate initialization code for a single layer"""
    node_type = get_node_type(node)
    config = node.get('data', {}).get('config', {})
    layer_name = f'layer_{idx}'
    
    if node_type == 'input':
        return None  # Input is handled separately
    
    elif node_type == 'dataloader':
        return None  # DataLoader is handled separately
    
    elif node_type == 'conv2d':
        filters = config.get('filters', 32)
        kernel_size = config.get('kernel_size', 3)
        strides = config.get('strides', 1)
        padding = config.get('padding', 'valid')
        activation = config.get('activation', 'None')
        activation_str = f"'{activation}'" if activation != 'None' else 'None'
        return f"self.{layer_name} = layers.Conv2D({filters}, {kernel_size}, strides={strides}, padding='{padding}', activation={activation_str})"
    
    elif node_type == 'conv1d':
        filters = config.get('filters', 32)
        kernel_size = config.get('kernel_size', 3)
        strides = config.get('strides', 1)
        padding = config.get('padding', 'valid')
        return f"self.{layer_name} = layers.Conv1D({filters}, {kernel_size}, strides={strides}, padding='{padding}')"
    
    elif node_type == 'conv3d':
        filters = config.get('filters', 32)
        kernel_size = config.get('kernel_size', 3)
        strides = config.get('strides', 1)
        padding = config.get('padding', 'valid')
        return f"self.{layer_name} = layers.Conv3D({filters}, {kernel_size}, strides={strides}, padding='{padding}')"
    
    elif node_type == 'linear':
        units = config.get('units', 128)
        activation = config.get('activation', 'None')
        use_bias = config.get('use_bias', True)
        activation_str = f"'{activation}'" if activation != 'None' else 'None'
        return f"self.{layer_name} = layers.Dense({units}, activation={activation_str}, use_bias={use_bias})"
    
    elif node_type == 'batchnorm2d':
        momentum = config.get('momentum', 0.99)
        epsilon = config.get('epsilon', 0.001)
        return f"self.{layer_name} = layers.BatchNormalization(momentum={momentum}, epsilon={epsilon})"
    
    elif node_type == 'dropout':
        rate = config.get('rate', 0.5)
        return f"self.{layer_name} = layers.Dropout({rate})"
    
    elif node_type == 'maxpool2d':
        pool_size = config.get('pool_size', 2)
        strides = config.get('strides', 2)
        padding = config.get('padding', 'valid')
        return f"self.{layer_name} = layers.MaxPooling2D(pool_size={pool_size}, strides={strides}, padding='{padding}')"
    
    elif node_type == 'avgpool2d':
        pool_size = config.get('pool_size', 2)
        strides = config.get('strides', 2)
        padding = config.get('padding', 'valid')
        return f"self.{layer_name} = layers.AveragePooling2D(pool_size={pool_size}, strides={strides}, padding='{padding}')"
    
    elif node_type == 'adaptiveavgpool2d':
        keepdims = config.get('keepdims', False)
        if keepdims:
            return f"self.{layer_name} = layers.GlobalAveragePooling2D(keepdims=True)"
        else:
            return f"self.{layer_name} = layers.GlobalAveragePooling2D()"
    
    elif node_type == 'flatten':
        return f"self.{layer_name} = layers.Flatten()"
    
    elif node_type == 'lstm':
        units = config.get('units', 128)
        return_sequences = config.get('return_sequences', False)
        dropout = config.get('dropout', 0.0)
        recurrent_dropout = config.get('recurrent_dropout', 0.0)
        return f"self.{layer_name} = layers.LSTM({units}, return_sequences={return_sequences}, dropout={dropout}, recurrent_dropout={recurrent_dropout})"
    
    elif node_type == 'gru':
        units = config.get('units', 128)
        return_sequences = config.get('return_sequences', False)
        dropout = config.get('dropout', 0.0)
        recurrent_dropout = config.get('recurrent_dropout', 0.0)
        return f"self.{layer_name} = layers.GRU({units}, return_sequences={return_sequences}, dropout={dropout}, recurrent_dropout={recurrent_dropout})"
    
    elif node_type == 'embedding':
        input_dim = config.get('input_dim', 1000)
        output_dim = config.get('output_dim', 128)
        mask_zero = config.get('mask_zero', False)
        return f"self.{layer_name} = layers.Embedding({input_dim}, {output_dim}, mask_zero={mask_zero})"
    
    elif node_type == 'concat':
        axis = config.get('axis', -1)
        return f"self.{layer_name} = layers.Concatenate(axis={axis})"
    
    elif node_type == 'add':
        return f"self.{layer_name} = layers.Add()"
    
    return None


def generate_forward_pass(nodes: List[Dict], edges: List[Dict]) -> List[str]:
    """Generate the call() method logic"""
    forward = []
    node_map = {node['id']: (idx, node) for idx, node in enumerate(nodes)}
    var_map = {}
    
    # Build edge map for finding inputs
    edge_map = {}
    for edge in edges:
        target = edge.get('target')
        source = edge.get('source')
        if target not in edge_map:
            edge_map[target] = []
        edge_map[target].append(source)
    
    for idx, node in enumerate(nodes):
        node_id = node['id']
        node_type = get_node_type(node)
        layer_name = f'layer_{idx}'
        
        # Get input variable name
        incoming = edge_map.get(node_id, [])
        
        if node_type in ('input', 'dataloader'):
            var_map[node_id] = 'inputs'
            continue
        
        # Determine input variable
        if not incoming:
            input_var = 'inputs'
        elif len(incoming) == 1:
            input_var = var_map.get(incoming[0], 'inputs')
        else:
            # Multiple inputs (for concat, add, etc.)
            input_vars = [var_map.get(src, 'inputs') for src in incoming]
            input_var = f"[{', '.join(input_vars)}]"
        
        # Generate forward pass line
        output_var = 'x' if idx == 0 or node_type in ('input', 'dataloader') else f'x'
        
        if node_type in ('concat', 'add'):
            forward.append(f"x = self.{layer_name}({input_var})")
        else:
            # Check if layer needs training parameter
            if node_type in ('dropout', 'batchnorm2d'):
                forward.append(f"x = self.{layer_name}({input_var}, training=training)")
            else:
                forward.append(f"x = self.{layer_name}({input_var})")
        
        var_map[node_id] = 'x'
    
    if not forward:
        forward.append("x = inputs")
    
    return forward


def generate_training_script(project_name: str) -> str:
    """Generate training script"""
    class_name = to_class_name(project_name)
    
    return f'''import tensorflow as tf
from tensorflow import keras
import numpy as np
from model import create_model
from dataset import CustomDataset

def train_model():
    """Train the generated model"""
    
    # Set random seed for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Create model
    model = create_model()
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    # TODO: Replace with your actual dataset
    # Example dataset creation:
    # train_dataset = CustomDataset(data_path='train/', batch_size=32)
    # val_dataset = CustomDataset(data_path='val/', batch_size=32)
    
    # For demonstration, create dummy data
    # Replace this with your actual data loading
    batch_size = 32
    num_samples = 1000
    
    # Create dummy dataset (NHWC format: batch, height, width, channels)
    x_train = np.random.randn(num_samples, 224, 224, 3).astype(np.float32)
    y_train = np.random.randint(0, 10, num_samples)
    
    x_val = np.random.randn(200, 224, 224, 3).astype(np.float32)
    y_val = np.random.randint(0, 10, 200)
    
    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7
        ),
        keras.callbacks.TensorBoard(log_dir='./logs')
    ]
    
    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=10,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save('{project_name}_final.h5')
    print(f"Model saved to {project_name}_final.h5")
    
    return history

if __name__ == '__main__':
    history = train_model()
    print("Training completed!")
'''


def generate_dataset_class(nodes: List[Dict]) -> str:
    """Generate PyDataset class for data loading"""
    
    return '''import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path

class CustomDataset(keras.utils.PyDataset):
    """
    Custom dataset using tf.keras.utils.PyDataset
    
    Replace this with your actual data loading logic.
    PyDataset is the recommended way to create datasets in TensorFlow 2.x
    """
    
    def __init__(self, data_path, batch_size=32, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # TODO: Load your actual data
        # Example: self.data = load_data(data_path)
        # For now, we'll create dummy data
        self.num_samples = 1000
        
    def __len__(self):
        """Return number of batches per epoch"""
        return self.num_samples // self.batch_size
    
    def __getitem__(self, idx):
        """
        Generate one batch of data
        
        Returns:
            tuple: (inputs, targets) where inputs are in NHWC format
        """
        # TODO: Replace with actual data loading
        # This is just a placeholder implementation
        
        # Generate dummy batch (NHWC format: batch, height, width, channels)
        batch_x = np.random.randn(self.batch_size, 224, 224, 3).astype(np.float32)
        batch_y = np.random.randint(0, 10, self.batch_size)
        
        return batch_x, batch_y
    
    def on_epoch_end(self):
        """Called at the end of each epoch"""
        if self.shuffle:
            # TODO: Implement shuffling logic
            pass

# Example usage:
if __name__ == '__main__':
    dataset = CustomDataset('data/', batch_size=32)
    print(f"Dataset has {len(dataset)} batches")
    
    # Get one batch
    batch_x, batch_y = dataset[0]
    print(f"Batch X shape: {batch_x.shape}")  # Should be (32, 224, 224, 3) in NHWC format
    print(f"Batch Y shape: {batch_y.shape}")
'''


def generate_config_file(nodes: List[Dict]) -> str:
    """Generate configuration file"""
    
    # Find input shape from nodes
    input_shape = "[1, 224, 224, 3]"
    for node in nodes:
        if get_node_type(node) in ('input', 'dataloader'):
            shape = node.get('data', {}).get('outputShape', {}).get('dims')
            if shape:
                input_shape = str(shape)
                break
    
    return f'''# TensorFlow Model Configuration

# Training Configuration
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
WEIGHT_DECAY = 0.0001

# Model Configuration (NHWC format: batch, height, width, channels)
INPUT_SHAPE = {input_shape}

# Optimizer
OPTIMIZER = 'adam'  # Options: 'adam', 'sgd', 'rmsprop', 'adamw'
MOMENTUM = 0.9  # For SGD

# Learning Rate Scheduler
USE_SCHEDULER = True
SCHEDULER_TYPE = 'reduce_on_plateau'  # Options: 'reduce_on_plateau', 'exponential', 'cosine'
LR_PATIENCE = 3
LR_FACTOR = 0.5

# Early Stopping
EARLY_STOPPING_PATIENCE = 5

# Data Augmentation
USE_AUGMENTATION = True
ROTATION_RANGE = 15
WIDTH_SHIFT_RANGE = 0.1
HEIGHT_SHIFT_RANGE = 0.1
HORIZONTAL_FLIP = True

# Device Configuration
# TensorFlow automatically uses GPU if available
# To disable GPU: tf.config.set_visible_devices([], 'GPU')

# Mixed Precision Training (for faster training on modern GPUs)
USE_MIXED_PRECISION = False
'''


def get_node_type(node: Dict) -> str:
    """Extract node type from node dictionary"""
    return node.get('data', {}).get('blockType', node.get('type', 'unknown'))


def to_class_name(name: str) -> str:
    """Convert project name to valid class name"""
    # Remove special characters and convert to PascalCase
    import re
    name = re.sub(r'[^a-zA-Z0-9]', ' ', name)
    name = ''.join(word.capitalize() for word in name.split())
    if not name:
        return 'GeneratedModel'
    if name[0].isdigit():
        name = 'Model' + name
    return name
