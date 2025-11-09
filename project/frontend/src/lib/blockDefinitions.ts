import { BlockDefinition, TensorShape, BlockConfig } from './types'

export const blockDefinitions: Record<string, BlockDefinition> = {
  input: {
    type: 'input',
    label: 'Input',
    category: 'input',
    color: 'var(--color-gray)',
    icon: 'CircleDashed',
    description: 'Input placeholder node',
    configSchema: [
      {
        name: 'label',
        label: 'Custom Label',
        type: 'text',
        default: 'Input',
        placeholder: 'Enter custom label...',
        description: 'Custom label for this input node'
      },
      {
        name: 'note',
        label: 'Note',
        type: 'text',
        placeholder: 'Add notes here...',
        description: 'Notes or comments about this input'
      }
    ],
    computeOutputShape: (inputShape) => inputShape
  },

  dataloader: {
    type: 'dataloader',
    label: 'Data Loader',
    category: 'input',
    color: 'var(--color-teal)',
    icon: 'Database',
    description: 'Load and prepare input data with optional ground truth',
    configSchema: [
      {
        name: 'shape',
        label: 'Input Shape',
        type: 'text',
        default: '[1, 3, 224, 224]',
        required: true,
        placeholder: '[batch, channels, height, width]',
        description: 'Input tensor dimensions as JSON array'
      },
      {
        name: 'has_ground_truth',
        label: 'Include Ground Truth Output',
        type: 'boolean',
        default: false,
        description: 'Enable a second output for ground truth labels'
      },
      {
        name: 'ground_truth_shape',
        label: 'Ground Truth Shape',
        type: 'text',
        default: '[1, 10]',
        placeholder: '[batch, num_classes]',
        description: 'Shape for ground truth labels (used when ground truth is enabled)'
      },
      {
        name: 'randomize',
        label: 'Randomize Data',
        type: 'boolean',
        default: false,
        description: 'Use random synthetic data for testing'
      },
      {
        name: 'csv_file',
        label: 'CSV File Path',
        type: 'text',
        placeholder: 'data/dataset.csv',
        description: 'Path to CSV file for data loading (optional)'
      }
    ],
    computeOutputShape: (_, config) => {
      const shapeStr = String(config.shape || '[1]')
      try {
        const dims = JSON.parse(shapeStr)
        if (Array.isArray(dims) && dims.length > 0 && dims.every(d => typeof d === 'number' && d > 0)) {
          return {
            dims,
            description: 'Input tensor'
          }
        }
      } catch {
        return undefined
      }
      return undefined
    }
  },

  output: {
    type: 'output',
    label: 'Output',
    category: 'output',
    color: 'var(--color-green)',
    icon: 'Export',
    description: 'Define model output and predictions',
    configSchema: [
      {
        name: 'output_type',
        label: 'Output Type',
        type: 'select',
        default: 'classification',
        options: [
          { value: 'classification', label: 'Classification' },
          { value: 'regression', label: 'Regression' },
          { value: 'segmentation', label: 'Segmentation' },
          { value: 'custom', label: 'Custom' }
        ],
        description: 'Type of model output'
      },
      {
        name: 'num_classes',
        label: 'Number of Classes',
        type: 'number',
        default: 10,
        min: 1,
        description: 'Number of output classes (for classification)'
      }
    ],
    computeOutputShape: (inputShape) => inputShape
  },

  loss: {
    type: 'loss',
    label: 'Loss Function',
    category: 'output',
    color: 'var(--color-destructive)',
    icon: 'Target',
    description: 'Define loss function for training',
    configSchema: [
      {
        name: 'loss_type',
        label: 'Loss Type',
        type: 'select',
        default: 'cross_entropy',
        required: true,
        options: [
          { value: 'cross_entropy', label: 'Cross Entropy Loss' },
          { value: 'mse', label: 'Mean Squared Error' },
          { value: 'mae', label: 'Mean Absolute Error' },
          { value: 'bce', label: 'Binary Cross Entropy' },
          { value: 'nll', label: 'Negative Log Likelihood' },
          { value: 'smooth_l1', label: 'Smooth L1 Loss' },
          { value: 'kl_div', label: 'KL Divergence' },
          { value: 'custom', label: 'Custom Loss' }
        ],
        description: 'Type of loss function to use'
      },
      {
        name: 'reduction',
        label: 'Reduction',
        type: 'select',
        default: 'mean',
        options: [
          { value: 'mean', label: 'Mean' },
          { value: 'sum', label: 'Sum' },
          { value: 'none', label: 'None' }
        ],
        description: 'How to reduce the loss'
      },
      {
        name: 'weight',
        label: 'Class Weights',
        type: 'text',
        placeholder: '[1.0, 1.0, 2.0, ...]',
        description: 'Optional class weights as JSON array'
      }
    ],
    computeOutputShape: () => ({ dims: [1], description: 'Scalar loss' })
  },

  empty: {
    type: 'empty',
    label: 'Empty Node',
    category: 'utility',
    color: 'var(--color-gray)',
    icon: 'Placeholder',
    description: 'Placeholder node for architecture planning',
    configSchema: [
      {
        name: 'label',
        label: 'Custom Label',
        type: 'text',
        default: 'Empty Node',
        placeholder: 'Enter custom label...',
        description: 'Custom label for this node'
      },
      {
        name: 'note',
        label: 'Note',
        type: 'text',
        placeholder: 'Add notes here...',
        description: 'Notes or comments about this placeholder'
      }
    ],
    computeOutputShape: (inputShape) => inputShape
  },

  linear: {
    type: 'linear',
    label: 'Linear',
    category: 'basic',
    color: 'var(--color-primary)',
    icon: 'Lightning',
    description: 'Fully connected layer',
    configSchema: [
      {
        name: 'out_features',
        label: 'Output Features',
        type: 'number',
        required: true,
        min: 1,
        description: 'Number of output features'
      },
      {
        name: 'bias',
        label: 'Use Bias',
        type: 'boolean',
        default: true,
        description: 'Add learnable bias'
      }
    ],
    computeOutputShape: (inputShape, config) => {
      if (!inputShape || !config.out_features) return undefined
      
      if (inputShape.dims.length !== 2) {
        return undefined
      }
      
      return {
        dims: [inputShape.dims[0], config.out_features as number],
        description: 'Fully connected output'
      }
    }
  },

  conv2d: {
    type: 'conv2d',
    label: 'Conv2D',
    category: 'basic',
    color: 'var(--color-purple)',
    icon: 'SquareHalf',
    description: '2D convolutional layer',
    configSchema: [
      {
        name: 'out_channels',
        label: 'Output Channels',
        type: 'number',
        required: true,
        min: 1,
        description: 'Number of output channels'
      },
      {
        name: 'kernel_size',
        label: 'Kernel Size',
        type: 'number',
        default: 3,
        min: 1,
        description: 'Size of convolving kernel'
      },
      {
        name: 'stride',
        label: 'Stride',
        type: 'number',
        default: 1,
        min: 1,
        description: 'Stride of convolution'
      },
      {
        name: 'padding',
        label: 'Padding',
        type: 'number',
        default: 0,
        min: 0,
        description: 'Zero-padding added to both sides'
      },
      {
        name: 'dilation',
        label: 'Dilation',
        type: 'number',
        default: 1,
        min: 1,
        description: 'Spacing between kernel elements'
      }
    ],
    computeOutputShape: (inputShape, config) => {
      if (!inputShape || !config.out_channels) return undefined
      
      if (inputShape.dims.length !== 4) {
        return undefined
      }
      
      const [batch, _, height, width] = inputShape.dims as number[]
      const kernel = config.kernel_size as number
      const stride = config.stride as number
      const padding = config.padding as number
      
      const outHeight = Math.floor((height + 2 * padding - kernel) / stride + 1)
      const outWidth = Math.floor((width + 2 * padding - kernel) / stride + 1)
      
      return {
        dims: [batch, config.out_channels as number, outHeight, outWidth],
        description: 'Convolutional output'
      }
    }
  },

  dropout: {
    type: 'dropout',
    label: 'Dropout',
    category: 'basic',
    color: 'var(--color-orange)',
    icon: 'Drop',
    description: 'Dropout regularization',
    configSchema: [
      {
        name: 'rate',
        label: 'Dropout Rate',
        type: 'number',
        default: 0.5,
        min: 0,
        max: 1,
        description: 'Probability of dropping a unit'
      }
    ],
    computeOutputShape: (inputShape) => inputShape
  },

  batchnorm: {
    type: 'batchnorm',
    label: 'BatchNorm',
    category: 'basic',
    color: 'var(--color-primary)',
    icon: 'ChartBar',
    description: 'Batch normalization',
    configSchema: [
      {
        name: 'momentum',
        label: 'Momentum',
        type: 'number',
        default: 0.1,
        min: 0,
        max: 1,
        description: 'Momentum for running mean/var'
      },
      {
        name: 'eps',
        label: 'Epsilon',
        type: 'number',
        default: 0.00001,
        min: 0,
        description: 'Value added for numerical stability'
      },
      {
        name: 'affine',
        label: 'Affine Transform',
        type: 'boolean',
        default: true,
        description: 'Learn affine parameters (gamma, beta)'
      }
    ],
    computeOutputShape: (inputShape) => inputShape
  },

  relu: {
    type: 'relu',
    label: 'ReLU',
    category: 'basic',
    color: 'var(--color-destructive)',
    icon: 'TrendUp',
    description: 'Rectified Linear Unit activation',
    configSchema: [],
    computeOutputShape: (inputShape) => inputShape
  },

  flatten: {
    type: 'flatten',
    label: 'Flatten',
    category: 'basic',
    color: 'var(--color-primary)',
    icon: 'ArrowsHorizontal',
    description: 'Flatten tensor to 2D',
    configSchema: [
      {
        name: 'start_dim',
        label: 'Start Dimension',
        type: 'number',
        default: 1,
        min: 0,
        description: 'First dimension to flatten'
      }
    ],
    computeOutputShape: (inputShape, config) => {
      if (!inputShape) return undefined
      
      const startDim = (config.start_dim as number) || 1
      const dims = inputShape.dims as number[]
      
      if (startDim >= dims.length) return undefined
      
      const batchSize = dims.slice(0, startDim).reduce((a, b) => a * b, 1)
      const flatSize = dims.slice(startDim).reduce((a, b) => a * b, 1)
      
      return {
        dims: [batchSize, flatSize],
        description: 'Flattened tensor'
      }
    }
  },

  maxpool: {
    type: 'maxpool',
    label: 'MaxPool2D',
    category: 'basic',
    color: 'var(--color-accent)',
    icon: 'ArrowDown',
    description: '2D max pooling',
    configSchema: [
      {
        name: 'kernel_size',
        label: 'Kernel Size',
        type: 'number',
        default: 2,
        min: 1,
        description: 'Size of pooling window'
      },
      {
        name: 'stride',
        label: 'Stride',
        type: 'number',
        default: 2,
        min: 1,
        description: 'Stride of pooling window'
      },
      {
        name: 'padding',
        label: 'Padding',
        type: 'number',
        default: 0,
        min: 0,
        description: 'Zero-padding added to both sides'
      }
    ],
    computeOutputShape: (inputShape, config) => {
      if (!inputShape) return undefined
      
      if (inputShape.dims.length !== 4) return undefined
      
      const [batch, channels, height, width] = inputShape.dims as number[]
      const kernel = config.kernel_size as number
      const stride = config.stride as number
      
      const outHeight = Math.floor((height - kernel) / stride + 1)
      const outWidth = Math.floor((width - kernel) / stride + 1)
      
      return {
        dims: [batch, channels, outHeight, outWidth],
        description: 'Pooled output'
      }
    }
  },

  attention: {
    type: 'attention',
    label: 'Multi-Head Attention',
    category: 'advanced',
    color: 'var(--color-purple)',
    icon: 'Eye',
    description: 'Multi-head self-attention',
    configSchema: [
      {
        name: 'num_heads',
        label: 'Number of Heads',
        type: 'number',
        required: true,
        default: 8,
        min: 1,
        description: 'Number of attention heads'
      },
      {
        name: 'dropout',
        label: 'Dropout',
        type: 'number',
        default: 0.1,
        min: 0,
        max: 1,
        description: 'Attention dropout rate'
      }
    ],
    computeOutputShape: (inputShape) => {
      if (!inputShape) return undefined
      
      if (inputShape.dims.length !== 3) return undefined
      
      return {
        dims: inputShape.dims,
        description: 'Attention output'
      }
    }
  },

  concat: {
    type: 'concat',
    label: 'Concatenate',
    category: 'merge',
    color: 'var(--color-accent)',
    icon: 'GitMerge',
    description: 'Concatenate multiple tensors',
    configSchema: [
      {
        name: 'dim',
        label: 'Dimension',
        type: 'number',
        default: 1,
        description: 'Dimension along which to concatenate'
      }
    ],
    computeOutputShape: () => undefined
  },

  add: {
    type: 'add',
    label: 'Add',
    category: 'merge',
    color: 'var(--color-accent)',
    icon: 'Plus',
    description: 'Element-wise addition of tensors',
    configSchema: [],
    computeOutputShape: (inputShape) => {
      if (!inputShape) return undefined
      return {
        ...inputShape,
        description: 'Element-wise sum'
      }
    }
  },

  custom: {
    type: 'custom',
    label: 'Custom Layer',
    category: 'advanced',
    color: 'var(--color-primary)',
    icon: 'Code',
    description: 'Custom layer with user-defined operations',
    configSchema: [
      {
        name: 'name',
        label: 'Layer Name',
        type: 'text',
        required: true,
        placeholder: 'my_custom_layer',
        description: 'Name for your custom layer'
      },
      {
        name: 'code',
        label: 'Python Code',
        type: 'text',
        default: '# Define your forward pass\n# Input: x\n# Output: return x\nreturn x',
        description: 'Custom forward pass implementation'
      },
      {
        name: 'output_shape',
        label: 'Output Shape',
        type: 'text',
        placeholder: '[batch, features]',
        description: 'Expected output shape (optional, leave empty to match input)'
      },
      {
        name: 'description',
        label: 'Description',
        type: 'text',
        placeholder: 'Describe what this layer does',
        description: 'Brief description of the layer functionality'
      }
    ],
    computeOutputShape: (inputShape, config) => {
      if (config.output_shape) {
        try {
          const dims = JSON.parse(String(config.output_shape))
          if (Array.isArray(dims) && dims.length > 0) {
            return {
              dims,
              description: String(config.description || 'Custom output')
            }
          }
        } catch {
          return inputShape
        }
      }
      return inputShape
    }
  },

  softmax: {
    type: 'softmax',
    label: 'Softmax',
    category: 'basic',
    color: 'var(--color-destructive)',
    icon: 'Percent',
    description: 'Softmax activation',
    configSchema: [
      {
        name: 'dim',
        label: 'Dimension',
        type: 'number',
        default: -1,
        description: 'Dimension along which to apply'
      }
    ],
    computeOutputShape: (inputShape) => inputShape
  }
}

export function getBlockDefinition(type: string): BlockDefinition | undefined {
  return blockDefinitions[type]
}

export function getBlocksByCategory(category: string): BlockDefinition[] {
  return Object.values(blockDefinitions).filter(b => b.category === category)
}

/**
 * Connection rules between blocks based on tensor dimensions
 * Returns an error message if connection is invalid, undefined if valid
 */
export function validateBlockConnection(
  sourceBlockType: string,
  targetBlockType: string,
  sourceOutputShape?: TensorShape
): string | undefined {
  // Data Loader blocks can't receive connections (they are source nodes)
  if (targetBlockType === 'dataloader') {
    return 'Data Loader blocks cannot receive connections'
  }

  // Output and Loss blocks can receive connections (they're terminal nodes)
  if (targetBlockType === 'output' || targetBlockType === 'loss') {
    return undefined // Always valid
  }

  // Empty nodes are passthrough, always valid
  if (targetBlockType === 'empty' || sourceBlockType === 'empty') {
    return undefined
  }

  // Input and DataLoader nodes can connect without configured shapes
  if (sourceBlockType === 'input' || sourceBlockType === 'dataloader') {
    return undefined
  }

  // Source must have valid output shape (except for custom blocks which are flexible)
  if (!sourceOutputShape && sourceBlockType !== 'custom') {
    return 'Source block must have a valid output shape'
  }

  // If we have a shape, validate dimension requirements
  if (sourceOutputShape) {
    const dims = sourceOutputShape.dims.length

    // Conv2D and MaxPool2D require 4D input [batch, channels, height, width]
    if ((targetBlockType === 'conv2d' || targetBlockType === 'maxpool') && dims !== 4) {
      return `${targetBlockType === 'conv2d' ? 'Conv2D' : 'MaxPool2D'} requires 4D input [batch, channels, height, width], got ${dims}D`
    }

    // Linear requires 2D input [batch, features]
    if (targetBlockType === 'linear' && dims !== 2) {
      return `Linear layer requires 2D input [batch, features], got ${dims}D. Consider adding a Flatten layer first.`
    }

    // Multi-Head Attention requires 3D input [batch, sequence, embedding]
    if (targetBlockType === 'attention' && dims !== 3) {
      return `Multi-Head Attention requires 3D input [batch, sequence, embedding], got ${dims}D`
    }

    // BatchNorm works with 2D or 4D
    if (targetBlockType === 'batchnorm' && dims !== 2 && dims !== 4) {
      return `BatchNorm requires 2D or 4D input, got ${dims}D`
    }
  }

  // Merge blocks (concat, add) have special handling in store
  // They're always valid from a type perspective
  
  return undefined // Connection is valid
}

/**
 * Check if a block type allows multiple inputs
 */
export function allowsMultipleInputs(blockType: string): boolean {
  return blockType === 'concat' || blockType === 'add'
}
