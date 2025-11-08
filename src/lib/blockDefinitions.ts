import { BlockDefinition, TensorShape, BlockConfig } from './types'

export const blockDefinitions: Record<string, BlockDefinition> = {
  input: {
    type: 'input',
    label: 'Input',
    category: 'input',
    color: 'var(--color-teal)',
    icon: 'ArrowDown',
    description: 'Define input tensor shape for any modality (text, image, audio, etc.)',
    configSchema: [
      {
        name: 'shape',
        label: 'Tensor Shape',
        type: 'text',
        default: '[1, 3, 224, 224]',
        required: true,
        placeholder: '[batch, channels, height, width]',
        description: 'Input tensor dimensions as JSON array. Examples: [1, 3, 224, 224] for images, [32, 512, 768] for text, [16, 1, 16000] for audio'
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
