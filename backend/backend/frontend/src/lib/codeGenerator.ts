import { Node, Edge } from '@xyflow/react'
import { BlockData } from './types'

export function generatePyTorchCode(
  nodes: Node<BlockData>[],
  edges: Edge[],
  projectName: string
): { model: string; train: string; config: string } {
  const inputNodes = nodes.filter((n) => n.data.blockType === 'input')
  if (inputNodes.length === 0) {
    throw new Error('No input node found')
  }

  const sortedNodes = topologicalSort(nodes, edges)
  const layers = generatePyTorchLayers(sortedNodes, edges)
  const forwardPass = generatePyTorchForward(sortedNodes, edges)

  const modelCode = `import torch
import torch.nn as nn
import torch.nn.functional as F

class ${toClassName(projectName)}(nn.Module):
    def __init__(self):
        super(${toClassName(projectName)}, self).__init__()
        
${layers.map((l) => `        ${l}`).join('\n')}
    
    def forward(self, x):
${forwardPass.map((l) => `        ${l}`).join('\n')}
        return x

def create_model():
    """Create and return the model instance."""
    return ${toClassName(projectName)}()

if __name__ == '__main__':
    model = create_model()
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
`

  const trainCode = `import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import create_model

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}')
    
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, accuracy

def main():
    # TODO: Replace with your actual dataset
    # train_dataset = YourDataset(train=True)
    # val_dataset = YourDataset(train=False)
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    model = create_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 10
    
    for epoch in range(num_epochs):
        print(f'\\nEpoch {epoch + 1}/{num_epochs}')
        
        # TODO: Uncomment when you have your dataset
        # train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        # val_loss, val_acc = validate(model, val_loader, criterion, device)
        # print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f'checkpoint_epoch_{epoch}.pt')

if __name__ == '__main__':
    main()
`

  const configCode = `# Training Configuration

BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
WEIGHT_DECAY = 0.0001

# Model Configuration
${inputNodes[0].data.outputShape
  ? `INPUT_SHAPE = ${JSON.stringify(inputNodes[0].data.outputShape.dims)}`
  : '# INPUT_SHAPE = [batch, channels, height, width]'}

# Optimizer
OPTIMIZER = 'adam'  # Options: 'adam', 'sgd', 'adamw'
MOMENTUM = 0.9  # For SGD

# Learning Rate Scheduler
USE_SCHEDULER = True
SCHEDULER_TYPE = 'step'  # Options: 'step', 'cosine', 'exponential'
STEP_SIZE = 5
GAMMA = 0.1

# Device
USE_CUDA = True
`

  return { model: modelCode, train: trainCode, config: configCode }
}

function generatePyTorchLayers(nodes: Node<BlockData>[], edges: Edge[]): string[] {
  const layers: string[] = []
  
  nodes.forEach((node, idx) => {
    const config = node.data.config
    const layerName = `layer_${idx}`
    
    switch (node.data.blockType) {
      case 'linear':
        if (config.out_features && node.data.inputShape) {
          const inFeatures = node.data.inputShape.dims[1]
          layers.push(
            `self.${layerName} = nn.Linear(${inFeatures}, ${config.out_features}, bias=${config.bias ?? true})`
          )
        }
        break
      
      case 'conv2d':
        if (config.out_channels && node.data.inputShape) {
          const inChannels = node.data.inputShape.dims[1]
          layers.push(
            `self.${layerName} = nn.Conv2d(${inChannels}, ${config.out_channels}, ` +
            `kernel_size=${config.kernel_size ?? 3}, stride=${config.stride ?? 1}, padding=${config.padding ?? 0})`
          )
        }
        break
      
      case 'batchnorm':
        if (node.data.inputShape) {
          const features = node.data.inputShape.dims[1]
          layers.push(`self.${layerName} = nn.BatchNorm2d(${features})`)
        }
        break
      
      case 'dropout':
        layers.push(`self.${layerName} = nn.Dropout(p=${config.rate ?? 0.5})`)
        break
      
      case 'maxpool':
        layers.push(
          `self.${layerName} = nn.MaxPool2d(kernel_size=${config.kernel_size ?? 2}, stride=${config.stride ?? 2})`
        )
        break
      
      case 'attention':
        if (node.data.inputShape) {
          const embedDim = node.data.inputShape.dims[2]
          layers.push(
            `self.${layerName} = nn.MultiheadAttention(embed_dim=${embedDim}, ` +
            `num_heads=${config.num_heads ?? 8}, dropout=${config.dropout ?? 0.1})`
          )
        }
        break
    }
  })
  
  return layers
}

function generatePyTorchForward(nodes: Node<BlockData>[], edges: Edge[]): string[] {
  const forward: string[] = []
  const nodeMap = new Map(nodes.map((n, idx) => [n.id, { node: n, idx }]))
  const varMap = new Map<string, string>()
  
  const sortedNodes = topologicalSort(nodes, edges)
  
  sortedNodes.forEach((node, idx) => {
    const layerName = `layer_${nodeMap.get(node.id)?.idx}`
    const varName = idx === 0 ? 'x' : `x${idx}`
    
    const incomingEdges = edges.filter((e) => e.target === node.id)
    const prevVarName = incomingEdges.length > 0 && incomingEdges[0].source
      ? varMap.get(incomingEdges[0].source) || 'x'
      : 'x'
    
    switch (node.data.blockType) {
      case 'input':
        varMap.set(node.id, 'x')
        break
      
      case 'linear':
      case 'conv2d':
      case 'batchnorm':
      case 'dropout':
      case 'maxpool':
        forward.push(`${varName} = self.${layerName}(${prevVarName})`)
        varMap.set(node.id, varName)
        break
      
      case 'relu':
        forward.push(`${varName} = F.relu(${prevVarName})`)
        varMap.set(node.id, varName)
        break
      
      case 'softmax':
        const dim = node.data.config.dim ?? -1
        forward.push(`${varName} = F.softmax(${prevVarName}, dim=${dim})`)
        varMap.set(node.id, varName)
        break
      
      case 'flatten':
        const startDim = node.data.config.start_dim ?? 1
        forward.push(`${varName} = torch.flatten(${prevVarName}, start_dim=${startDim})`)
        varMap.set(node.id, varName)
        break
      
      case 'attention':
        forward.push(`${varName}, _ = self.${layerName}(${prevVarName}, ${prevVarName}, ${prevVarName})`)
        varMap.set(node.id, varName)
        break
    }
  })
  
  const lastVar = varMap.get(sortedNodes[sortedNodes.length - 1]?.id) || 'x'
  if (lastVar !== 'x') {
    forward.push(`x = ${lastVar}`)
  }
  
  return forward
}

function topologicalSort(nodes: Node<BlockData>[], edges: Edge[]): Node<BlockData>[] {
  const adjList = new Map<string, string[]>()
  const inDegree = new Map<string, number>()
  
  nodes.forEach((node) => {
    adjList.set(node.id, [])
    inDegree.set(node.id, 0)
  })
  
  edges.forEach((edge) => {
    adjList.get(edge.source)?.push(edge.target)
    inDegree.set(edge.target, (inDegree.get(edge.target) || 0) + 1)
  })
  
  const queue: Node<BlockData>[] = []
  nodes.forEach((node) => {
    if (inDegree.get(node.id) === 0) {
      queue.push(node)
    }
  })
  
  const sorted: Node<BlockData>[] = []
  
  while (queue.length > 0) {
    const node = queue.shift()!
    sorted.push(node)
    
    const neighbors = adjList.get(node.id) || []
    neighbors.forEach((neighborId) => {
      const degree = (inDegree.get(neighborId) || 1) - 1
      inDegree.set(neighborId, degree)
      
      if (degree === 0) {
        const neighborNode = nodes.find((n) => n.id === neighborId)
        if (neighborNode) {
          queue.push(neighborNode)
        }
      }
    })
  }
  
  return sorted
}

function toClassName(name: string): string {
  return name
    .split(/[\s_-]+/)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
    .join('')
}
