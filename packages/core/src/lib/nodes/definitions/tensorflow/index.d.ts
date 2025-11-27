/**
 * TensorFlow node definitions - currently mirror PyTorch implementations
 * Future: Implement TensorFlow-specific behaviors where they diverge
 */
export { InputNode } from '../pytorch/input';
export { DataLoaderNode } from '../pytorch/dataloader';
export { OutputNode } from '../pytorch/output';
export { LossNode } from '../pytorch/loss';
export { EmptyNode } from '../pytorch/empty';
export { LinearNode } from '../pytorch/linear';
export { Conv2DNode } from '../pytorch/conv2d';
export { FlattenNode } from '../pytorch/flatten';
export { ReLUNode } from '../pytorch/relu';
export { DropoutNode } from '../pytorch/dropout';
export { BatchNormNode } from '../pytorch/batchnorm';
export { MaxPool2DNode } from '../pytorch/maxpool';
export { SoftmaxNode } from '../pytorch/softmax';
export { ConcatNode } from '../pytorch/concat';
export { AddNode } from '../pytorch/add';
export { AttentionNode } from '../pytorch/attention';
export { CustomNode } from '../pytorch/custom';
/**
 * Note: These are currently identical to PyTorch definitions.
 * As TensorFlow-specific requirements emerge, create separate implementations
 * in this directory that override the framework metadata and any divergent logic.
 */
//# sourceMappingURL=index.d.ts.map