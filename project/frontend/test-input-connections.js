/**
 * Test: Input Node Connection Rules
 * Verifies that Input nodes can receive connections from DataLoader but not other nodes
 */

import { getNodeDefinition, BackendFramework } from '../src/lib/nodes/registry'

console.log('Testing Input Node Connection Rules...\n')

const inputNode = getNodeDefinition('input', BackendFramework.PyTorch)
if (!inputNode) {
  console.error('❌ Failed to get Input node definition')
  process.exit(1)
}

// Test 1: DataLoader → Input (should be allowed)
console.log('Test 1: DataLoader → Input')
const result1 = inputNode.validateIncomingConnection('dataloader', { dims: [1, 3, 224, 224] }, {})
if (result1 === undefined) {
  console.log('✅ PASS: DataLoader can connect to Input')
} else {
  console.log(`❌ FAIL: ${result1}`)
}

// Test 2: Conv2D → Input (should be rejected)
console.log('\nTest 2: Conv2D → Input')
const result2 = inputNode.validateIncomingConnection('conv2d', { dims: [1, 64, 112, 112] }, {})
if (result2 && result2.includes('data source')) {
  console.log('✅ PASS: Conv2D correctly rejected from Input')
} else {
  console.log(`❌ FAIL: Expected rejection, got: ${result2}`)
}

// Test 3: Linear → Input (should be rejected)
console.log('\nTest 3: Linear → Input')
const result3 = inputNode.validateIncomingConnection('linear', { dims: [1, 512] }, {})
if (result3 && result3.includes('data source')) {
  console.log('✅ PASS: Linear correctly rejected from Input')
} else {
  console.log(`❌ FAIL: Expected rejection, got: ${result3}`)
}

// Test 4: DataLoader cannot receive connections
console.log('\nTest 4: Input → DataLoader (DataLoader validation)')
const dataloaderNode = getNodeDefinition('dataloader', BackendFramework.PyTorch)
if (!dataloaderNode) {
  console.error('❌ Failed to get DataLoader node definition')
  process.exit(1)
}

const result4 = dataloaderNode.validateIncomingConnection('input', { dims: [1, 3, 224, 224] }, {})
if (result4 && result4.includes('cannot receive connections')) {
  console.log('✅ PASS: DataLoader correctly rejects all incoming connections')
} else {
  console.log(`❌ FAIL: Expected rejection, got: ${result4}`)
}

console.log('\n✅ All tests passed!')
