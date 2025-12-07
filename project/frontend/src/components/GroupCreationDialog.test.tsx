import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import GroupCreationDialog from './GroupCreationDialog'
import { useModelBuilderStore } from '@/lib/store'
import { Node, Edge } from '@xyflow/react'
import { BlockData } from '@/lib/types'

// Mock the store
vi.mock('@/lib/store', () => ({
  useModelBuilderStore: vi.fn()
}))

// Mock the node registry
vi.mock('@/lib/nodes/registry', () => ({
  getNodeDefinition: vi.fn((blockType) => {
    if (blockType === 'conv2d') {
      return {
        getInputPorts: () => [{ id: 'input', label: 'Input', semantic: 'data' }],
        getOutputPorts: () => [{ id: 'output', label: 'Output', semantic: 'data' }]
      }
    }
    if (blockType === 'linear') {
      return {
        getInputPorts: () => [{ id: 'input', label: 'Input', semantic: 'data' }],
        getOutputPorts: () => [{ id: 'output', label: 'Output', semantic: 'data' }]
      }
    }
    return null
  }),
  BackendFramework: {
    PyTorch: 'pytorch'
  }
}))

// Mock blockValidation
vi.mock('@/lib/blockValidation', () => ({
  validateConnectivity: vi.fn(() => ({ isValid: true, errors: [] })),
  detectCycles: vi.fn(() => ({ isValid: true, errors: [] })),
  validateBlockName: vi.fn((name: string) => {
    if (!name) return { isValid: false, errors: ['Block name is required'] }
    if (name.length > 50) return { isValid: false, errors: ['Block name must be 50 characters or less'] }
    return { isValid: true, errors: [] }
  })
}))

describe('GroupCreationDialog - Port Configuration', () => {
  const mockNodes: Node<BlockData>[] = [
    {
      id: 'node1',
      type: 'custom',
      position: { x: 0, y: 0 },
      data: {
        blockType: 'conv2d',
        label: 'Conv2D',
        config: {},
        category: 'basic'
      }
    },
    {
      id: 'node2',
      type: 'custom',
      position: { x: 100, y: 0 },
      data: {
        blockType: 'linear',
        label: 'Linear',
        config: {},
        category: 'basic'
      }
    }
  ]

  const mockEdges: Edge[] = [
    {
      id: 'edge1',
      source: 'node1',
      target: 'node2',
      sourceHandle: 'output',
      targetHandle: 'input'
    }
  ]

  const mockOnSave = vi.fn()
  const mockOnClose = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
    ;(useModelBuilderStore as any).mockImplementation((selector: any) => {
      const state = {
        nodes: mockNodes,
        edges: mockEdges,
        currentProject: { framework: 'pytorch' },
        groupDefinitions: new Map()
      }
      return selector ? selector(state) : state
    })
  })

  it('should display comprehensive port selection UI in step 2', async () => {
    render(
      <GroupCreationDialog
        isOpen={true}
        onClose={mockOnClose}
        onSave={mockOnSave}
        selectedNodeIds={['node1', 'node2']}
      />
    )

    // Fill in name and proceed to step 2
    const nameInput = screen.getByLabelText(/Block Name/i)
    fireEvent.change(nameInput, { target: { value: 'TestBlock' } })
    
    const nextButton = screen.getByText(/Next: Select Ports/i)
    fireEvent.click(nextButton)

    // Verify step 2 is displayed
    await waitFor(() => {
      expect(screen.getByText(/Input Ports/i)).toBeInTheDocument()
      expect(screen.getByText(/Output Ports/i)).toBeInTheDocument()
    })
  })

  it('should display all available input and output ports from internal layers', async () => {
    render(
      <GroupCreationDialog
        isOpen={true}
        onClose={mockOnClose}
        onSave={mockOnSave}
        selectedNodeIds={['node1', 'node2']}
      />
    )

    // Navigate to step 2
    const nameInput = screen.getByLabelText(/Block Name/i)
    fireEvent.change(nameInput, { target: { value: 'TestBlock' } })
    
    const nextButton = screen.getByText(/Next: Select Ports/i)
    fireEvent.click(nextButton)

    // Verify ports are displayed
    await waitFor(() => {
      // Should show Conv2D and Linear nodes
      expect(screen.getByText('Conv2D')).toBeInTheDocument()
      expect(screen.getByText('Linear')).toBeInTheDocument()
    })
  })

  it('should allow port selection and deselection', async () => {
    render(
      <GroupCreationDialog
        isOpen={true}
        onClose={mockOnClose}
        onSave={mockOnSave}
        selectedNodeIds={['node1', 'node2']}
      />
    )

    // Navigate to step 2
    const nameInput = screen.getByLabelText(/Block Name/i)
    fireEvent.change(nameInput, { target: { value: 'TestBlock' } })
    
    const nextButton = screen.getByText(/Next: Select Ports/i)
    fireEvent.click(nextButton)

    await waitFor(() => {
      expect(screen.getByText(/Input Ports/i)).toBeInTheDocument()
    })

    // Find checkboxes
    const checkboxes = screen.getAllByRole('checkbox')
    expect(checkboxes.length).toBeGreaterThan(0)

    // Toggle a checkbox
    const firstCheckbox = checkboxes[0]
    const initialChecked = firstCheckbox.getAttribute('data-state') === 'checked'
    
    fireEvent.click(firstCheckbox)
    
    // Verify state changed
    await waitFor(() => {
      const newState = firstCheckbox.getAttribute('data-state')
      expect(newState).not.toBe(initialChecked ? 'checked' : 'unchecked')
    })
  })

  it('should provide custom label editing for selected ports', async () => {
    render(
      <GroupCreationDialog
        isOpen={true}
        onClose={mockOnClose}
        onSave={mockOnSave}
        selectedNodeIds={['node1', 'node2']}
      />
    )

    // Navigate to step 2
    const nameInput = screen.getByLabelText(/Block Name/i)
    fireEvent.change(nameInput, { target: { value: 'TestBlock' } })
    
    const nextButton = screen.getByText(/Next: Select Ports/i)
    fireEvent.click(nextButton)

    await waitFor(() => {
      expect(screen.getByText(/Input Ports/i)).toBeInTheDocument()
    })

    // Find a checkbox and select it
    const checkboxes = screen.getAllByRole('checkbox')
    const firstCheckbox = checkboxes[0]
    
    // If not checked, check it
    if (firstCheckbox.getAttribute('data-state') !== 'checked') {
      fireEvent.click(firstCheckbox)
    }

    // Look for label input field
    await waitFor(() => {
      const labelInputs = screen.getAllByPlaceholderText(/External port label/i)
      expect(labelInputs.length).toBeGreaterThan(0)
    })
  })

  it('should validate that at least one port is exposed before allowing creation', async () => {
    // Mock edges with no external connections
    ;(useModelBuilderStore as any).mockImplementation((selector: any) => {
      const state = {
        nodes: mockNodes,
        edges: mockEdges,
        currentProject: { framework: 'pytorch' },
        groupDefinitions: new Map()
      }
      return selector ? selector(state) : state
    })
    
    render(
      <GroupCreationDialog
        isOpen={true}
        onClose={mockOnClose}
        onSave={mockOnSave}
        selectedNodeIds={['node1', 'node2']}
      />
    )

    // Navigate to step 2
    const nameInput = screen.getByLabelText(/Block Name/i)
    fireEvent.change(nameInput, { target: { value: 'TestBlock' } })
    
    const nextButton = screen.getByText(/Next: Select Ports/i)
    fireEvent.click(nextButton)

    await waitFor(() => {
      expect(screen.getByText(/Input Ports/i)).toBeInTheDocument()
    })

    // Deselect all ports
    const checkboxes = screen.getAllByRole('checkbox')
    for (const checkbox of checkboxes) {
      if (checkbox.getAttribute('data-state') === 'checked') {
        fireEvent.click(checkbox)
      }
    }

    // Try to create block
    const createButton = screen.getByText(/Create Block/i)
    fireEvent.click(createButton)

    // Should show validation error
    await waitFor(() => {
      expect(screen.getByText(/At least one port must be exposed/i)).toBeInTheDocument()
    })

    // onSave should not be called
    expect(mockOnSave).not.toHaveBeenCalled()
  })

  it('should mark ports with external connections as "External"', async () => {
    // Add external edge
    const edgesWithExternal: Edge[] = [
      ...mockEdges,
      {
        id: 'external1',
        source: 'external-node',
        target: 'node1',
        sourceHandle: 'output',
        targetHandle: 'input'
      }
    ]
    
    ;(useModelBuilderStore as any).mockImplementation((selector: any) => {
      const state = {
        nodes: [...mockNodes, {
          id: 'external-node',
          type: 'custom',
          position: { x: -100, y: 0 },
          data: { blockType: 'input', label: 'Input', config: {}, category: 'basic' }
        }],
        edges: edgesWithExternal,
        currentProject: { framework: 'pytorch' },
        groupDefinitions: new Map()
      }
      return selector ? selector(state) : state
    })
    
    render(
      <GroupCreationDialog
        isOpen={true}
        onClose={mockOnClose}
        onSave={mockOnSave}
        selectedNodeIds={['node1', 'node2']}
      />
    )

    // Navigate to step 2
    const nameInput = screen.getByLabelText(/Block Name/i)
    fireEvent.change(nameInput, { target: { value: 'TestBlock' } })
    
    const nextButton = screen.getByText(/Next: Select Ports/i)
    fireEvent.click(nextButton)

    // Verify "External" badge is shown
    await waitFor(() => {
      expect(screen.getByText('External')).toBeInTheDocument()
    })
  })

  it('should call onSave with correct port mappings configuration', async () => {
    render(
      <GroupCreationDialog
        isOpen={true}
        onClose={mockOnClose}
        onSave={mockOnSave}
        selectedNodeIds={['node1', 'node2']}
      />
    )

    // Fill in name
    const nameInput = screen.getByLabelText(/Block Name/i)
    fireEvent.change(nameInput, { target: { value: 'TestBlock' } })
    
    // Navigate to step 2
    const nextButton = screen.getByText(/Next: Select Ports/i)
    fireEvent.click(nextButton)

    await waitFor(() => {
      expect(screen.getByText(/Input Ports/i)).toBeInTheDocument()
    })

    // Ensure at least one port is selected
    const checkboxes = screen.getAllByRole('checkbox')
    if (checkboxes[0].getAttribute('data-state') !== 'checked') {
      fireEvent.click(checkboxes[0])
    }

    // Create block
    const createButton = screen.getByText(/Create Block/i)
    fireEvent.click(createButton)

    // Verify onSave was called with correct structure
    await waitFor(() => {
      expect(mockOnSave).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'TestBlock',
          description: '',
          category: expect.any(String),
          color: expect.any(String),
          portMappings: expect.any(Array)
        })
      )
    })
  })
})
