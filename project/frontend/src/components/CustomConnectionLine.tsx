import { ConnectionLineComponentProps } from '@xyflow/react'
import { useModelBuilderStore } from '@/lib/store'
import { validateBlockConnection } from '@/lib/blockDefinitions'

export default function CustomConnectionLine({
  fromX,
  fromY,
  toX,
  toY,
  fromNode,
  fromHandle
}: ConnectionLineComponentProps) {
  const { nodes } = useModelBuilderStore()
  
  // Get source node
  const sourceNode = fromNode ? nodes.find(n => n.id === fromNode.id) : null
  
  // Calculate position for tooltip
  const midX = (fromX + toX) / 2
  const midY = (fromY + toY) / 2
  
  // Default to valid (green) connection line
  let strokeColor = 'var(--color-accent)'
  let errorMessage: string | null = null
  
  // Check if we're hovering over a target node
  const targetNode = nodes.find(n => {
    const nodeEl = document.querySelector(`[data-id="${n.id}"]`)
    if (!nodeEl) return false
    const rect = nodeEl.getBoundingClientRect()
    return (
      toX >= rect.left &&
      toX <= rect.right &&
      toY >= rect.top &&
      toY <= rect.bottom
    )
  })
  
  // Validate connection if we have both nodes
  if (sourceNode && targetNode) {
    const validationError = validateBlockConnection(
      sourceNode.data.blockType,
      targetNode.data.blockType,
      sourceNode.data.outputShape
    )
    
    if (validationError) {
      strokeColor = 'var(--color-destructive)'
      errorMessage = validationError
    }
  }
  
  return (
    <g>
      <path
        d={`M ${fromX} ${fromY} C ${fromX + 50} ${fromY}, ${toX - 50} ${toY}, ${toX} ${toY}`}
        fill="none"
        stroke={strokeColor}
        strokeWidth={3}
        strokeDasharray="5,5"
        className="animated-dash"
      />
      
      {/* Circle at the end */}
      <circle
        cx={toX}
        cy={toY}
        r={5}
        fill={strokeColor}
        stroke="white"
        strokeWidth={2}
      />
      
      {/* Error tooltip */}
      {errorMessage && (
        <g transform={`translate(${midX}, ${midY - 30})`}>
          <rect
            x={-100}
            y={-20}
            width={200}
            height={40}
            rx={4}
            fill="var(--color-destructive)"
            opacity={0.95}
          />
          <text
            x={0}
            y={0}
            textAnchor="middle"
            dominantBaseline="middle"
            fill="white"
            fontSize={11}
            fontWeight="500"
          >
            <tspan x={0} dy={-5}>❌ Connection Invalid</tspan>
            <tspan x={0} dy={15} fontSize={9}>
              {errorMessage.length > 40 
                ? errorMessage.substring(0, 40) + '...' 
                : errorMessage}
            </tspan>
          </text>
        </g>
      )}
    </g>
  )
}
