from rest_framework import serializers
from .models import Project, ModelArchitecture, Block, Connection, GroupBlockDefinition


class GroupBlockDefinitionSerializer(serializers.ModelSerializer):
    """
    Serializer for GroupBlockDefinition model
    
    Handles validation of internal structure to ensure data integrity
    and prevent malformed group definitions from corrupting the database.
    """
    internalNodes = serializers.ListField(
        child=serializers.DictField(),
        write_only=True,
        required=False,
        default=list
    )
    internalEdges = serializers.ListField(
        child=serializers.DictField(),
        write_only=True,
        required=False,
        default=list
    )
    portMappings = serializers.ListField(
        child=serializers.DictField(),
        write_only=True,
        required=False,
        default=list
    )
    
    class Meta:
        model = GroupBlockDefinition
        fields = [
            'id', 'name', 'description', 'category', 'color',
            'internalNodes', 'internalEdges', 'portMappings',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']
    
    def validate_internalNodes(self, value):
        """Validate internal nodes structure"""
        if not isinstance(value, list):
            raise serializers.ValidationError("internalNodes must be a list")
        
        for node in value:
            if not isinstance(node, dict):
                raise serializers.ValidationError("Each node must be a dictionary")
            if 'id' not in node:
                raise serializers.ValidationError("Each node must have an 'id' field")
        
        return value
    
    def validate_internalEdges(self, value):
        """Validate internal edges structure"""
        if not isinstance(value, list):
            raise serializers.ValidationError("internalEdges must be a list")
        
        for edge in value:
            if not isinstance(edge, dict):
                raise serializers.ValidationError("Each edge must be a dictionary")
            required_fields = ['id', 'source', 'target']
            for field in required_fields:
                if field not in edge:
                    raise serializers.ValidationError(
                        f"Each edge must have a '{field}' field"
                    )
        
        return value
    
    def validate_portMappings(self, value):
        """Validate port mappings structure"""
        if not isinstance(value, list):
            raise serializers.ValidationError("portMappings must be a list")
        
        for mapping in value:
            if not isinstance(mapping, dict):
                raise serializers.ValidationError("Each port mapping must be a dictionary")
            required_fields = ['type', 'externalPortLabel', 'internalNodeId']
            for field in required_fields:
                if field not in mapping:
                    raise serializers.ValidationError(
                        f"Each port mapping must have a '{field}' field"
                    )
            if mapping['type'] not in ['input', 'output']:
                raise serializers.ValidationError(
                    "Port mapping type must be 'input' or 'output'"
                )
        
        return value
    
    def validate(self, data):
        """Cross-field validation for internal structure consistency"""
        internal_nodes = data.get('internalNodes', [])
        internal_edges = data.get('internalEdges', [])
        port_mappings = data.get('portMappings', [])
        
        # Build node ID set for validation
        node_ids = {node['id'] for node in internal_nodes}
        
        # Validate edge references
        for edge in internal_edges:
            if edge['source'] not in node_ids:
                raise serializers.ValidationError(
                    f"Edge references non-existent source node: {edge['source']}"
                )
            if edge['target'] not in node_ids:
                raise serializers.ValidationError(
                    f"Edge references non-existent target node: {edge['target']}"
                )
        
        # Validate port mapping references
        for mapping in port_mappings:
            if mapping['internalNodeId'] not in node_ids:
                raise serializers.ValidationError(
                    f"Port mapping references non-existent node: {mapping['internalNodeId']}"
                )
        
        return data
    
    def create(self, validated_data):
        """Create group definition with validated internal structure"""
        internal_nodes = validated_data.pop('internalNodes', [])
        internal_edges = validated_data.pop('internalEdges', [])
        port_mappings = validated_data.pop('portMappings', [])
        
        validated_data['internal_structure'] = {
            'nodes': internal_nodes,
            'edges': internal_edges,
            'portMappings': port_mappings
        }
        
        return super().create(validated_data)
    
    def update(self, instance, validated_data):
        """Update group definition with validated internal structure"""
        internal_nodes = validated_data.pop('internalNodes', None)
        internal_edges = validated_data.pop('internalEdges', None)
        port_mappings = validated_data.pop('portMappings', None)
        
        # Update internal structure if any component is provided
        if any([internal_nodes is not None, internal_edges is not None, port_mappings is not None]):
            current_structure = instance.internal_structure or {}
            validated_data['internal_structure'] = {
                'nodes': internal_nodes if internal_nodes is not None else current_structure.get('nodes', []),
                'edges': internal_edges if internal_edges is not None else current_structure.get('edges', []),
                'portMappings': port_mappings if port_mappings is not None else current_structure.get('portMappings', [])
            }
        
        return super().update(instance, validated_data)
    
    def to_representation(self, instance):
        """Convert instance to dictionary for read operations"""
        representation = super().to_representation(instance)
        # Add internal structure fields for read operations
        representation['internalNodes'] = instance.internal_structure.get('nodes', [])
        representation['internalEdges'] = instance.internal_structure.get('edges', [])
        representation['portMappings'] = instance.internal_structure.get('portMappings', [])
        return representation


class BlockSerializer(serializers.ModelSerializer):
    """Serializer for Block model"""
    group_definition = GroupBlockDefinitionSerializer(read_only=True)
    instance_config_overrides = serializers.JSONField(required=False, allow_null=True)

    class Meta:
        model = Block
        fields = [
            'id', 'node_id', 'block_type', 'position_x', 'position_y',
            'config', 'input_shape', 'output_shape',
            'group_definition', 'is_expanded', 'repetition_metadata',
            'instance_config_overrides',
            'created_at'
        ]
        read_only_fields = ['id', 'created_at']


class ConnectionSerializer(serializers.ModelSerializer):
    """Serializer for Connection model"""
    source_node_id = serializers.CharField(source='source_block.node_id', read_only=True)
    target_node_id = serializers.CharField(source='target_block.node_id', read_only=True)
    
    class Meta:
        model = Connection
        fields = [
            'id', 'edge_id', 'source_node_id', 'target_node_id',
            'source_handle', 'target_handle', 'is_valid', 'created_at'
        ]
        read_only_fields = ['id', 'created_at']


class ModelArchitectureSerializer(serializers.ModelSerializer):
    """Serializer for ModelArchitecture model"""
    blocks = BlockSerializer(many=True, read_only=True)
    connections = ConnectionSerializer(many=True, read_only=True)
    
    class Meta:
        model = ModelArchitecture
        fields = [
            'id', 'canvas_state', 'is_valid', 'validation_errors',
            'blocks', 'connections', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']


class ProjectSerializer(serializers.ModelSerializer):
    """Serializer for Project model"""
    class Meta:
        model = Project
        fields = [
            'id', 'name', 'description', 'framework',
            'share_token', 'is_shared',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'share_token', 'is_shared', 'created_at', 'updated_at']


class ProjectDetailSerializer(serializers.ModelSerializer):
    """Detailed serializer for Project with architecture"""
    architecture = ModelArchitectureSerializer(read_only=True)
    
    class Meta:
        model = Project
        fields = [
            'id', 'name', 'description', 'framework',
            'share_token', 'is_shared',
            'architecture', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'share_token', 'is_shared', 'created_at', 'updated_at']


class SaveArchitectureSerializer(serializers.Serializer):
    """Serializer for saving architecture from frontend"""
    nodes = serializers.ListField(child=serializers.DictField())
    edges = serializers.ListField(child=serializers.DictField())
    groupDefinitions = serializers.ListField(child=serializers.DictField(), required=False, default=list)


class ValidationResponseSerializer(serializers.Serializer):
    """Serializer for validation response"""
    is_valid = serializers.BooleanField()
    errors = serializers.ListField(child=serializers.DictField(), required=False)
    warnings = serializers.ListField(child=serializers.DictField(), required=False)
    inferred_shapes = serializers.DictField(required=False)


class ExportRequestSerializer(serializers.Serializer):
    """Serializer for code export request"""
    nodes = serializers.ListField(child=serializers.DictField())
    edges = serializers.ListField(child=serializers.DictField())
    format = serializers.ChoiceField(choices=['pytorch', 'tensorflow', 'onnx'])
    include_training = serializers.BooleanField(default=True)
    include_requirements = serializers.BooleanField(default=True)


class ExportResponseSerializer(serializers.Serializer):
    """Serializer for code export response"""
    code = serializers.CharField()
    files = serializers.DictField(required=False)
    download_url = serializers.CharField(required=False)
