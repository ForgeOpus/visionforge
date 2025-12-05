from rest_framework import serializers
from .models import Project, ModelArchitecture, Block, Connection, GroupBlockDefinition


class GroupBlockDefinitionSerializer(serializers.ModelSerializer):
    """Serializer for GroupBlockDefinition model"""
    internalNodes = serializers.SerializerMethodField()
    internalEdges = serializers.SerializerMethodField()
    portMappings = serializers.SerializerMethodField()
    
    class Meta:
        model = GroupBlockDefinition
        fields = [
            'id', 'name', 'description', 'category', 'color',
            'internalNodes', 'internalEdges', 'portMappings',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']
    
    def get_internalNodes(self, obj):
        return obj.internal_structure.get('nodes', [])
    
    def get_internalEdges(self, obj):
        return obj.internal_structure.get('edges', [])
    
    def get_portMappings(self, obj):
        return obj.internal_structure.get('portMappings', [])


class BlockSerializer(serializers.ModelSerializer):
    """Serializer for Block model"""
    group_definition = GroupBlockDefinitionSerializer(read_only=True)

    class Meta:
        model = Block
        fields = [
            'id', 'node_id', 'block_type', 'position_x', 'position_y',
            'config', 'input_shape', 'output_shape',
            'group_definition', 'is_expanded', 'repetition_metadata',
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
            'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']


class ProjectDetailSerializer(serializers.ModelSerializer):
    """Detailed serializer for Project with architecture"""
    architecture = ModelArchitectureSerializer(read_only=True)
    
    class Meta:
        model = Project
        fields = [
            'id', 'name', 'description', 'framework',
            'architecture', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']


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
