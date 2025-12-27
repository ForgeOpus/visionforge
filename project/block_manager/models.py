from django.db import models
import uuid
import json
from .validators import (
    validate_canvas_state,
    validate_block_config,
    validate_group_internal_structure,
    validate_shape_data,
)


class Project(models.Model):
    """Represents a model building project"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        'authentication.User',
        on_delete=models.CASCADE,
        related_name='projects',
        null=True,  # Allow null for existing projects during migration
        blank=True,
        db_index=True,  # Index for faster queries
        help_text='User who owns this project'
    )
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, default='')
    framework = models.CharField(
        max_length=20,
        choices=[('pytorch', 'PyTorch'), ('tensorflow', 'TensorFlow')],
        default='pytorch'
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-updated_at']

    def __str__(self):
        return self.name
    
class ModelArchitecture(models.Model):
    """Stores the architecture graph for a project"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    project = models.OneToOneField(
        Project,
        on_delete=models.CASCADE,
        related_name='architecture'
    )
    canvas_state = models.JSONField(default=dict, blank=True, validators=[validate_canvas_state])
    is_valid = models.BooleanField(default=False)
    validation_errors = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Architecture for {self.project.name}"


class GroupBlockDefinition(models.Model):
    """Project-specific block template for group blocks"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    project = models.ForeignKey(
        Project,
        on_delete=models.CASCADE,
        related_name='group_definitions'
    )
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, default='')
    category = models.CharField(max_length=50, default='utility')
    color = models.CharField(max_length=50, default='#9333ea')

    # Serialized structure: {nodes, edges, portMappings}
    internal_structure = models.JSONField(default=dict, blank=True, validators=[validate_group_internal_structure])

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-updated_at']
        unique_together = ['project', 'name']

    def __str__(self):
        return f"{self.name} ({self.project.name})"


class Block(models.Model):
    """Represents a single block/layer in the architecture"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    architecture = models.ForeignKey(
        ModelArchitecture,
        on_delete=models.CASCADE,
        related_name='blocks'
    )
    # Store the frontend node ID for reference
    node_id = models.CharField(max_length=255)
    block_type = models.CharField(max_length=50)
    position_x = models.FloatField(default=0)
    position_y = models.FloatField(default=0)
    config = models.JSONField(default=dict, blank=True, validators=[validate_block_config])
    input_shape = models.JSONField(null=True, blank=True, validators=[validate_shape_data])
    output_shape = models.JSONField(null=True, blank=True, validators=[validate_shape_data])

    # Group block fields
    group_definition = models.ForeignKey(
        GroupBlockDefinition,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='instances'
    )
    is_expanded = models.BooleanField(default=False)
    repetition_metadata = models.JSONField(null=True, blank=True)
    instance_config_overrides = models.JSONField(null=True, blank=True, default=dict)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']

    def __str__(self):
        return f"{self.block_type} ({self.node_id})"


class Connection(models.Model):
    """Represents a connection/edge between blocks"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    architecture = models.ForeignKey(
        ModelArchitecture,
        on_delete=models.CASCADE,
        related_name='connections'
    )
    # Store the frontend edge ID for reference
    edge_id = models.CharField(max_length=255)
    source_block = models.ForeignKey(
        Block,
        on_delete=models.CASCADE,
        related_name='outgoing_connections'
    )
    target_block = models.ForeignKey(
        Block,
        on_delete=models.CASCADE,
        related_name='incoming_connections'
    )
    source_handle = models.CharField(max_length=50, blank=True, default='')
    target_handle = models.CharField(max_length=50, blank=True, default='')
    is_valid = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']

    def __str__(self):
        return f"{self.source_block.node_id} -> {self.target_block.node_id}"
