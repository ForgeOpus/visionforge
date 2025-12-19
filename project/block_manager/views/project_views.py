from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.shortcuts import get_object_or_404

from block_manager.models import Project, ModelArchitecture
from block_manager.serializers import (
    ProjectSerializer,
    ProjectDetailSerializer,
)


class ProjectViewSet(viewsets.ModelViewSet):
    """
    ViewSet for Project CRUD operations
    """
    queryset = Project.objects.all()
    serializer_class = ProjectSerializer

    def get_queryset(self):
        """
        Filter projects by authenticated user.
        Anonymous users see all projects (for backward compatibility).
        """
        queryset = super().get_queryset()

        # Check if user is authenticated via Firebase
        if hasattr(self.request, 'firebase_user') and self.request.firebase_user:
            # Show only user's own projects
            return queryset.filter(user=self.request.firebase_user)

        # For guests/anonymous users, show empty list (they must login to see projects)
        return queryset.none()

    def get_serializer_class(self):
        """Use detailed serializer for retrieve action"""
        if self.action == 'retrieve':
            return ProjectDetailSerializer
        return ProjectSerializer

    def create(self, request, *args, **kwargs):
        """
        Create a new project
        Automatically creates an empty ModelArchitecture
        Links project to authenticated user and increments their project count
        """
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        # Save project and link to authenticated user if present
        project = serializer.save()

        # Link to authenticated user
        if hasattr(request, 'firebase_user') and request.firebase_user:
            project.user = request.firebase_user
            project.save()

            # Increment user's project count
            request.firebase_user.increment_project_count()

        # Create associated architecture
        ModelArchitecture.objects.create(project=project)

        headers = self.get_success_headers(serializer.data)
        return Response(
            serializer.data,
            status=status.HTTP_201_CREATED,
            headers=headers
        )

    def list(self, request, *args, **kwargs):
        """List all projects"""
        queryset = self.filter_queryset(self.get_queryset())
        serializer = self.get_serializer(queryset, many=True)
        return Response({
            'projects': serializer.data
        })

    def retrieve(self, request, *args, **kwargs):
        """Get a single project with full architecture details"""
        instance = self.get_object()
        serializer = self.get_serializer(instance)
        return Response(serializer.data)

    def update(self, request, *args, **kwargs):
        """Update project metadata"""
        partial = kwargs.pop('partial', False)
        instance = self.get_object()
        serializer = self.get_serializer(instance, data=request.data, partial=partial)
        serializer.is_valid(raise_exception=True)
        self.perform_update(serializer)
        return Response(serializer.data)

    def destroy(self, request, *args, **kwargs):
        """Delete a project and all associated data"""
        instance = self.get_object()

        # Decrement user's project count if project has an owner
        if instance.user:
            instance.user.decrement_project_count()

        self.perform_destroy(instance)
        return Response(status=status.HTTP_204_NO_CONTENT)
