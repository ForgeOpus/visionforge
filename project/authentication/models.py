from django.db import models
from django.utils import timezone


class User(models.Model):
    """
    Firebase-authenticated user model for VisionForge.
    Stores user profile, subscription, and engagement data in Oracle database.
    """

    # Core Authentication
    firebase_uid = models.CharField(max_length=128, unique=True, primary_key=True,
                                   help_text="Firebase user ID (primary key)")
    email = models.EmailField(unique=True, help_text="User email from Firebase")
    display_name = models.CharField(max_length=255, blank=True, null=True,
                                   help_text="Display name from OAuth provider")
    avatar_url = models.URLField(blank=True, null=True,
                                help_text="Profile photo URL from OAuth provider")
    auth_provider = models.CharField(max_length=20,
                                    choices=[('google', 'Google'), ('github', 'GitHub')],
                                    help_text="OAuth provider used for authentication")

    # Subscription
    tier = models.CharField(max_length=10,
                          choices=[('free', 'Free'), ('pro', 'Pro')],
                          default='free',
                          help_text="User subscription tier")
    project_count = models.IntegerField(default=0,
                                       help_text="Number of projects created by user")

    # Engagement Signals
    created_at = models.DateTimeField(auto_now_add=True,
                                     help_text="Account creation timestamp")
    last_login_at = models.DateTimeField(null=True, blank=True,
                                        help_text="Last login timestamp")
    total_sessions = models.IntegerField(default=0,
                                        help_text="Total number of user sessions")
    total_time_spent_minutes = models.IntegerField(default=0,
                                                   help_text="Total time spent in app (minutes)")

    # Conversion Funnel
    first_model_created_at = models.DateTimeField(null=True, blank=True,
                                                 help_text="Timestamp of first model creation")
    first_export_at = models.DateTimeField(null=True, blank=True,
                                          help_text="Timestamp of first export")
    days_to_first_export = models.IntegerField(null=True, blank=True,
                                              help_text="Days from signup to first export")

    class Meta:
        db_table = 'users'
        verbose_name = 'User'
        verbose_name_plural = 'Users'
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.display_name or self.email} ({self.auth_provider})"

    def update_last_login(self):
        """Update last login timestamp"""
        self.last_login_at = timezone.now()
        self.save(update_fields=['last_login_at'])

    def increment_session(self):
        """Increment total session count"""
        self.total_sessions += 1
        self.save(update_fields=['total_sessions'])

    def add_time_spent(self, minutes):
        """Add minutes to total time spent"""
        self.total_time_spent_minutes += minutes
        self.save(update_fields=['total_time_spent_minutes'])

    def mark_first_model_created(self):
        """Mark timestamp of first model creation"""
        if not self.first_model_created_at:
            self.first_model_created_at = timezone.now()
            self.save(update_fields=['first_model_created_at'])

    def mark_first_export(self):
        """Mark timestamp of first export and calculate days to export"""
        if not self.first_export_at:
            self.first_export_at = timezone.now()
            if self.created_at:
                delta = self.first_export_at - self.created_at
                self.days_to_first_export = delta.days
            self.save(update_fields=['first_export_at', 'days_to_first_export'])

    def increment_project_count(self):
        """Increment project count"""
        self.project_count += 1
        self.save(update_fields=['project_count'])
