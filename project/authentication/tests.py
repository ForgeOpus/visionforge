"""
Unit tests for authentication app
"""
from django.test import TestCase, Client
from django.utils import timezone
from unittest.mock import patch, MagicMock
import json

from .models import User
from .firebase_auth import get_user_info_from_token


class UserModelTestCase(TestCase):
    """Test cases for User model"""

    def setUp(self):
        """Set up test fixtures"""
        self.user_data = {
            'firebase_uid': 'test_uid_123',
            'email': 'test@example.com',
            'display_name': 'Test User',
            'avatar_url': 'https://example.com/avatar.jpg',
            'auth_provider': 'google',
        }

    def test_create_user(self):
        """Test creating a new user"""
        user = User.objects.create(**self.user_data)

        self.assertEqual(user.firebase_uid, 'test_uid_123')
        self.assertEqual(user.email, 'test@example.com')
        self.assertEqual(user.display_name, 'Test User')
        self.assertEqual(user.tier, 'free')
        self.assertEqual(user.project_count, 0)
        self.assertIsNotNone(user.created_at)

    def test_increment_session(self):
        """Test incrementing user sessions"""
        user = User.objects.create(**self.user_data)
        initial_sessions = user.total_sessions

        user.increment_session()

        self.assertEqual(user.total_sessions, initial_sessions + 1)

    def test_mark_first_export(self):
        """Test marking first export"""
        user = User.objects.create(**self.user_data)

        self.assertIsNone(user.first_export_at)
        self.assertIsNone(user.days_to_first_export)

        user.mark_first_export()

        self.assertIsNotNone(user.first_export_at)
        self.assertIsNotNone(user.days_to_first_export)

    def test_update_activity(self):
        """Test updating user activity"""
        user = User.objects.create(**self.user_data)
        initial_time = user.total_time_spent_minutes

        user.update_activity(minutes=30)

        self.assertEqual(user.total_time_spent_minutes, initial_time + 30)


class UserInfoExtractionTestCase(TestCase):
    """Test cases for user info extraction from Firebase token"""

    def test_extract_google_user_info(self):
        """Test extracting user info from Google token"""
        decoded_token = {
            'uid': 'google_uid_123',
            'email': 'user@gmail.com',
            'name': 'Google User',
            'picture': 'https://example.com/photo.jpg',
            'firebase': {
                'sign_in_provider': 'google.com'
            }
        }

        user_info = get_user_info_from_token(decoded_token)

        self.assertEqual(user_info['firebase_uid'], 'google_uid_123')
        self.assertEqual(user_info['email'], 'user@gmail.com')
        self.assertEqual(user_info['display_name'], 'Google User')
        self.assertEqual(user_info['avatar_url'], 'https://example.com/photo.jpg')
        self.assertEqual(user_info['auth_provider'], 'google.com')

    def test_extract_github_user_info(self):
        """Test extracting user info from GitHub token"""
        decoded_token = {
            'uid': 'github_uid_456',
            'email': 'user@github.com',
            'name': 'GitHub User',
            'picture': 'https://avatars.githubusercontent.com/u/123',
            'firebase': {
                'sign_in_provider': 'github.com'
            }
        }

        user_info = get_user_info_from_token(decoded_token)

        self.assertEqual(user_info['firebase_uid'], 'github_uid_456')
        self.assertEqual(user_info['auth_provider'], 'github.com')
