"""
Management command to test Firebase authentication and Oracle database connectivity.

Usage:
    python manage.py test_firebase_auth
"""
from django.core.management.base import BaseCommand
from django.utils import timezone
from django.db import connections
from django.db.utils import OperationalError
from authentication.models import User
import uuid


class Command(BaseCommand):
    help = 'Test Firebase authentication and Oracle database connectivity'

    def handle(self, *args, **options):
        """
        Run test suite for Firebase authentication setup.
        """
        self.stdout.write(self.style.SUCCESS('\n' + '='*70))
        self.stdout.write(self.style.SUCCESS('Firebase Authentication & Oracle Database Test Suite'))
        self.stdout.write(self.style.SUCCESS('='*70 + '\n'))

        # Test 1: Oracle Database Connection
        self.stdout.write(self.style.WARNING('Test 1: Testing Oracle database connection...'))
        if not self.test_oracle_connection():
            return

        # Test 2: Create Test User
        self.stdout.write(self.style.WARNING('\nTest 2: Creating test user...'))
        test_user = self.create_test_user()
        if not test_user:
            return

        # Test 3: Verify User Creation
        self.stdout.write(self.style.WARNING('\nTest 3: Verifying user was created in Oracle...'))
        if not self.verify_user_exists(test_user.firebase_uid):
            return

        # Test 4: Update User Data
        self.stdout.write(self.style.WARNING('\nTest 4: Testing user data updates...'))
        if not self.test_user_updates(test_user):
            return

        # Test 5: Query User Data
        self.stdout.write(self.style.WARNING('\nTest 5: Querying user data...'))
        if not self.test_user_query(test_user.firebase_uid):
            return

        # Test 6: Cleanup
        self.stdout.write(self.style.WARNING('\nTest 6: Cleaning up test data...'))
        if not self.cleanup_test_user(test_user.firebase_uid):
            return

        self.stdout.write(self.style.SUCCESS('\n' + '='*70))
        self.stdout.write(self.style.SUCCESS('All tests passed successfully! ✓'))
        self.stdout.write(self.style.SUCCESS('='*70 + '\n'))

    def test_oracle_connection(self):
        """Test Oracle database connection."""
        try:
            # Try to connect to Oracle database
            conn = connections['oracle']
            conn.ensure_connection()

            # Test basic query
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM DUAL")
            result = cursor.fetchone()

            if result and result[0] == 1:
                self.stdout.write(self.style.SUCCESS('✓ Oracle database connection successful'))
                self.stdout.write(self.style.SUCCESS(f'  Database: {conn.settings_dict["NAME"]}'))
                self.stdout.write(self.style.SUCCESS(f'  User: {conn.settings_dict["USER"]}'))
                return True
            else:
                self.stdout.write(self.style.ERROR('✗ Oracle connection test query failed'))
                return False

        except OperationalError as e:
            self.stdout.write(self.style.ERROR(f'✗ Oracle database connection failed: {str(e)}'))
            self.stdout.write(self.style.ERROR('\nPlease check:'))
            self.stdout.write(self.style.ERROR('  1. Oracle credentials in .env file'))
            self.stdout.write(self.style.ERROR('  2. Wallet location is correct'))
            self.stdout.write(self.style.ERROR('  3. DSN (visionforgedb_medium) is valid'))
            return False
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'✗ Unexpected error: {str(e)}'))
            return False

    def create_test_user(self):
        """Create a test user in the database."""
        try:
            test_uid = f'test_user_{uuid.uuid4().hex[:8]}'
            test_email = f'test_{uuid.uuid4().hex[:8]}@visionforge.test'

            user = User.objects.create(
                firebase_uid=test_uid,
                email=test_email,
                display_name='Test User',
                avatar_url='https://example.com/avatar.jpg',
                auth_provider='google',
                tier='free',
                project_count=0,
                total_sessions=0,
                total_time_spent_minutes=0,
            )

            self.stdout.write(self.style.SUCCESS(f'✓ Test user created successfully'))
            self.stdout.write(self.style.SUCCESS(f'  UID: {user.firebase_uid}'))
            self.stdout.write(self.style.SUCCESS(f'  Email: {user.email}'))
            self.stdout.write(self.style.SUCCESS(f'  Provider: {user.auth_provider}'))
            return user

        except Exception as e:
            self.stdout.write(self.style.ERROR(f'✗ Failed to create test user: {str(e)}'))
            return None

    def verify_user_exists(self, firebase_uid):
        """Verify user was created in Oracle database."""
        try:
            user = User.objects.using('oracle').get(firebase_uid=firebase_uid)
            self.stdout.write(self.style.SUCCESS('✓ User found in Oracle database'))
            self.stdout.write(self.style.SUCCESS(f'  Retrieved: {user.email}'))
            return True
        except User.DoesNotExist:
            self.stdout.write(self.style.ERROR('✗ User not found in Oracle database'))
            return False
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'✗ Error verifying user: {str(e)}'))
            return False

    def test_user_updates(self, user):
        """Test updating user data."""
        try:
            # Test update_last_login
            user.update_last_login()
            self.stdout.write(self.style.SUCCESS('✓ Updated last_login_at'))

            # Test increment_session
            original_sessions = user.total_sessions
            user.increment_session()
            user.refresh_from_db()
            if user.total_sessions == original_sessions + 1:
                self.stdout.write(self.style.SUCCESS(f'✓ Incremented session count: {user.total_sessions}'))
            else:
                self.stdout.write(self.style.ERROR('✗ Session increment failed'))
                return False

            # Test add_time_spent
            original_time = user.total_time_spent_minutes
            user.add_time_spent(15)
            user.refresh_from_db()
            if user.total_time_spent_minutes == original_time + 15:
                self.stdout.write(self.style.SUCCESS(f'✓ Added time spent: {user.total_time_spent_minutes} minutes'))
            else:
                self.stdout.write(self.style.ERROR('✗ Time spent update failed'))
                return False

            # Test mark_first_model_created
            user.mark_first_model_created()
            user.refresh_from_db()
            if user.first_model_created_at:
                self.stdout.write(self.style.SUCCESS('✓ Marked first model created'))
            else:
                self.stdout.write(self.style.ERROR('✗ First model creation failed'))
                return False

            # Test mark_first_export
            user.mark_first_export()
            user.refresh_from_db()
            if user.first_export_at and user.days_to_first_export is not None:
                self.stdout.write(self.style.SUCCESS(f'✓ Marked first export (days: {user.days_to_first_export})'))
            else:
                self.stdout.write(self.style.ERROR('✗ First export marking failed'))
                return False

            return True

        except Exception as e:
            self.stdout.write(self.style.ERROR(f'✗ Update test failed: {str(e)}'))
            return False

    def test_user_query(self, firebase_uid):
        """Test querying user data."""
        try:
            user = User.objects.get(firebase_uid=firebase_uid)

            self.stdout.write(self.style.SUCCESS('✓ Successfully queried user data:'))
            self.stdout.write(self.style.SUCCESS(f'  Firebase UID: {user.firebase_uid}'))
            self.stdout.write(self.style.SUCCESS(f'  Email: {user.email}'))
            self.stdout.write(self.style.SUCCESS(f'  Display Name: {user.display_name}'))
            self.stdout.write(self.style.SUCCESS(f'  Provider: {user.auth_provider}'))
            self.stdout.write(self.style.SUCCESS(f'  Tier: {user.tier}'))
            self.stdout.write(self.style.SUCCESS(f'  Total Sessions: {user.total_sessions}'))
            self.stdout.write(self.style.SUCCESS(f'  Time Spent: {user.total_time_spent_minutes} minutes'))
            self.stdout.write(self.style.SUCCESS(f'  Created: {user.created_at}'))

            return True

        except Exception as e:
            self.stdout.write(self.style.ERROR(f'✗ Query test failed: {str(e)}'))
            return False

    def cleanup_test_user(self, firebase_uid):
        """Delete the test user."""
        try:
            user = User.objects.get(firebase_uid=firebase_uid)
            user.delete()

            # Verify deletion
            try:
                User.objects.get(firebase_uid=firebase_uid)
                self.stdout.write(self.style.ERROR('✗ User deletion verification failed'))
                return False
            except User.DoesNotExist:
                self.stdout.write(self.style.SUCCESS('✓ Test user deleted successfully'))
                return True

        except Exception as e:
            self.stdout.write(self.style.ERROR(f'✗ Cleanup failed: {str(e)}'))
            return False
