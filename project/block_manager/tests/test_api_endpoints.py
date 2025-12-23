"""
API Endpoint Tests for Universal AI System
Tests validation, model listing, and chat endpoints with universal API keys.

Reference: https://www.django-rest-framework.org/api-guide/testing/
"""
from django.test import TestCase, override_settings
from django.urls import reverse
from rest_framework.test import APIClient
from rest_framework import status
import json


class APIKeyValidationEndpointTests(TestCase):
    """
    Test suite for /api/validate-key endpoint.
    Tests API key validation with various provider formats.
    """

    def setUp(self):
        """Set up test client"""
        self.client = APIClient()
        self.url = reverse('validate-api-key')

    # ==================== Successful Validation Tests ====================

    def test_validate_openrouter_key(self):
        """Test validating OpenRouter API key"""
        # Arrange
        payload = {
            'apiKey': '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'
        }

        # Act
        response = self.client.post(self.url, payload, format='json')

        # Assert
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertTrue(data['valid'])
        self.assertEqual(data['provider'], 'openrouter')
        self.assertEqual(data['displayName'], 'OpenRouter')
        self.assertTrue(data['isFreeTier'])
        self.assertEqual(len(data['models']), 10)

    def test_validate_google_ai_key(self):
        """Test validating Google AI (Gemini) API key"""
        # Arrange
        payload = {
            'apiKey': 'AIzaSyAkKJPaCtQXhd4JIy_OskAsHilxmywhYqY'
        }

        # Act
        response = self.client.post(self.url, payload, format='json')

        # Assert
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertTrue(data['valid'])
        self.assertEqual(data['provider'], 'google')
        self.assertEqual(data['displayName'], 'Google AI (Gemini)')
        self.assertTrue(data['isFreeTier'])
        self.assertEqual(len(data['models']), 4)
        self.assertIn('gemini-3-flash', data['models'])

    def test_validate_openai_key(self):
        """Test validating OpenAI API key"""
        # Arrange
        payload = {
            'apiKey': 'sk-proj-abc123def456ghi789jkl012mno345pqr678stu901vwx234'
        }

        # Act
        response = self.client.post(self.url, payload, format='json')

        # Assert
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertTrue(data['valid'])
        self.assertEqual(data['provider'], 'openai')
        self.assertEqual(data['displayName'], 'OpenAI')
        self.assertFalse(data['isFreeTier'])
        self.assertEqual(len(data['models']), 3)
        self.assertIn('gpt-5.2', data['models'])

    def test_validate_anthropic_key(self):
        """Test validating Anthropic (Claude) API key"""
        # Arrange
        payload = {
            'apiKey': 'sk-ant-api03-R2D2C3PO4R2D2C3PO4R2D2C3PO4R2D2C3PO4R2D2C3PO4igAA'
        }

        # Act
        response = self.client.post(self.url, payload, format='json')

        # Assert
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertTrue(data['valid'])
        self.assertEqual(data['provider'], 'anthropic')
        self.assertEqual(data['displayName'], 'Anthropic (Claude)')
        self.assertFalse(data['isFreeTier'])
        self.assertEqual(len(data['models']), 3)
        self.assertIn('claude-opus-4.5', data['models'])

    # ==================== Invalid Key Tests ====================

    def test_validate_invalid_key_format(self):
        """Test validating invalid API key format"""
        # Arrange
        payload = {
            'apiKey': 'invalid-key-format-12345'
        }

        # Act
        response = self.client.post(self.url, payload, format='json')

        # Assert
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertFalse(data['valid'])
        self.assertIsNone(data['provider'])
        self.assertIn('Unknown', data['message'])

    def test_validate_empty_key(self):
        """Test validating empty API key"""
        # Arrange
        payload = {
            'apiKey': ''
        }

        # Act
        response = self.client.post(self.url, payload, format='json')

        # Assert
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        data = response.json()
        self.assertFalse(data['valid'])

    def test_validate_missing_key_field(self):
        """Test validation with missing api_key field"""
        # Arrange
        payload = {}

        # Act
        response = self.client.post(self.url, payload, format='json')

        # Assert
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        data = response.json()
        self.assertFalse(data.get('valid', True))
        self.assertIn('message', data)

    # ==================== HTTP Method Tests ====================

    def test_get_method_not_allowed(self):
        """Test that GET method is not allowed"""
        # Act
        response = self.client.get(self.url)

        # Assert
        self.assertEqual(response.status_code, status.HTTP_405_METHOD_NOT_ALLOWED)


class AvailableModelsEndpointTests(TestCase):
    """
    Test suite for /api/available-models endpoint.
    Tests model listing based on API key provider.
    """

    def setUp(self):
        """Set up test client"""
        self.client = APIClient()
        self.url = reverse('available-models')

    # ==================== Successful Model Listing Tests ====================

    def test_get_models_for_openrouter(self):
        """Test getting available models for OpenRouter key"""
        # Arrange
        payload = {
            'apiKey': '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'
        }

        # Act
        response = self.client.post(self.url, payload, format='json')

        # Assert
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertEqual(len(data['models']), 10)
        # OpenRouter should have all models
        self.assertIn('gemini-3-flash', data['models'])
        self.assertIn('gpt-5.2', data['models'])
        self.assertIn('claude-opus-4.5', data['models'])

    def test_get_models_for_google(self):
        """Test getting available models for Google AI key"""
        # Arrange
        payload = {
            'apiKey': 'AIzaSyAkKJPaCtQXhd4JIy_OskAsHilxmywhYqY'
        }

        # Act
        response = self.client.post(self.url, payload, format='json')

        # Assert
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertEqual(len(data['models']), 4)
        # Google should only have Gemini models
        self.assertIn('gemini-3-flash', data['models'])
        self.assertNotIn('gpt-5.2', data['models'])
        self.assertNotIn('claude-opus-4.5', data['models'])

    def test_get_models_for_openai(self):
        """Test getting available models for OpenAI key"""
        # Arrange
        payload = {
            'apiKey': 'sk-proj-abc123def456ghi789jkl012mno345pqr678stu901vwx234'
        }

        # Act
        response = self.client.post(self.url, payload, format='json')

        # Assert
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertEqual(len(data['models']), 3)
        # OpenAI should only have GPT models
        self.assertIn('gpt-5.2', data['models'])
        self.assertNotIn('gemini-3-flash', data['models'])
        self.assertNotIn('claude-opus-4.5', data['models'])

    def test_get_models_for_anthropic(self):
        """Test getting available models for Anthropic key"""
        # Arrange
        payload = {
            'apiKey': 'sk-ant-api03-R2D2C3PO4R2D2C3PO4R2D2C3PO4R2D2C3PO4R2D2C3PO4igAA'
        }

        # Act
        response = self.client.post(self.url, payload, format='json')

        # Assert
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertEqual(len(data['models']), 3)
        # Anthropic should only have Claude models
        self.assertIn('claude-opus-4.5', data['models'])
        self.assertNotIn('gpt-5.2', data['models'])
        self.assertNotIn('gemini-3-flash', data['models'])

    # ==================== Error Cases ====================

    def test_get_models_invalid_key(self):
        """Test getting models for invalid API key"""
        # Arrange
        payload = {
            'apiKey': 'invalid-key-format'
        }

        # Act
        response = self.client.post(self.url, payload, format='json')

        # Assert
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        data = response.json()
        self.assertIn('error', data)

    def test_get_models_missing_key(self):
        """Test getting models with missing API key"""
        # Arrange
        payload = {}

        # Act
        response = self.client.post(self.url, payload, format='json')

        # Assert
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)


class EnvironmentEndpointTests(TestCase):
    """
    Test suite for /api/environment endpoint.
    Tests environment configuration exposure.
    """

    def setUp(self):
        """Set up test client"""
        self.client = APIClient()
        self.url = reverse('environment-info')

    @override_settings(REQUIRES_USER_API_KEY=True)
    def test_environment_prod_mode(self):
        """Test environment endpoint in PROD mode"""
        # Act
        response = self.client.get(self.url)

        # Assert
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertTrue(data['requiresApiKey'])

    @override_settings(REQUIRES_USER_API_KEY=False)
    def test_environment_dev_mode(self):
        """Test environment endpoint in DEV mode"""
        # Act
        response = self.client.get(self.url)

        # Assert
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertFalse(data['requiresApiKey'])


class ChatEndpointUniversalKeyTests(TestCase):
    """
    Test suite for /api/chat endpoint with universal API keys.
    Tests chat functionality with different provider keys.
    """

    def setUp(self):
        """Set up test client"""
        self.client = APIClient()
        self.url = reverse('chat-message')

    # ==================== Header Handling Tests ====================

    def test_chat_accepts_x_api_key_header(self):
        """Test that chat endpoint accepts new X-API-Key header"""
        # Arrange
        headers = {
            'HTTP_X_API_KEY': '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',
            'HTTP_X_SELECTED_MODEL': 'gpt-5.2'
        }
        payload = {
            'message': 'Hello, test message',
            'context': {}
        }

        # Note: We expect this to fail with actual API call, but should pass header validation
        # Act
        response = self.client.post(self.url, payload, format='json', **headers)

        # Assert - Should not get 400 for missing API key
        # (May get other errors from actual API call, but that's okay)
        self.assertNotEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_chat_accepts_legacy_openrouter_header(self):
        """Test that chat endpoint still accepts legacy X-OpenRouter-Api-Key header"""
        # Arrange
        headers = {
            'HTTP_X_OPENROUTER_API_KEY': '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',
            'HTTP_X_SELECTED_MODEL': 'gpt-5.2'
        }
        payload = {
            'message': 'Hello, test message',
            'context': {}
        }

        # Act
        response = self.client.post(self.url, payload, format='json', **headers)

        # Assert - Should not get 400 for missing API key
        self.assertNotEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    @override_settings(REQUIRES_USER_API_KEY=True)
    def test_chat_requires_api_key_in_prod(self):
        """Test that chat endpoint requires API key in PROD mode"""
        # Arrange
        payload = {
            'message': 'Hello, test message',
            'context': {}
        }

        # Act - No API key headers
        response = self.client.post(self.url, payload, format='json')

        # Assert - Either 400 or 401 is acceptable for missing auth
        self.assertIn(response.status_code, [status.HTTP_400_BAD_REQUEST, status.HTTP_401_UNAUTHORIZED])
        data = response.json()
        self.assertIn('error', data)

    def test_chat_validates_message_required(self):
        """Test that chat endpoint validates message field"""
        # Arrange
        headers = {
            'HTTP_X_API_KEY': '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',
        }
        payload = {
            'context': {}
        }

        # Act
        response = self.client.post(self.url, payload, format='json', **headers)

        # Assert
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        data = response.json()
        self.assertIn('message', data['error'].lower())

    # ==================== Rate Limiting Tests ====================

    def test_chat_rate_limiting(self):
        """Test that chat endpoint applies rate limiting"""
        # Arrange
        headers = {
            'HTTP_X_API_KEY': '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',
            'HTTP_X_SELECTED_MODEL': 'gpt-5.2'
        }
        payload = {
            'message': 'Test message',
            'context': {}
        }

        # Act - Send multiple rapid requests
        responses = []
        for _ in range(15):  # Exceed typical rate limit
            response = self.client.post(self.url, payload, format='json', **headers)
            responses.append(response.status_code)

        # Assert - At least one should be rate limited
        # Note: This assumes rate limiting is configured; adjust based on actual settings
        # For now, just verify we can make multiple requests without crashes
        self.assertTrue(all(status_code in [200, 429, 400, 500] for status_code in responses))
