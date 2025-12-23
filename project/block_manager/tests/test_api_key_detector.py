"""
Unit tests for API Key Detector - Universal API Key System
Tests provider detection, validation, and model availability.

Reference: https://www.django-rest-framework.org/api-guide/testing/
"""
from django.test import TestCase
from block_manager.services.api_key_detector import APIKeyDetector


class APIKeyDetectorTestCase(TestCase):
    """
    Test suite for APIKeyDetector class.
    Follows AAA pattern (Arrange, Act, Assert) for each test.
    """

    # ==================== Provider Detection Tests ====================

    def test_detect_openrouter_key(self):
        """Test detection of valid OpenRouter API key format"""
        # Arrange
        test_key = "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"

        # Act
        result = APIKeyDetector.detect_provider(test_key)

        # Assert
        self.assertEqual(result, 'openrouter')

    def test_detect_google_ai_key(self):
        """Test detection of valid Google AI (Gemini) API key format"""
        # Arrange - Google AI keys start with AIza and are 39 chars total
        test_key = "AIzaSyAkKJPaCtQXhd4JIy_OskAsHilxmywhYqY"  # Exactly 39 chars

        # Act
        result = APIKeyDetector.detect_provider(test_key)

        # Assert
        self.assertEqual(result, 'google')

    def test_detect_openai_key_proj_format(self):
        """Test detection of OpenAI project API key (sk-proj- prefix)"""
        # Arrange
        test_key = "sk-proj-abc123def456ghi789jkl012mno345pqr678stu901vwx234"

        # Act
        result = APIKeyDetector.detect_provider(test_key)

        # Assert
        self.assertEqual(result, 'openai')

    def test_detect_openai_key_svcacct_format(self):
        """Test detection of OpenAI service account key (sk-svcacct- prefix)"""
        # Arrange
        test_key = "sk-svcacct-abc123def456ghi789jkl012mno345pqr678"

        # Act
        result = APIKeyDetector.detect_provider(test_key)

        # Assert
        self.assertEqual(result, 'openai')

    def test_detect_openai_legacy_key(self):
        """Test detection of legacy OpenAI key (sk- prefix only)"""
        # Arrange
        test_key = "sk-abc123def456ghi789jkl012mno345pqr678stu901"

        # Act
        result = APIKeyDetector.detect_provider(test_key)

        # Assert
        self.assertEqual(result, 'openai')

    def test_detect_anthropic_key(self):
        """Test detection of Anthropic Claude API key (sk-ant-api03- prefix)"""
        # Arrange
        test_key = "sk-ant-api03-R2D2C3PO4R2D2C3PO4R2D2C3PO4R2D2C3PO4R2D2C3PO4igAA"

        # Act
        result = APIKeyDetector.detect_provider(test_key)

        # Assert
        self.assertEqual(result, 'anthropic')

    def test_detect_invalid_key_returns_none(self):
        """Test that invalid key formats return None"""
        # Arrange
        invalid_keys = [
            "invalid-key-format",
            "random-text-123",
            "",
            "   ",
            None
        ]

        # Act & Assert
        for key in invalid_keys:
            result = APIKeyDetector.detect_provider(key)
            self.assertIsNone(result, f"Expected None for key: {key}")

    def test_detect_google_key_wrong_length(self):
        """Test that Google AI key with wrong length is not detected"""
        # Arrange - Google AI keys must be exactly 39 chars
        test_key = "AIzaShortKey"  # Too short

        # Act
        result = APIKeyDetector.detect_provider(test_key)

        # Assert
        self.assertIsNone(result)

    def test_detect_key_with_whitespace(self):
        """Test that keys with surrounding whitespace are handled"""
        # Arrange
        test_key = "  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  "

        # Act
        result = APIKeyDetector.detect_provider(test_key)

        # Assert
        self.assertEqual(result, 'openrouter')

    # ==================== Provider Info Tests ====================

    def test_get_provider_info_openrouter(self):
        """Test retrieving provider info for OpenRouter"""
        # Act
        info = APIKeyDetector.get_provider_info('openrouter')

        # Assert
        self.assertIsNotNone(info)
        self.assertEqual(info.name, 'openrouter')
        self.assertEqual(info.display_name, 'OpenRouter')
        self.assertEqual(info.models_count, 10)
        self.assertTrue(info.is_free_tier)

    def test_get_provider_info_google(self):
        """Test retrieving provider info for Google AI"""
        # Act
        info = APIKeyDetector.get_provider_info('google')

        # Assert
        self.assertIsNotNone(info)
        self.assertEqual(info.name, 'google')
        self.assertEqual(info.display_name, 'Google AI (Gemini)')
        self.assertEqual(info.models_count, 4)
        self.assertTrue(info.is_free_tier)

    def test_get_provider_info_openai(self):
        """Test retrieving provider info for OpenAI"""
        # Act
        info = APIKeyDetector.get_provider_info('openai')

        # Assert
        self.assertIsNotNone(info)
        self.assertEqual(info.name, 'openai')
        self.assertEqual(info.display_name, 'OpenAI')
        self.assertEqual(info.models_count, 3)
        self.assertFalse(info.is_free_tier)

    def test_get_provider_info_anthropic(self):
        """Test retrieving provider info for Anthropic"""
        # Act
        info = APIKeyDetector.get_provider_info('anthropic')

        # Assert
        self.assertIsNotNone(info)
        self.assertEqual(info.name, 'anthropic')
        self.assertEqual(info.display_name, 'Anthropic (Claude)')
        self.assertEqual(info.models_count, 3)
        self.assertFalse(info.is_free_tier)

    def test_get_provider_info_invalid(self):
        """Test that invalid provider returns None"""
        # Act
        info = APIKeyDetector.get_provider_info('invalid_provider')

        # Assert
        self.assertIsNone(info)

    # ==================== Available Models Tests ====================

    def test_get_available_models_openrouter(self):
        """Test getting available models for OpenRouter"""
        # Act
        models = APIKeyDetector.get_available_models('openrouter')

        # Assert
        self.assertEqual(len(models), 10)
        self.assertIn('gemini-3-flash', models)
        self.assertIn('gpt-5.2', models)
        self.assertIn('claude-opus-4.5', models)

    def test_get_available_models_google(self):
        """Test getting available models for Google AI"""
        # Act
        models = APIKeyDetector.get_available_models('google')

        # Assert
        self.assertEqual(len(models), 4)
        self.assertIn('gemini-3-flash', models)
        self.assertIn('gemini-3-pro', models)
        self.assertNotIn('gpt-5.2', models)  # Should not have GPT models

    def test_get_available_models_openai(self):
        """Test getting available models for OpenAI"""
        # Act
        models = APIKeyDetector.get_available_models('openai')

        # Assert
        self.assertEqual(len(models), 3)
        self.assertIn('gpt-5.2', models)
        self.assertIn('gpt-4o', models)
        self.assertNotIn('gemini-3-flash', models)  # Should not have Gemini models

    def test_get_available_models_anthropic(self):
        """Test getting available models for Anthropic"""
        # Act
        models = APIKeyDetector.get_available_models('anthropic')

        # Assert
        self.assertEqual(len(models), 3)
        self.assertIn('claude-opus-4.5', models)
        self.assertIn('claude-sonnet-4.5', models)
        self.assertNotIn('gpt-5.2', models)  # Should not have GPT models

    def test_is_model_available_valid(self):
        """Test checking if a model is available for a provider"""
        # Assert - OpenRouter has all models
        self.assertTrue(APIKeyDetector.is_model_available('gemini-3-flash', 'openrouter'))
        self.assertTrue(APIKeyDetector.is_model_available('gpt-5.2', 'openrouter'))

        # Assert - Google only has Gemini
        self.assertTrue(APIKeyDetector.is_model_available('gemini-3-flash', 'google'))
        self.assertFalse(APIKeyDetector.is_model_available('gpt-5.2', 'google'))

        # Assert - OpenAI only has GPT
        self.assertTrue(APIKeyDetector.is_model_available('gpt-5.2', 'openai'))
        self.assertFalse(APIKeyDetector.is_model_available('gemini-3-flash', 'openai'))

    # ==================== Validation Tests ====================

    def test_validate_key_format_valid_openrouter(self):
        """Test validation of valid OpenRouter key"""
        # Arrange
        test_key = "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"

        # Act
        result = APIKeyDetector.validate_key_format(test_key, 'openrouter')

        # Assert
        self.assertTrue(result['valid'])
        self.assertEqual(result['provider'], 'openrouter')
        self.assertIn('OpenRouter', result['message'])

    def test_validate_key_format_mismatch(self):
        """Test validation when key doesn't match expected provider"""
        # Arrange
        google_key = "AIzaSyAkKJPaCtQXhd4JIy_OskAsHilxmywhYqY"

        # Act
        result = APIKeyDetector.validate_key_format(google_key, 'openai')

        # Assert
        self.assertFalse(result['valid'])
        self.assertEqual(result['provider'], 'google')
        self.assertIn('Google AI', result['message'])

    def test_validate_key_format_unknown(self):
        """Test validation of unknown key format"""
        # Arrange
        invalid_key = "not-a-valid-api-key"

        # Act
        result = APIKeyDetector.validate_key_format(invalid_key, None)

        # Assert
        self.assertFalse(result['valid'])
        self.assertIsNone(result['provider'])
        self.assertIn('Unknown', result['message'])
        self.assertGreater(len(result['suggestions']), 0)
