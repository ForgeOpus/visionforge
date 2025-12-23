"""
Integration tests for Universal AI Factory
Tests service creation, provider detection, and model validation.

Reference: https://www.django-rest-framework.org/api-guide/testing/
"""
from django.test import TestCase, override_settings
from unittest.mock import patch, MagicMock
from block_manager.services.universal_ai_factory import UniversalAIFactory
from block_manager.services.openrouter_service import OpenRouterService
from block_manager.services.gemini_service import GeminiChatService
from block_manager.services.openai_service import OpenAIChatService
from block_manager.services.claude_service import ClaudeChatService


class UniversalAIFactoryTestCase(TestCase):
    """
    Integration test suite for UniversalAIFactory.
    Tests the complete flow of API key detection and service creation.
    """

    # ==================== Service Creation Tests ====================

    def test_create_openrouter_service(self):
        """Test creating OpenRouter service from valid key"""
        # Arrange
        api_key = "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"
        model = "gpt-5.2"

        # Act
        service = UniversalAIFactory.detect_and_create_service(
            api_key=api_key,
            model=model
        )

        # Assert
        self.assertIsInstance(service, OpenRouterService)

    def test_create_google_service(self):
        """Test creating Google AI service from valid key"""
        # Arrange
        api_key = "AIzaSyAkKJPaCtQXhd4JIy_OskAsHilxmywhYqY"
        model = "gemini-3-flash"

        # Act
        service = UniversalAIFactory.detect_and_create_service(
            api_key=api_key,
            model=model
        )

        # Assert
        self.assertIsInstance(service, GeminiChatService)

    def test_create_openai_service_proj_key(self):
        """Test creating OpenAI service from project API key"""
        # Arrange
        api_key = "sk-proj-abc123def456ghi789jkl012mno345pqr678stu901vwx234"
        model = "gpt-5.2"

        # Act
        service = UniversalAIFactory.detect_and_create_service(
            api_key=api_key,
            model=model
        )

        # Assert
        self.assertIsInstance(service, OpenAIChatService)

    def test_create_openai_service_svcacct_key(self):
        """Test creating OpenAI service from service account key"""
        # Arrange
        api_key = "sk-svcacct-abc123def456ghi789jkl012mno345pqr678"
        model = "gpt-4o"

        # Act
        service = UniversalAIFactory.detect_and_create_service(
            api_key=api_key,
            model=model
        )

        # Assert
        self.assertIsInstance(service, OpenAIChatService)

    def test_create_openai_service_legacy_key(self):
        """Test creating OpenAI service from legacy sk- key"""
        # Arrange
        api_key = "sk-abc123def456ghi789jkl012mno345pqr678stu901"
        model = "gpt-4o-mini"

        # Act
        service = UniversalAIFactory.detect_and_create_service(
            api_key=api_key,
            model=model
        )

        # Assert
        self.assertIsInstance(service, OpenAIChatService)

    def test_create_anthropic_service(self):
        """Test creating Anthropic service from valid key"""
        # Arrange
        api_key = "sk-ant-api03-R2D2C3PO4R2D2C3PO4R2D2C3PO4R2D2C3PO4R2D2C3PO4igAA"
        model = "claude-opus-4.5"

        # Act
        service = UniversalAIFactory.detect_and_create_service(
            api_key=api_key,
            model=model
        )

        # Assert
        self.assertIsInstance(service, ClaudeChatService)

    # ==================== Error Handling Tests ====================

    @override_settings(REQUIRES_USER_API_KEY=True)
    def test_missing_api_key_in_prod(self):
        """Test that missing API key raises error in PROD mode"""
        # Act & Assert
        with self.assertRaises(ValueError) as context:
            UniversalAIFactory.detect_and_create_service(api_key=None)

        self.assertIn("API key is required", str(context.exception))

    def test_invalid_api_key_format(self):
        """Test that invalid API key format raises error"""
        # Arrange
        invalid_key = "invalid-key-format-12345"

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            UniversalAIFactory.detect_and_create_service(api_key=invalid_key)

        self.assertIn("Unable to detect API provider", str(context.exception))

    def test_provider_hint_mismatch(self):
        """Test error when provider hint doesn't match detected provider"""
        # Arrange
        google_key = "AIzaSyAkKJPaCtQXhd4JIy_OskAsHilxmywhYqY"

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            UniversalAIFactory.detect_and_create_service(
                api_key=google_key,
                provider_hint="openai"
            )

        self.assertIn("appears to be for Google AI", str(context.exception))

    def test_invalid_model_for_provider(self):
        """Test error when requesting unavailable model for provider"""
        # Arrange - Google AI key but requesting GPT model
        google_key = "AIzaSyAkKJPaCtQXhd4JIy_OskAsHilxmywhYqY"

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            UniversalAIFactory.detect_and_create_service(
                api_key=google_key,
                model="gpt-5.2"  # GPT not available with Google AI key
            )

        self.assertIn("not available with Google AI", str(context.exception))

    # ==================== Model Validation Tests ====================

    def test_openrouter_accepts_all_models(self):
        """Test that OpenRouter key accepts any model"""
        # Arrange
        openrouter_key = "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"
        models_to_test = ["gemini-3-flash", "gpt-5.2", "claude-opus-4.5"]

        # Act & Assert - All should succeed
        for model in models_to_test:
            service = UniversalAIFactory.detect_and_create_service(
                api_key=openrouter_key,
                model=model
            )
            self.assertIsInstance(service, OpenRouterService)

    def test_google_only_accepts_gemini(self):
        """Test that Google AI key only accepts Gemini models"""
        # Arrange
        google_key = "AIzaSyAkKJPaCtQXhd4JIy_OskAsHilxmywhYqY"

        # Act & Assert - Gemini should work
        service = UniversalAIFactory.detect_and_create_service(
            api_key=google_key,
            model="gemini-3-flash"
        )
        self.assertIsInstance(service, GeminiChatService)

        # Act & Assert - GPT should fail
        with self.assertRaises(ValueError):
            UniversalAIFactory.detect_and_create_service(
                api_key=google_key,
                model="gpt-5.2"
            )

        # Act & Assert - Claude should fail
        with self.assertRaises(ValueError):
            UniversalAIFactory.detect_and_create_service(
                api_key=google_key,
                model="claude-opus-4.5"
            )

    def test_openai_only_accepts_gpt(self):
        """Test that OpenAI key only accepts GPT models"""
        # Arrange
        openai_key = "sk-proj-abc123def456ghi789jkl012mno345pqr678stu901vwx234"

        # Act & Assert - GPT should work
        service = UniversalAIFactory.detect_and_create_service(
            api_key=openai_key,
            model="gpt-5.2"
        )
        self.assertIsInstance(service, OpenAIChatService)

        # Act & Assert - Gemini should fail
        with self.assertRaises(ValueError):
            UniversalAIFactory.detect_and_create_service(
                api_key=openai_key,
                model="gemini-3-flash"
            )

    def test_anthropic_only_accepts_claude(self):
        """Test that Anthropic key only accepts Claude models"""
        # Arrange
        anthropic_key = "sk-ant-api03-R2D2C3PO4R2D2C3PO4R2D2C3PO4R2D2C3PO4R2D2C3PO4igAA"

        # Act & Assert - Claude should work
        service = UniversalAIFactory.detect_and_create_service(
            api_key=anthropic_key,
            model="claude-opus-4.5"
        )
        self.assertIsInstance(service, ClaudeChatService)

        # Act & Assert - GPT should fail
        with self.assertRaises(ValueError):
            UniversalAIFactory.detect_and_create_service(
                api_key=anthropic_key,
                model="gpt-5.2"
            )

    # ==================== Server-Side Keys Tests (DEV Mode) ====================

    @override_settings(REQUIRES_USER_API_KEY=False)
    @patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-openrouter-key'})
    def test_dev_mode_uses_openrouter_server_key(self):
        """Test that DEV mode uses server-side OpenRouter key"""
        # Act
        service = UniversalAIFactory.detect_and_create_service(
            api_key=None,
            model="gpt-5.2"
        )

        # Assert
        self.assertIsInstance(service, OpenRouterService)

    @override_settings(REQUIRES_USER_API_KEY=False)
    @patch.dict('os.environ', {'GEMINI_API_KEY': 'test-gemini-key'}, clear=True)
    def test_dev_mode_falls_back_to_gemini(self):
        """Test that DEV mode falls back to Gemini if no OpenRouter key"""
        # Act
        service = UniversalAIFactory.detect_and_create_service(api_key=None)

        # Assert
        self.assertIsInstance(service, GeminiChatService)

    @override_settings(REQUIRES_USER_API_KEY=False)
    @patch.dict('os.environ', {}, clear=True)
    def test_dev_mode_no_keys_raises_error(self):
        """Test that DEV mode raises error if no server keys configured"""
        # Act & Assert
        with self.assertRaises(ValueError) as context:
            UniversalAIFactory.detect_and_create_service(api_key=None)

        self.assertIn("No server-side API keys configured", str(context.exception))

    # ==================== Utility Function Tests ====================

    def test_get_provider_name_openrouter(self):
        """Test getting provider name for OpenRouter key"""
        # Arrange
        api_key = "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"

        # Act
        name = UniversalAIFactory.get_provider_name(api_key)

        # Assert
        self.assertEqual(name, 'OpenRouter')

    def test_get_provider_name_google(self):
        """Test getting provider name for Google AI key"""
        # Arrange
        api_key = "AIzaSyAkKJPaCtQXhd4JIy_OskAsHilxmywhYqY"

        # Act
        name = UniversalAIFactory.get_provider_name(api_key)

        # Assert
        self.assertEqual(name, 'Google AI (Gemini)')

    def test_get_provider_name_no_key(self):
        """Test getting provider name with no key returns default"""
        # Act
        name = UniversalAIFactory.get_provider_name(None)

        # Assert
        self.assertEqual(name, 'Universal AI (Multi-Provider)')

    def test_validate_api_key(self):
        """Test API key validation wrapper"""
        # Arrange
        valid_key = "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"

        # Act
        result = UniversalAIFactory.validate_api_key(valid_key)

        # Assert
        self.assertTrue(result['valid'])
        self.assertEqual(result['provider'], 'openrouter')

    def test_get_available_models_for_key(self):
        """Test getting available models for a given key"""
        # Arrange
        google_key = "AIzaSyAkKJPaCtQXhd4JIy_OskAsHilxmywhYqY"

        # Act
        models = UniversalAIFactory.get_available_models_for_key(google_key)

        # Assert
        self.assertIsInstance(models, list)
        self.assertGreater(len(models), 0)
        self.assertIn('gemini-3-flash', models)
        self.assertNotIn('gpt-5.2', models)  # Should not include GPT models

    def test_get_available_models_invalid_key(self):
        """Test getting models for invalid key returns empty list"""
        # Arrange
        invalid_key = "invalid-key"

        # Act
        models = UniversalAIFactory.get_available_models_for_key(invalid_key)

        # Assert
        self.assertEqual(models, [])

    # ==================== Environment Mode Tests ====================

    @patch.dict('os.environ', {'ENVIRONMENT': 'PROD'})
    def test_get_environment_mode_prod(self):
        """Test getting environment mode in PROD"""
        # Act
        mode = UniversalAIFactory.get_environment_mode()

        # Assert
        self.assertEqual(mode, 'PROD')

    @patch.dict('os.environ', {'ENVIRONMENT': 'DEV'})
    def test_get_environment_mode_dev(self):
        """Test getting environment mode in DEV"""
        # Act
        mode = UniversalAIFactory.get_environment_mode()

        # Assert
        self.assertEqual(mode, 'DEV')

    @patch.dict('os.environ', {}, clear=True)
    def test_get_environment_mode_default(self):
        """Test getting environment mode defaults to PROD"""
        # Act
        mode = UniversalAIFactory.get_environment_mode()

        # Assert
        self.assertEqual(mode, 'PROD')
