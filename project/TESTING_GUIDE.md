# Universal API Key System - Testing Guide

This guide will help you test the new universal API key system that automatically detects and validates API keys from multiple providers (OpenRouter, Google AI, OpenAI, and Anthropic).

## Test Summary

- **Backend Unit Tests**: 69 tests - ALL PASSING ✅
- **Manual API Tests**: ALL PASSING ✅
- **Frontend Integration**: Ready for testing

## Prerequisites

1. Ensure the backend server is running:
   ```bash
   cd /Users/gunbirsingh/Desktop/Code\ folders/visionforge/project
   python3 manage.py runserver
   ```

2. Ensure the frontend dev server is running (if not already):
   ```bash
   cd /Users/gunbirsingh/Desktop/Code\ folders/visionforge/project/frontend
   npm run dev
   ```

## Manual Testing Checklist

### 1. API Key Validation Endpoint

Test that the system correctly detects and validates different API key formats:

#### Test OpenRouter Key
```bash
curl -X POST http://localhost:8000/api/v1/validate-key \
  -H "Content-Type: application/json" \
  -d '{"apiKey": "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"}'
```

**Expected Response:**
```json
{
  "valid": true,
  "provider": "openrouter",
  "displayName": "OpenRouter",
  "availableModels": 10,
  "models": ["gemini-3-flash", "gemini-3-pro", "gemini-2.5-flash", "gemini-2.5-pro", "gpt-5.2", "gpt-4o", "gpt-4o-mini", "claude-opus-4.5", "claude-sonnet-4.5", "claude-haiku-4.5"],
  "isFreeTier": true,
  "message": "Valid OpenRouter API key detected"
}
```

#### Test Google AI Key
```bash
curl -X POST http://localhost:8000/api/v1/validate-key \
  -H "Content-Type: application/json" \
  -d '{"apiKey": "AIzaSyAkKJPaCtQXhd4JIy_OskAsHilxmywhYqY"}'
```

**Expected Response:**
```json
{
  "valid": true,
  "provider": "google",
  "displayName": "Google AI (Gemini)",
  "availableModels": 4,
  "models": ["gemini-3-flash", "gemini-3-pro", "gemini-2.5-flash", "gemini-2.5-pro"],
  "isFreeTier": true,
  "message": "Valid Google AI (Gemini) API key detected"
}
```

#### Test OpenAI Key
```bash
curl -X POST http://localhost:8000/api/v1/validate-key \
  -H "Content-Type: application/json" \
  -d '{"apiKey": "sk-proj-abc123def456ghi789jkl012mno345pqr678stu901vwx234"}'
```

**Expected Response:**
```json
{
  "valid": true,
  "provider": "openai",
  "displayName": "OpenAI",
  "availableModels": 3,
  "models": ["gpt-5.2", "gpt-4o", "gpt-4o-mini"],
  "isFreeTier": false,
  "message": "Valid OpenAI API key detected"
}
```

#### Test Anthropic (Claude) Key
```bash
curl -X POST http://localhost:8000/api/v1/validate-key \
  -H "Content-Type: application/json" \
  -d '{"apiKey": "sk-ant-api03-R2D2C3PO4R2D2C3PO4R2D2C3PO4R2D2C3PO4R2D2C3PO4igAA"}'
```

**Expected Response:**
```json
{
  "valid": true,
  "provider": "anthropic",
  "displayName": "Anthropic (Claude)",
  "availableModels": 3,
  "models": ["claude-opus-4.5", "claude-sonnet-4.5", "claude-haiku-4.5"],
  "isFreeTier": false,
  "message": "Valid Anthropic (Claude) API key detected"
}
```

#### Test Invalid Key
```bash
curl -X POST http://localhost:8000/api/v1/validate-key \
  -H "Content-Type: application/json" \
  -d '{"apiKey": "invalid-key-format-12345"}'
```

**Expected Response:**
```json
{
  "valid": false,
  "provider": null,
  "displayName": null,
  "availableModels": 0,
  "models": [],
  "isFreeTier": false,
  "message": "Unknown API key format. Supported: OpenRouter (sk-or-v1-...), Google AI (AIza...), OpenAI (sk-proj-... or sk-...), Anthropic (sk-ant-api03-...)"
}
```

### 2. Available Models Endpoint

Test that the system returns correct models for each provider:

#### Test with Google AI Key
```bash
curl -X POST http://localhost:8000/api/v1/available-models \
  -H "Content-Type: application/json" \
  -d '{"apiKey": "AIzaSyAkKJPaCtQXhd4JIy_OskAsHilxmywhYqY"}'
```

**Expected Response:**
```json
{
  "provider": "google",
  "displayName": "Google AI (Gemini)",
  "models": ["gemini-3-flash", "gemini-3-pro", "gemini-2.5-flash", "gemini-2.5-pro"],
  "defaultModel": "gemini-3-flash",
  "isFreeTier": true
}
```

✅ Verify that only Gemini models are returned for Google AI keys
✅ Verify that only GPT models are returned for OpenAI keys (test with sk-proj- key)
✅ Verify that only Claude models are returned for Anthropic keys

### 3. Frontend UI Testing

#### Access the Application
1. Open your browser and navigate to: `http://localhost:5173`

#### Test the Universal API Key Modal

**Test 1: Opening the Modal**
1. Click on the API key button in the navigation bar
2. ✅ Verify the modal opens with title "API Key Setup"
3. ✅ Verify the modal shows instructions about supported providers

**Test 2: Real-time Key Validation**
1. Paste an OpenRouter key (starts with `sk-or-v1-`)
2. ✅ Verify "Detecting provider..." appears briefly
3. ✅ Verify green checkmark appears with "Valid OpenRouter API key detected"
4. ✅ Verify "Free Tier" badge is shown
5. ✅ Verify model dropdown shows 10 models (all providers)

**Test 3: Google AI Key**
1. Clear the input and paste a Google AI key (starts with `AIza`, 39 chars)
2. ✅ Verify detection shows "Valid Google AI (Gemini) API key detected"
3. ✅ Verify model dropdown shows only 4 Gemini models
4. ✅ Verify GPT and Claude models are disabled or hidden
5. ✅ Verify "Free Tier" badge is shown

**Test 4: OpenAI Key**
1. Clear and paste an OpenAI key (starts with `sk-proj-` or `sk-`)
2. ✅ Verify detection shows "Valid OpenAI API key detected"
3. ✅ Verify model dropdown shows only 3 GPT models
4. ✅ Verify Gemini and Claude models are disabled or hidden
5. ✅ Verify NO "Free Tier" badge (OpenAI is paid)

**Test 5: Anthropic Key**
1. Clear and paste an Anthropic key (starts with `sk-ant-api03-`)
2. ✅ Verify detection shows "Valid Anthropic (Claude) API key detected"
3. ✅ Verify model dropdown shows only 3 Claude models
4. ✅ Verify Gemini and GPT models are disabled or hidden
5. ✅ Verify NO "Free Tier" badge (Anthropic is paid)

**Test 6: Invalid Key**
1. Type an invalid key like "invalid-test-key"
2. ✅ Verify red X icon appears
3. ✅ Verify error message shows "Unknown API key format"
4. ✅ Verify "Save & Continue" button is disabled

**Test 7: Model Selection**
1. Enter a valid Google AI key
2. Select "Gemini 3 Flash" from the dropdown
3. Click "Save & Continue"
4. ✅ Verify modal closes
5. ✅ Verify the selected model is remembered on page refresh

**Test 8: Key Persistence**
1. Enter a valid API key and save
2. Refresh the page
3. Reopen the modal
4. ✅ Verify the key is still there (masked)
5. ✅ Verify the selected model is still selected

**Test 9: Clear Keys**
1. With a key saved, open the modal
2. Click "Clear Key" button
3. ✅ Verify the key is removed from storage
4. ✅ Verify modal is empty when reopened

### 4. Chat Integration Testing

**Note**: You'll need a REAL API key from one of the providers for this test.

#### Test with Real API Key
1. Get a free API key from:
   - Google AI Studio: https://aistudio.google.com/app/apikey (FREE)
   - OpenRouter: https://openrouter.ai/keys (FREE trial)

2. Enter your real API key in the modal
3. Select a model
4. Navigate to the chat interface
5. Send a test message: "Hello, can you respond to confirm you're working?"
6. ✅ Verify you get a response from the AI
7. ✅ Verify the correct model was used (check response style)

#### Test Provider-Specific Models
1. If using Google AI key, select "Gemini 3 Flash"
2. Send a message
3. ✅ Verify it works (no errors about incompatible models)
4. Try selecting "GPT-5.2"
5. ✅ Verify you get an error (model not available for this provider)

## Running Automated Tests

### Backend Tests
```bash
cd /Users/gunbirsingh/Desktop/Code\ folders/visionforge/project
python3 manage.py test block_manager.tests
```

**Expected Output:**
```
Ran 69 tests in X.XXXs

OK
```

### Test Coverage Breakdown
- **API Key Detection Tests**: 23 tests
  - Provider detection (OpenRouter, Google AI, OpenAI, Anthropic)
  - Invalid key handling
  - Edge cases (whitespace, wrong length, etc.)
  - Provider info retrieval
  - Model availability checking

- **Universal AI Factory Tests**: 24 tests
  - Service creation for each provider
  - Model validation and filtering
  - Server-side key handling (DEV mode)
  - Error handling

- **API Endpoint Tests**: 22 tests
  - Key validation endpoint
  - Available models endpoint
  - Environment info endpoint
  - Chat endpoint with universal keys

## Known Issues & Limitations

1. **Frontend Token Persistence**: API keys are stored in `sessionStorage` (cleared on browser close) for security. This is intentional.

2. **Rate Limiting**: The chat endpoint has rate limiting (15 requests/minute per IP in dev mode).

3. **Test API Keys**: The test keys in this guide are fake examples. You need real keys to test actual AI responses.

## Success Criteria

✅ **All 69 backend tests pass**
✅ **API key validation works for all 4 providers**
✅ **Model filtering works correctly per provider**
✅ **Frontend modal shows real-time validation**
✅ **Invalid keys are properly rejected**
✅ **Free tier badges display correctly**
✅ **Selected models are saved and restored**
✅ **Chat integration works with real API keys**

## Troubleshooting

### Issue: "Cannot import name 'GeminiService'"
**Solution**: Restart the Django server. The auto-reloader may have cached old imports.

### Issue: "401 Unauthorized" when testing chat
**Solution**: This is expected in PROD mode without a valid API key. Set `ENVIRONMENT=DEV` in `.env` or provide a real API key.

### Issue: Frontend can't connect to backend
**Solution**:
1. Verify backend is running on port 8000
2. Check CORS settings in `backend/settings.py`
3. Ensure `http://localhost:5173` is in `CORS_ALLOWED_ORIGINS`

## Next Steps

After testing, you can:
1. Deploy to production with real API keys
2. Add more AI providers (Cohere, Mistral, etc.)
3. Implement API key encryption for storage
4. Add usage tracking and quotas per key
5. Create a key management dashboard

## Test Results Summary

**Date**: December 22, 2025
**Backend Tests**: 69/69 PASSED ✅
**Manual API Tests**: ALL PASSED ✅
**Status**: Ready for User Acceptance Testing

---

For any issues during testing, please check:
1. Django server logs in terminal
2. Browser console for frontend errors
3. Network tab for failed API requests
