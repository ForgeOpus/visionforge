# Quick Start: AI Provider Configuration

## 🚀 Quick Setup

### Option 1: Use Gemini (Free, Fast)

1. **Get your key:** https://aistudio.google.com/app/apikey

2. **Edit `.env`:**
   ```env
   AI_PROVIDER=gemini
   GEMINI_API_KEY=AIzaSy_your_key_here
   ```

3. **Restart server:**
   ```bash
   python manage.py runserver
   ```

**Done!** Your chatbot is now powered by Gemini.

---

### Option 2: Use Claude (High Quality)

1. **Get your key:** https://console.anthropic.com/

2. **Install package:**
   ```bash
   pip install anthropic
   ```

3. **Edit `.env`:**
   ```env
   AI_PROVIDER=claude
   ANTHROPIC_API_KEY=sk-ant-your_key_here
   ```

4. **Restart server:**
   ```bash
   python manage.py runserver
   ```

**Done!** Your chatbot is now powered by Claude.

---

## 🔄 Switch Providers Anytime

Just change `AI_PROVIDER` in `.env` and restart:

```env
AI_PROVIDER=claude  # or 'gemini'
```

---

## ⚡ Which Should I Choose?

### Choose Gemini if you want:
- ✅ Free tier (15 requests/minute)
- ✅ Fastest response times
- ✅ Easy setup

### Choose Claude if you want:
- ✅ Best reasoning and code understanding
- ✅ Most detailed explanations
- ✅ Highest quality suggestions
- ⚠️ Paid API (no free tier)

---

## 📝 Environment Variables Reference

### Required Variables

```env
# Choose provider: 'gemini' or 'claude'
AI_PROVIDER=gemini

# Gemini setup
GEMINI_API_KEY=your_gemini_key

# Claude setup
ANTHROPIC_API_KEY=your_anthropic_key
```

### Complete `.env` Template

```env
# Django Settings
SECRET_KEY=your-secret-key-here
DEBUG=True

# AI Provider Configuration
# Choose which AI provider to use: 'gemini' or 'claude'
AI_PROVIDER=gemini

# Gemini AI Configuration
# Get your API key from: https://aistudio.google.com/app/apikey
GEMINI_API_KEY=your-gemini-api-key-here

# Claude AI Configuration
# Get your API key from: https://console.anthropic.com/
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# Database (optional, defaults to SQLite)
# DATABASE_URL=postgresql://user:password@localhost/dbname
```

---

## 🆘 Troubleshooting

### Error: "AI service not properly configured"

**Fix:**
1. Check `AI_PROVIDER` is set to `gemini` or `claude`
2. Check the corresponding API key is set
3. Restart Django server: `python manage.py runserver`

### Error: "Invalid AI_PROVIDER"

**Fix:** Set `AI_PROVIDER` to exactly `gemini` or `claude` (lowercase)

### Chatbot not responding

**Fix:**
1. Open browser console (F12)
2. Check backend terminal for errors
3. Verify API key is correct
4. Check you haven't exceeded rate limits

---

## 💡 Pro Tips

1. **Keep both API keys configured** - makes switching instant
2. **Start with Gemini** - it's free and great for testing
3. **Upgrade to Claude** - when you need highest quality suggestions
4. **Never commit `.env`** - it's already in `.gitignore`

---

## 📚 More Information

- Full setup guide: `docs/CHATBOT_SETUP.md`
- Implementation details: `docs/AI_PROVIDER_IMPLEMENTATION.md`
- Gemini docs: https://ai.google.dev/docs
- Claude docs: https://docs.anthropic.com/
