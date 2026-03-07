# Common Issues & Solutions

Find solutions to the most frequently encountered problems when using VisionForge.

## 🚨 Quick Fixes

### Backend Issues

#### Django Server Won't Start
**Problem**: Server fails to start with port errors

**Solution**:
```bash
# Check if port is in use
netstat -ano | findstr :8000

# Kill the process (Windows)
taskkill /PID <PID> /F

# Kill the process (macOS/Linux)
kill -9 <PID>

# Or use different port
python manage.py runserver 8080
```

#### Database Migration Errors
**Problem**: `django.db.migrations.exceptions.InconsistentMigrationHistory`

**Solution**:
```bash
# Delete database and migrate fresh
rm db.sqlite3
python manage.py migrate
```

#### CORS Errors
**Problem**: `Access-Control-Allow-Origin` header missing

**Solution**:
1. Ensure both backend and frontend are running
2. Check `settings.py` CORS configuration:
   ```python
   CORS_ALLOWED_ORIGINS = [
       "http://localhost:3000",
       "http://localhost:5173",
   ]
   ```

### Frontend Issues

#### npm Install Fails
**Problem**: `npm ERR! code ERESOLVE`

**Solution**:
```bash
# Clear npm cache
npm cache clean --force

# Delete node_modules and package-lock.json
rm -rf node_modules package-lock.json

# Reinstall
npm install
```

#### Development Server Errors
**Problem**: Vite dev server shows compilation errors

**Solution**:
```bash
# Check Node.js version
node --version  # Should be 16+

# Update to latest version
npm install -g npm@latest
npm install
```

## 🔗 Connection & Architecture Issues

### Invalid Connections

#### Red Connection Lines
**Problem**: Connections show as red (invalid)

**Common Causes**:
- Shape mismatch between layers
- Incompatible layer types
- Missing required parameters

**Solutions**:
1. **Check shape compatibility**:
   - Hover over connection to see shapes
   - Review [Layer Connection Rules](../architecture/connection-rules.md)

2. **Verify layer configuration**:
   - Click each layer to check parameters
   - Ensure all required fields are filled

3. **Fix common shape issues**:
   ```
   Conv2D → Linear (needs Flatten)
   Input[1,3,224,224] → Conv2D → Flatten → Linear
   ```

#### Orphaned Blocks
**Problem**: Blocks not connected to the main graph

**Solution**:
- Connect orphaned blocks to the main flow
- Or delete them if unnecessary
- Check validation panel for warnings

### Shape Inference Errors

#### "Cannot determine output shape"
**Problem**: Shape inference fails for a layer

**Common Causes**:
- Missing input shape configuration
- Invalid parameter values
- Unsupported layer combinations

**Solutions**:
1. **Set input shape**:
   ```json
   {
     "inputShape": {
       "dims": [1, 3, 224, 224]
     }
   }
   ```

2. **Check layer parameters**:
   - Verify kernel sizes are positive
   - Ensure stride values are reasonable
   - Check padding values

3. **Review layer sequence**:
   - Some layers require specific input types
   - Check [Shape Inference Guide](../architecture/shape-inference.md)

## 🚀 Export Issues

### Code Generation Fails

#### "Export failed: Invalid architecture"
**Problem**: Export fails with validation errors

**Solution**:
1. **Fix all validation errors**:
   - Check the validation panel at the bottom
   - All red indicators must be resolved

2. **Ensure complete architecture**:
   - Input layer must be connected
   - Output layer should be present
   - No circular dependencies

3. **Verify layer configuration**:
   - All required parameters filled
   - Parameter values in valid ranges

#### Generated Code Has Errors

**Problem**: Exported Python code doesn't run

**Common Issues**:
1. **Missing imports**:
   ```python
   import torch
   import torch.nn as nn
   import torch.nn.functional as F
   ```

2. **Shape mismatches**:
   - Check calculated vs actual tensor shapes
   - Verify flatten operations

3. **Framework-specific issues**:
   - PyTorch vs TensorFlow syntax differences
   - Parameter naming conventions

**Debug Steps**:
```python
# Test your model step by step
model = YourModel()
x = torch.randn(1, 3, 224, 224)

# Forward pass with error catching
try:
    output = model(x)
    print(f"Success! Output shape: {output.shape}")
except Exception as e:
    print(f"Error: {e}")
```

## 🖥️ Interface Issues

### Canvas Problems

#### Blocks Not Responding
**Problem**: Can't select or move blocks

**Solutions**:
1. **Refresh the page** - Simple browser refresh
2. **Clear browser cache**:
   - Ctrl+Shift+R (hard refresh)
   - Or clear cache in browser settings

#### Zoom/Pan Not Working
**Problem**: Canvas navigation issues

**Solutions**:
1. **Check browser compatibility** - Use Chrome, Firefox, or Edge
2. **Try keyboard shortcuts**:
   - `Ctrl + Mouse wheel` for zoom
   - `Click + drag` for pan

#### Performance Issues
**Problem**: Interface is slow or laggy

**Solutions**:
1. **Reduce canvas complexity**:
   - Use group blocks for large architectures
   - Delete unused blocks

2. **Browser optimization**:
   - Close unnecessary tabs
   - Update browser to latest version

## 🔧 Configuration Issues

### Environment Variables

#### API Key Not Working
**Problem**: AI assistant features not available

**Solution**:
1. **Check .env file**:
   ```env
   GEMINI_API_KEY=your_actual_api_key_here
   ```

2. **Verify API key validity**:
   - Check Google AI Studio dashboard
   - Ensure key is active

3. **Restart servers**:
   ```bash
   # Stop both servers and restart
   python manage.py runserver
   npm run dev
   ```

### Database Issues

#### "Database is locked"
**Problem**: SQLite database access errors

**Solution**:
```bash
# Stop Django server
# Delete database file
rm db.sqlite3

# Recreate database
python manage.py migrate
```

#### Migration Conflicts
**Problem**: Django migration errors

**Solution**:
```bash
# Reset migrations
python manage.py migrate --fake
python manage.py migrate
```

## 📱 Browser-Specific Issues

### Chrome/Edge
- **Works best** with latest versions
- **Enable hardware acceleration** in settings

### Firefox
- **May need** to enable WebRTC features
- **Check** about:config for media settings

### Safari
- **Limited support** - some features may not work
- **Use Chrome** for full functionality

## 🆘 Getting Help

### Self-Service Resources

1. **Check the validation panel** - Bottom of the interface
2. **Hover over elements** - Tooltips provide helpful information
3. **Review the documentation**:
   - [Layer Connection Rules](../architecture/connection-rules.md)
   - [Shape Inference](../architecture/shape-inference.md)
   - [API Reference](../api/rest-api.md)

### Community Support

1. **GitHub Issues**:
   - Search existing issues first
   - Provide detailed error messages
   - Include screenshots when helpful

2. **Discord Community**:
   - Real-time help from other users
   - Share your architecture for feedback

### Reporting Issues

When reporting problems, include:

1. **System Information**:
   - Operating system
   - Browser version
   - Python and Node.js versions

2. **Error Details**:
   - Full error messages
   - Steps to reproduce
   - Screenshots if applicable

3. **Architecture Details**:
   - Export your architecture (JSON)
   - Describe expected vs actual behavior

## 📋 Prevention Checklist

To avoid common issues:

### Before Starting
- [ ] Check system requirements
- [ ] Update browsers and dependencies
- [ ] Set up environment variables

### During Development
- [ ] Save work frequently
- [ ] Validate after each major change
- [ ] Check connection colors (green = valid)

### Before Export
- [ ] Fix all validation errors
- [ ] Verify all parameters are set
- [ ] Test with sample data

### Regular Maintenance
- [ ] Clear browser cache weekly
- [ ] Update dependencies monthly
- [ ] Backup important projects

---

**Still stuck?** → [Contact Support](mailto:support@visionforge.ai)
