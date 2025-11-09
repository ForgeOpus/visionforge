# VisionForge - Quick Start Guide

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or yarn

### 1. Setup Backend

```bash
cd project

# Install Python dependencies (if not already done)
pip install -r requirements.txt

# Run migrations
python manage.py makemigrations
python manage.py migrate

# Start Django server
python manage.py runserver
```

Backend runs on: `http://localhost:8000`

### 2. Setup Frontend

```bash
cd project/frontend

# Install dependencies (if not already done)
npm install

# Create environment file
cp .env.example .env

# Start development server
npm run dev
```

Frontend runs on: `http://localhost:5173`

### 3. Access Application

Open browser: `http://localhost:5173`

---

## 📖 User Guide

### Creating Your First Project

1. **Click** "Create New Project" from project dropdown
2. **Enter** project details:
   - Name: "My First Model"
   - Description: "Testing VisionForge"
   - Framework: PyTorch
3. **Click** "Create Project"
4. **Result**: Navigates to `/project/{id}`

### Building an Architecture

1. **Drag** an "Input" block from left sidebar onto canvas
2. **Click** the Input block to configure:
   - Shape: `[1, 3, 224, 224]` (for images)
   - Enable ground truth if needed
3. **Add** more blocks (Conv2D, ReLU, etc.)
4. **Connect** blocks by dragging from output handle to input handle
5. **Configure** each block by clicking it

### Saving Your Work

1. **Click** "Save" button in header
2. **Result**: Architecture saved to database
3. **Benefit**: Can reload from URL or project list

### Loading a Project

**Method 1**: Use URL directly
- `http://localhost:5173/project/{project-id}`

**Method 2**: Use project dropdown
- Click dropdown → Select project from list

### Exporting

**PyTorch Code**:
1. Click "Export" → "PyTorch Code"
2. View generated files (model.py, train.py, config.py)
3. Copy to clipboard or download

**JSON Architecture**:
1. Click "Export" → "JSON Architecture"
2. Downloads JSON file with architecture

### Importing

1. Click "Import" button
2. Select JSON file
3. Creates new project automatically

---

## 🎨 New Features

### New Block Types

**Output Node** (Green)
- Define model output type
- Configure number of classes

**Loss Function** (Red)
- Choose loss function
- Configure reduction method
- Set class weights

**Empty Node** (Gray)
- Placeholder for planning
- Add notes/comments

### Enhanced Input Node

- **Ground Truth**: Dual output for input + labels
- **Randomize**: Use synthetic data for testing
- **CSV File**: Load data from CSV

### UI Improvements

- **Error Badges**: Red badges on invalid nodes
- **Reset Button**: Clear canvas in toolbar
- **Scrollable Config**: Scroll through long configurations
- **URL Routing**: Shareable project links

---

## 💡 Tips & Tricks

### Quick Actions
- **Ctrl+Z**: Undo
- **Ctrl+Y**: Redo
- **Delete**: Remove selected block
- **Reset**: Clear entire canvas (in toolbar)

### Best Practices

1. **Start with Input**: Always begin with an Input node
2. **Configure Shapes**: Set input shape before connecting
3. **Validate Often**: Click "Validate" to check for errors
4. **Save Frequently**: Don't lose your work!
5. **Use Descriptions**: Add project descriptions for clarity

### Common Patterns

**Image Classification**:
```
Input → Conv2D → ReLU → MaxPool2D → Flatten → Linear → Output → Loss
```

**Text Processing**:
```
Input → Linear → ReLU → Attention → Linear → Output → Loss
```

**Regression**:
```
Input → Flatten → Linear → ReLU → Linear → Output → Loss(MSE)
```

---

## 🐛 Troubleshooting

### Backend Not Running
**Error**: Failed to fetch projects
**Solution**:
```bash
cd project
python manage.py runserver
```

### Frontend Build Errors
**Error**: Module not found
**Solution**:
```bash
cd project/frontend
rm -rf node_modules package-lock.json
npm install
```

### Database Issues
**Error**: No such table
**Solution**:
```bash
cd project
python manage.py migrate
```

### CORS Errors
**Error**: CORS policy blocked
**Solution**: Check `CORS_ALLOWED_ORIGINS` in `backend/settings.py`

### Port Conflicts
**Backend**: Change port with `python manage.py runserver 8001`
**Frontend**: Change port in `vite.config.ts`

---

## 📚 Additional Resources

- **Backend API**: http://localhost:8000/api/
- **Django Admin**: http://localhost:8000/admin/
- **Implementation Details**: See `IMPLEMENTATION_SUMMARY.md`
- **Export Format**: See `EXPORT_FORMAT.md`
- **README**: See `project/frontend/README.md`

---

## 🎯 Next Steps

1. **Create Example Projects**: Build sample architectures
2. **Export Code**: Generate PyTorch code and test it
3. **Share Projects**: Share URLs with your team
4. **Explore Blocks**: Try all 16 block types
5. **Customize**: Modify blocks to fit your needs

---

## ✨ Quick Tips

- Projects auto-save when you click "Save"
- URLs are shareable - copy and send to teammates
- JSON export is human-readable - open in editor
- Ground truth output enables supervised learning setups
- Use Empty nodes to plan architecture before implementing
- Reset button clears canvas but doesn't delete project

---

Happy model building! 🚀
