"""
VisionForge FastAPI Server

Local web server that runs on the user's machine.
Reads API keys from .env file (never exposed to frontend).
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .services import AIServiceFactory, infer_dimensions, validate_architecture
from .services.pytorch_codegen import PytorchCodeGenerator
from .services.tensorflow_codegen import TensorflowCodeGenerator

# Load environment variables from .env file
load_dotenv()

# Get configuration from environment
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


# Request/Response Models
class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = []
    modificationMode: Optional[bool] = False
    workflowState: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    response: str
    modifications: Optional[List[Dict[str, Any]]] = None


class ValidationRequest(BaseModel):
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]


class ValidationResponse(BaseModel):
    isValid: bool
    errors: Optional[List[str]] = None
    warnings: Optional[List[str]] = None


class ExportRequest(BaseModel):
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    format: str  # 'pytorch' or 'tensorflow'
    projectName: str


class ExportResponse(BaseModel):
    success: bool
    framework: str
    projectName: str
    files: Dict[str, str]
    zip: Optional[str] = None
    filename: Optional[str] = None


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""

    app = FastAPI(
        title="VisionForge",
        description="Visual Neural Network Builder - Local Server",
        version="0.1.0",
    )

    # CORS configuration for local development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, restrict this
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Get the web directory path
    web_dir = Path(__file__).parent / "web"

    # API Routes
    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        """
        Chat with AI assistant

        API keys are read from environment variables (.env file).
        Frontend never sees or sends API keys.
        """
        # Check if AI features are available
        if not GEMINI_API_KEY and not ANTHROPIC_API_KEY:
            raise HTTPException(
                status_code=400,
                detail="AI features not configured. Add GEMINI_API_KEY or ANTHROPIC_API_KEY to .env file."
            )

        try:
            # Create AI service (uses environment variables internally)
            ai_service = AIServiceFactory.create_service(
                gemini_key=GEMINI_API_KEY,
                anthropic_key=ANTHROPIC_API_KEY
            )

            # Process chat request
            response = await ai_service.chat(
                message=request.message,
                history=request.history,
                modification_mode=request.modificationMode,
                workflow_state=request.workflowState
            )

            return ChatResponse(
                response=response.get("response", ""),
                modifications=response.get("modifications")
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/validate", response_model=ValidationResponse)
    async def validate(request: ValidationRequest):
        """Validate model architecture"""
        try:
            result = validate_architecture(request.nodes, request.edges)
            return ValidationResponse(
                isValid=result.get("isValid", False),
                errors=result.get("errors", []),
                warnings=result.get("warnings", [])
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/export", response_model=ExportResponse)
    async def export_model(request: ExportRequest):
        """Export model to PyTorch or TensorFlow code"""
        try:
            # Select code generator based on framework
            if request.format.lower() == "pytorch":
                generator = PytorchCodeGenerator()
            elif request.format.lower() == "tensorflow":
                generator = TensorflowCodeGenerator()
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported framework: {request.format}")

            # Generate code
            result = generator.generate(
                nodes=request.nodes,
                edges=request.edges,
                project_name=request.projectName
            )

            return ExportResponse(
                success=True,
                framework=request.format,
                projectName=request.projectName,
                files=result.get("files", {}),
                zip=result.get("zip"),
                filename=result.get("filename")
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/infer-shapes")
    async def infer_shapes(request: ValidationRequest):
        """Infer tensor shapes throughout the architecture"""
        try:
            shapes = infer_dimensions(request.nodes, request.edges)
            return {"shapes": shapes}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/health")
    async def health():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "version": "0.1.0",
            "ai_enabled": bool(GEMINI_API_KEY or ANTHROPIC_API_KEY)
        }

    # Serve frontend static files
    if web_dir.exists():
        # Serve static assets
        assets_dir = web_dir / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

        # Serve index.html for all other routes (SPA support)
        @app.get("/{full_path:path}")
        async def serve_frontend(full_path: str):
            # Try to serve file if it exists
            file_path = web_dir / full_path
            if file_path.is_file():
                return FileResponse(file_path)

            # Otherwise serve index.html (SPA routing)
            index_path = web_dir / "index.html"
            if index_path.exists():
                return FileResponse(index_path)

            raise HTTPException(status_code=404, detail="Frontend not found")
    else:
        @app.get("/")
        async def no_frontend():
            return JSONResponse({
                "error": "Frontend not built",
                "message": "Run 'npm run build' in the frontend directory first"
            }, status_code=503)

    return app


# For running with uvicorn directly
app = create_app()


if __name__ == "__main__":
    import uvicorn

    print(f"""
    🚀 VisionForge is starting...

    Server: http://{HOST}:{PORT}
    AI Features: {'✅ Enabled' if (GEMINI_API_KEY or ANTHROPIC_API_KEY) else '❌ Disabled (add API keys to .env)'}

    Press Ctrl+C to stop
    """)

    uvicorn.run(app, host=HOST, port=PORT)
