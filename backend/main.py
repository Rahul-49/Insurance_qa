# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import router
import uvicorn

app = FastAPI(
    title="Insurance QA System",
    description="API for querying insurance policy documents.",
    version="1.0.0"
)

# Add CORS middleware
# Using "*" is permissive for development. For production, you should list
# your specific frontend domain(s), e.g., ["https://your-app.vercel.app"].
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include the API routes from routes.py
app.include_router(router, prefix="/api") # Added a /api prefix for good practice

if __name__ == "__main__":
    # This block allows running the server directly for development
    # Use: python main.py
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
