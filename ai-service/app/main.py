from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import realtime_analysis

app = FastAPI(title="AI Interview Microservice")

# Allow frontend origin
# origins = [
#     "http://localhost:5173",  # Adjust this to match your frontend
#     "http://127.0.0.1:5173"
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# Production: Post-Interview Analysis
# app.include_router(post_analysis.router, prefix="/post")

# Testing: Real-Time Audio Analysis
app.include_router(realtime_analysis.router, prefix="/test")

@app.get("/")
async def root():
    return {"message": "AI Microservice for Interviews"}
