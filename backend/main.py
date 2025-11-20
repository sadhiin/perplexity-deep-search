from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from .api.v1.endpoints import api_routes

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Welcome to the Perplexity application!"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


app.include_router(api_routes)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
