from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ollamafreeapi import OllamaFreeAPI

app = FastAPI(title="My Free AI Proxy Endpoint")
client = OllamaFreeAPI()

class ChatRequest(BaseModel):
    model: str = "deepseek-r1"
    prompt: str

@app.post("/v1/chat")
async def chat_endpoint(request: ChatRequest):
    # Order of backup models to try if the requested one is down
    fallback_models = [request.model, "llama3", "mistral", "qwen", "phi3"]
    
    last_error = ""
    
    # Loop through models until one works
    for model_name in fallback_models:
        try:
            response = client.chat(
                model=model_name, 
                prompt=request.prompt
            )
            # If successful, return the response along with the model that actually answered
            return {
                "status": "success",
                "model_used": model_name,
                "response": response
            }
        except Exception as e:
            last_error = str(e)
            print(f"Model '{model_name}' failed or unavailable. Trying next...")
            continue
            
    # If absolutely every model fails, raise the final error
    raise HTTPException(
        status_code=500, 
        detail=f"All available community models are currently offline. Last error: {last_error}"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
