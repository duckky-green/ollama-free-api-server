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
    try:
        # FIXED: Pass 'prompt' keyword argument directly instead of 'messages'
        response = client.chat(
            model=request.model, 
            prompt=request.prompt
        )
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
