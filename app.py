import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from typing import Optional

# ------------------------
# Logging Configuration
# ------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ------------------------
# FastAPI App
# ------------------------
app = FastAPI(title="Gemma-2-9b API")

class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 1024
    top_p: float = 0.9
    temperature: float = 0.7

# ------------------------
# Global model variable
# ------------------------
llm: Optional[Llama] = None

# ------------------------
# Startup - Load Model
# ------------------------
@app.on_event("startup")
def load_model():
    global llm
    try:
        if llm is None:
            repo_id = "ArchishSkyllect/gemma-2-9b-it-gguf"
            filename = "gemma-2-9b-it-Q4_K_M.gguf"

            logger.info("📥 Downloading model from Hugging Face Hub...")
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                token=os.environ.get("HF_TOKEN")
            )

            logger.info("⚙️ Initializing model (this may take a few minutes)...")
            llm = Llama(
                model_path=model_path,
                n_ctx=2048,
                n_batch=128,
                n_threads=2, 
                verbose=False
            )
            logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.error("❌ Failed to load model", exc_info=True)
        raise RuntimeError(f"Model loading failed: {e}")

# ------------------------
# Generate Endpoint
# ------------------------
@app.post("/generate")
async def generate(request: PromptRequest):
    try:
        if llm is None:
            logger.warning("⚠️ Model not ready yet")
            raise HTTPException(status_code=503, detail="Model is still loading. Please try again soon.")

        full_prompt = request.prompt.strip()

        logger.info(f"➡️ Incoming prompt: {full_prompt}")

        output = llm(
            prompt=full_prompt,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            temperature=request.temperature,
            stop=["<end_of_turn>", "</s>"],
            echo=False,
            repeat_penalty=1.25
        )

        generated_text = output["choices"][0]["text"].strip()

        logger.info(f"✅ Model response: {generated_text}")

        return {"bot_response": generated_text}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("❌ Error in /generate endpoint", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
