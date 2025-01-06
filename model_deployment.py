from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Optional
import uvicorn
from huggingface_hub import login
import os

# First, ensure we have the required packages
try:
    from transformers import GemmaTokenizer, GemmaForCausalLM
except ImportError:
    raise ImportError("Please install the latest transformers package: pip install --upgrade transformers")

app = FastAPI(title="Multi-Model LLM API")

# Model configurations with your specific models
MODEL_CONFIGS = {
    "gem_marketing": {
        "model_id": "marketeam/GemMarketing",
        "tokenizer_id": "google/gemma-2b",
        "model_class": GemmaForCausalLM,
        "tokenizer_class": GemmaTokenizer,
        "description": "Marketing Domain LLM"
    },
    "lla_marketing": {
        "model_id": "marketeam/LLaMarketing",
        "tokenizer_id": "google/gemma-2b",
        "model_class": GemmaForCausalLM,
        "tokenizer_class": GemmaTokenizer,
        "description": "Marketing LLaMA Model"
    },
    "cannabis": {
        "model_id": "aznatkoiny/GemmaLM-for-Cannabis",
        "tokenizer_id": "google/gemma-2b",
        "model_class": GemmaForCausalLM,
        "tokenizer_class": GemmaTokenizer,
        "description": "Cannabis Domain LLM"
    }
}

# Global storage for loaded models
loaded_models = {}
loaded_tokenizers = {}

class PredictionRequest(BaseModel):
    text: str
    model_name: str
    max_length: Optional[int] = 512
    temperature: Optional[float] = 0.7

    class Config:
        protected_namespaces = ()  # This removes the warning about model_name

class PredictionResponse(BaseModel):
    generated_text: str
    model_name: str

def load_model(model_name: str):
    """Load model and tokenizer if not already loaded"""
    if model_name not in MODEL_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Model {model_name} not configured")
    
    if model_name not in loaded_models:
        config = MODEL_CONFIGS[model_name]
        try:
            print(f"Loading {model_name} model...")
            
            # Load tokenizer using specific tokenizer class
            tokenizer = config["tokenizer_class"].from_pretrained(
                config["tokenizer_id"],
                token=os.getenv("HF_TOKEN")
            )
            loaded_tokenizers[model_name] = tokenizer
            
            # Load model using specific model class
            model = config["model_class"].from_pretrained(
                config["model_id"],
                token=os.getenv("HF_TOKEN"),
                device_map="cpu",
                torch_dtype=torch.float32
            )
            loaded_models[model_name] = model
            
            print(f"{model_name} model loaded successfully")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

async def startup():
    """Initialize environment and models"""
    # Set Hugging Face token
    hf_token = "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"
    os.environ["HF_TOKEN"] = hf_token
    
    # Login to Hugging Face
    login(token=hf_token)
    
    # Load first model initially
    load_model("gem_marketing")

app.add_event_handler("startup", startup)

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Generate text using the specified model"""
    try:
        # Ensure model is loaded
        if request.model_name not in loaded_models:
            load_model(request.model_name)
        
        model = loaded_models[request.model_name]
        tokenizer = loaded_tokenizers[request.model_name]
        
        # Prepare input
        inputs = tokenizer(request.text, return_tensors="pt")
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=request.max_length,
                temperature=request.temperature,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Decode output
        generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        
        return PredictionResponse(
            generated_text=generated_text,
            model_name=request.model_name
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
    """List available models"""
    return {
        "available_models": [
            {
                "name": name,
                "description": config["description"],
                "loaded": name in loaded_models
            }
            for name, config in MODEL_CONFIGS.items()
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "loaded_models": list(loaded_models.keys()),
        "device": "CPU",
        "memory_info": {
            "available_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
            if torch.cuda.is_available() else "CPU Only"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)