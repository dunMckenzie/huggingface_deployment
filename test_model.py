import requests
import json
import time
from typing import Dict, Any
import sys

def test_health_endpoint(base_url: str) -> Dict[str, Any]:
    """Test the health check endpoint"""
    try:
        response = requests.get(f"{base_url}/health")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Health check failed: {str(e)}")
        sys.exit(1)

def test_models_endpoint(base_url: str) -> Dict[str, Any]:
    """Test the models listing endpoint"""
    try:
        response = requests.get(f"{base_url}/models")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Models listing failed: {str(e)}")
        sys.exit(1)

def test_prediction(base_url: str, model_name: str, test_prompt: str) -> Dict[str, Any]:
    """Test model prediction"""
    payload = {
        "text": test_prompt,
        "model_name": model_name,
        "max_length": 256,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(
            f"{base_url}/predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Prediction failed for {model_name}: {str(e)}")
        return None

def run_tests(base_url: str = "http://localhost:8000"):
    """Run all tests"""
    print("\n=== Starting Model API Tests ===\n")
    
    # Test 1: Health Check
    print("1. Testing Health Endpoint...")
    health_status = test_health_endpoint(base_url)
    print(f"Health Status: {json.dumps(health_status, indent=2)}\n")
    
    # Test 2: Available Models
    print("2. Testing Models Endpoint...")
    models_info = test_models_endpoint(base_url)
    print(f"Available Models: {json.dumps(models_info, indent=2)}\n")
    
    # Test 3: Test each model with different prompts
    print("3. Testing Model Predictions...")
    
    test_cases = {
        "gem_marketing": "Write a compelling product description for a new eco-friendly water bottle.",
        "lla_marketing": "Create a social media post about a summer sale event.",
        "cannabis": "Explain the differences between indica and sativa strains."
    }
    
    for model_name, prompt in test_cases.items():
        print(f"\nTesting {model_name}...")
        print(f"Prompt: {prompt}")
        
        start_time = time.time()
        result = test_prediction(base_url, model_name, prompt)
        end_time = time.time()
        
        if result:
            print(f"\nGenerated Text:\n{result['generated_text']}")
            print(f"\nResponse Time: {end_time - start_time:.2f} seconds")
        
        print("\n" + "="*50)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test LLM API endpoints")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL for the API (default: http://localhost:8000)"
    )
    
    args = parser.parse_args()
    run_tests(args.url)