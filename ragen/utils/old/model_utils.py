# model_utils.py
import os
import torch
import openai
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict

def load_model(model_id: str) -> tuple:
    """Load and configure a pretrained model.
    
    Args:
        model_id: Identifier for the pretrained model
        
    Returns:
        Tuple of (model, tokenizer)
    """
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model = model.to(torch.bfloat16).eval().to("cuda")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    model.gradient_checkpointing_enable()
    
    return model, tokenizer

def create(
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 1000,
    n: int = 1,
    temperature: float = 0.7,
    **kwargs
) -> str:
    """Create completion using specified model.
    
    Args:
        model: Model identifier
        messages: List of conversation messages
        max_tokens: Maximum tokens in response
        n: Number of completions
        temperature: Sampling temperature
        **kwargs: Additional parameters
        
    Returns:
        Generated completion text
    """
    if 'deepseek' in model.lower():
        client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
    else:
        client = openai.OpenAI()

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        n=n,
        temperature=temperature,
        **kwargs
    )
    return response.choices[0].message.content