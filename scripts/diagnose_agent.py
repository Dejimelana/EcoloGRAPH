"""
Quick diagnostic for Ollama + EcoloGRAPH agent.
Tests each component independently to find the bottleneck.
"""
import sys
import time
import httpx
sys.path.insert(0, ".")

OLLAMA_URL = "http://localhost:11434/v1"

def test_raw_ollama(model="qwen3:8b"):
    """Test 1: Raw Ollama speed (no agent overhead)."""
    print(f"\n{'='*50}")
    print(f"TEST 1: Raw Ollama ({model})")
    print(f"{'='*50}")
    
    client = httpx.Client(timeout=120)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "What species live in coral reefs? Answer in 2 sentences."}],
        "temperature": 0.3,
        "max_tokens": 200,
    }
    
    t0 = time.time()
    try:
        r = client.post(f"{OLLAMA_URL}/chat/completions", json=payload)
        r.raise_for_status()
        elapsed = time.time() - t0
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        tokens = data.get("usage", {}).get("completion_tokens", "?")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Tokens: {tokens}")
        print(f"  Response: {content[:200]}")
        return elapsed
    except Exception as e:
        print(f"  ERROR: {e}")
        return None

def test_tool_calling(model="qwen3:8b"):
    """Test 2: Does the model support tool calling?"""
    print(f"\n{'='*50}")
    print(f"TEST 2: Tool Calling ({model})")
    print(f"{'='*50}")
    
    client = httpx.Client(timeout=120)
    tools = [{
        "type": "function",
        "function": {
            "name": "search_papers",
            "description": "Search papers by keyword",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    }]
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Search for papers about coral reef bleaching"}],
        "temperature": 0.3,
        "max_tokens": 500,
        "tools": tools,
    }
    
    t0 = time.time()
    try:
        r = client.post(f"{OLLAMA_URL}/chat/completions", json=payload)
        r.raise_for_status()
        elapsed = time.time() - t0
        data = r.json()
        
        choice = data["choices"][0]
        content = choice["message"].get("content", "")
        tool_calls = choice["message"].get("tool_calls", [])
        
        print(f"  Time: {elapsed:.1f}s")
        if tool_calls:
            print(f"  Tool calls: {len(tool_calls)}")
            for tc in tool_calls:
                print(f"    -> {tc['function']['name']}({tc['function']['arguments']})")
        else:
            print(f"  No tool calls (responded directly)")
            print(f"  Content: {content[:200]}")
        
        return elapsed
    except Exception as e:
        print(f"  ERROR: {e}")
        return None

def test_agent():
    """Test 3: Full QueryAgent pipeline."""
    print(f"\n{'='*50}")
    print(f"TEST 3: Full QueryAgent Pipeline")
    print(f"{'='*50}")
    
    try:
        from src.core.config import _load_api_key_file, get_settings
        _load_api_key_file()
        settings = get_settings()
        
        import os
        api_key = os.environ.get("OPENAI_API_KEY", "ollama")
        
        from src.agent import QueryAgent
        model = settings.llm.reasoning_model
        print(f"  Model: {model}")
        print(f"  Base URL: {settings.llm.base_url}")
        
        agent = QueryAgent(model=model, api_key=api_key, base_url=settings.llm.base_url)
        info = agent.get_info()
        print(f"  Tools: {len(info['tools'])} ({', '.join(info['tools'])})")
        
        query = "What species are in the database?"
        print(f"\n  Query: '{query}'")
        print(f"  Waiting for response...")
        
        t0 = time.time()
        for event_type, content in agent.ask_streaming(query):
            elapsed = time.time() - t0
            if event_type == "routing":
                print(f"  [{elapsed:.1f}s] Routing: {content}")
            elif event_type == "tool_call":
                print(f"  [{elapsed:.1f}s] Tool call: {content}")
            elif event_type == "tool_result":
                preview = content[:100].replace("\n", " ")
                print(f"  [{elapsed:.1f}s] Tool result: {preview}...")
            elif event_type == "answer":
                print(f"  [{elapsed:.1f}s] ANSWER ({len(content)} chars): {content[:200]}...")
            elif event_type == "error":
                print(f"  [{elapsed:.1f}s] ERROR: {content}")
        
        total = time.time() - t0
        print(f"\n  Total time: {total:.1f}s")
        return total
        
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    models = ["qwen3:8b"]
    
    for model in models:
        t1 = test_raw_ollama(model)
        t2 = test_tool_calling(model)
    
    t3 = test_agent()
    
    print(f"\n\n{'='*50}")
    print(f"SUMMARY")
    print(f"{'='*50}")
    if t1: print(f"  Raw Ollama:     {t1:.1f}s")
    if t2: print(f"  Tool calling:   {t2:.1f}s")
    if t3: print(f"  Full agent:     {t3:.1f}s")
    if t1 and t3:
        overhead = t3 / t1
        print(f"  Agent overhead: {overhead:.1f}x")
