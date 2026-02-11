"""
EcoloGRAPH Chat Demo - Interactive terminal interface.

Usage:
    python scripts/chat_demo.py [--model MODEL] [--base-url URL]

Connects to a local LLM (LM Studio / Ollama) and provides
an interactive chat with the EcoloGRAPH toolset.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="EcoloGRAPH Chat Demo")
    parser.add_argument(
        "--model", "-m",
        default="auto",
        help="Model name, or 'auto' to detect from LM Studio (default: auto)"
    )
    parser.add_argument(
        "--base-url", "-u",
        default="http://localhost:1234/v1",
        help="API base URL (default: http://localhost:1234/v1)"
    )
    parser.add_argument(
        "--api-key", "-k",
        default="lm-studio",
        help="API key (default: lm-studio)"
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.3,
        help="Temperature (default: 0.3)"
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming output"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("🌿 EcoloGRAPH Chat (beta)")
    print("=" * 60)
    print(f"  API:      {args.base_url}")
    print(f"  Temp:     {args.temperature}")
    print()
    
    # Initialize agent
    print("🔧 Initializing agent...")
    try:
        from src.agent import QueryAgent
        agent = QueryAgent(
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            temperature=args.temperature,
        )
        
        info = agent.get_info()
        print(f"  ✅ Model:  {info['model']}")
        if info.get('model_info', {}).get('owned_by'):
            print(f"  Owner:   {info['model_info']['owned_by']}")
        print(f"  Tools:   {len(info['tools'])} available")
        print(f"           ({', '.join(info['tools'])})")
        
    except ConnectionError as e:
        print(f"\n❌ {e}")
        return
    except RuntimeError as e:
        print(f"\n❌ {e}")
        return
    except Exception as e:
        print(f"\n❌ Failed to initialize agent: {e}")
        return
    
    print()
    print("-" * 60)
    print("Type your questions below. Commands:")
    print("  /help     - Show available tools")
    print("  /info     - Show agent config")
    print("  /quit     - Exit")
    print("-" * 60)
    print()
    
    # Chat loop
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Goodbye!")
            break
        
        if not user_input:
            continue
        
        # Handle commands
        if user_input.startswith("/"):
            cmd = user_input.lower()
            
            if cmd in ("/quit", "/exit", "/q"):
                print("\n👋 Goodbye!")
                break
            
            elif cmd == "/help":
                from src.agent import get_tool_descriptions
                print(f"\n{get_tool_descriptions()}\n")
                continue
            
            elif cmd == "/info":
                info = agent.get_info()
                print(f"\n  Model: {info['model']}")
                print(f"  URL:   {info['base_url']}")
                print(f"  Tools: {info['tools']}")
                print(f"  Max iterations: {info['max_iterations']}\n")
                continue
            
            else:
                print(f"  Unknown command: {cmd}\n")
                continue
        
        # Query the agent
        print()
        
        if args.no_stream:
            # Non-streaming mode
            print("🧠 Thinking...")
            answer = agent.ask(user_input)
            print(f"\nEcoloGRAPH: {answer}")
        else:
            # Streaming mode
            for event_type, content in agent.ask_streaming(user_input):
                if event_type == "tool_call":
                    print(f"  {content}")
                elif event_type == "tool_result":
                    # Show truncated result
                    preview = content[:100].replace("\n", " ")
                    print(f"  {preview}...")
                elif event_type == "answer":
                    print(f"\nEcoloGRAPH: {content}")
                elif event_type == "error":
                    print(f"\n❌ {content}")
        
        print()


if __name__ == "__main__":
    main()
