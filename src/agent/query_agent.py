"""
Query Agent using LangGraph for EcoloGRAPH.

Two-Tier Architecture:
  Tier 1 (Fast): Router classifies intent → direct response for chat/meta
  Tier 2 (Full): ReAct agent with tools for research/analysis queries

LangGraph StateGraph flow:
  START → router → chat_respond  → END   (fast, no tools)
                 → agent         → tools ↔ agent → END  (full)
                 → meta_respond  → END   (instant, no LLM)
"""
import logging
import operator
from typing import Annotated, TypedDict, Literal

import httpx

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from .tool_registry import ALL_TOOLS, get_tool_descriptions
from .tool_groups import get_tools_for_strategy

logger = logging.getLogger(__name__)


# ============================================================
# Model Auto-Detection
# ============================================================

def detect_loaded_model(base_url: str = "http://localhost:11434/v1") -> dict:
    """
    Query Ollama to detect the currently loaded model.
    
    Returns:
        dict with 'id', 'object', 'owned_by' or empty dict if no model loaded.
    """
    try:
        response = httpx.get(f"{base_url}/models", timeout=5.0)
        response.raise_for_status()
        data = response.json()
        models = data.get("data", [])
        
        if models:
            model = models[0]
            return {
                "id": model.get("id", "unknown"),
                "object": model.get("object", "model"),
                "owned_by": model.get("owned_by", "unknown"),
            }
        return {}
    except httpx.ConnectError:
        raise ConnectionError(
            f"Cannot connect to LLM server at {base_url}. "
            "Make sure Ollama is running (ollama serve)."
        )
    except Exception as e:
        logger.warning(f"Could not detect model: {e}")
        return {}


# ============================================================
# Agent State
# ============================================================

class AgentState(TypedDict):
    """State that flows through the LangGraph nodes."""
    messages: Annotated[list[BaseMessage], operator.add]
    intent: str  # chat, search, analyze, meta
    strategy: str  # search, graph, external, inference, full


# ============================================================
# Prompts
# ============================================================

ROUTER_PROMPT = """Classify this user message into ONE category:
- "meta" = questions about the system, model, tools, or greetings (hola, hi, help)
- "search" = looking for papers, species info, or specific data
- "analyze" = domain classification, cross-domain links, hypothesis generation
- "chat" = general ecology questions answerable from your knowledge

Respond with ONLY one word: meta, search, analyze, or chat."""

STRATEGY_ROUTER_PROMPT = """Classify this research query into ONE strategy:
- "search" = finding papers, searching by domain, or classifying text
- "graph" = knowledge graph queries, species relationships, paper connections
- "external" = external species lookups (GBIF, FishBase, IUCN)
- "inference" = cross-domain connections, hypothesis generation, novel insights
- "full" = complex query needing multiple tool types

Respond with ONLY one word: search, graph, external, inference, or full."""

CHAT_SYSTEM = """You are EcoloGRAPH, a scientific research assistant specialized in ecology.
You have access to a database of indexed scientific papers, a Neo4j knowledge graph,
and 8 tools: search_papers, search_by_domain, query_graph, classify_text,
get_species_info, find_cross_domain_links, generate_hypotheses, search_related_papers.

Answer the user's question. If data from your database would help, suggest using your tools.
Be precise with scientific terminology.
Always respond in the same language the user writes in."""

AGENT_SYSTEM = """You are EcoloGRAPH, a scientific research assistant with access to tools.

{tool_descriptions}

Guidelines:
- Use tools to find real data — don't make up facts
- Cite paper titles or sources when providing information
- For domain-specific queries, use search_by_domain
- Available domains: marine_ecology, freshwater_ecology, conservation, genetics, 
  botany, zoology, entomology, ornithology, toxicology, remote_sensing, machine_learning, 
  computer_vision, soundscape_ecology, deep_learning, physiology, ethology, 
  biotic_interactions, geology, biogeography, and more.

Answer in the same language the user uses."""


# ============================================================
# QueryAgent
# ============================================================

class QueryAgent:
    """
    Two-tier LangGraph agent for EcoloGRAPH.
    
    Tier 1 (Fast): Router + direct response — no tool overhead
    Tier 2 (Full): ReAct agent with tools — for research queries
    """
    
    def __init__(
        self,
        model: str = "auto",
        base_url: str = "http://localhost:1234/v1",
        api_key: str = "lm-studio",
        temperature: float = 0.3,
        max_iterations: int = 5,
    ):
        self.max_iterations = max_iterations
        self.base_url = base_url
        
        # Auto-detect model
        if model == "auto":
            detected = detect_loaded_model(base_url)
            if not detected:
                raise RuntimeError(
                    "No model loaded in LM Studio/Ollama. "
                    "Please load a model and try again."
                )
            model = detected["id"]
            self.model_info = detected
            logger.info(f"Auto-detected model: {model}")
        else:
            self.model_info = {"id": model, "owned_by": "user-specified"}
        
        self.model_name = model
        
        # Tier 1: Fast LLM (no tools bound — minimal overhead)
        self.llm_fast = ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            max_tokens=2048,
        )
        
        # Tier 2: Full LLM with tools
        self.tools = ALL_TOOLS
        self.llm_with_tools = self.llm_fast.bind_tools(self.tools)
        
        # Build the graph
        self.graph = self._build_graph()
        
        logger.info(
            f"QueryAgent initialized: model={model}, "
            f"tools={len(self.tools)}, base_url={base_url}"
        )
    
    # --------------------------------------------------------
    # LangGraph Construction
    # --------------------------------------------------------
    
    def _build_graph(self):
        """
        Build the three-tier hierarchical LangGraph:
        
        START → router (tier 1: meta/chat/research)
                      ↓
                   meta_respond  → END  (instant)
                   chat_respond  → END  (fast, no tools)
                   research_router (tier 2: strategy classification)
                      ↓
                   agent (tier 3: subset tools) ↔ tools → END
        """
        graph = StateGraph(AgentState)
        
        # Nodes
        graph.add_node("router", self._router_node)
        graph.add_node("meta_respond", self._meta_node)
        graph.add_node("chat_respond", self._chat_node)
        graph.add_node("research_router", self._research_router_node)
        graph.add_node("agent", self._agent_node)
        graph.add_node("tools", ToolNode(self.tools))  # Has all tools, agent limits exposure
        
        # Entry
        graph.set_entry_point("router")
        
        # Tier 1: Router → branches
        graph.add_conditional_edges(
            "router",
            self._route_intent,
            {
                "meta": "meta_respond",
                "chat": "chat_respond",
                "research": "research_router",  # Tier 2 router
            }
        )
        
        # meta/chat → END (fast paths)
        graph.add_edge("meta_respond", END)
        graph.add_edge("chat_respond", END)
        
        # Tier 2: Research router → agent (with strategy set)
        graph.add_edge("research_router", "agent")
        
        # Tier 3: agent → tools or END
        graph.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "tools": "tools",
                "end": END,
            }
        )
        
        # tools → back to agent (ReAct loop)
        graph.add_edge("tools", "agent")
        
        return graph.compile()
    
    # --------------------------------------------------------
    # Node Implementations
    # --------------------------------------------------------
    
    def _router_node(self, state: AgentState) -> dict:
        """
        Tier 1: Classify intent with regex first, LLM fallback.
        Expanded regex covers ~80% of queries without an LLM call.
        """
        user_msg = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_msg = msg.content
                break
        
        lower = user_msg.lower().strip()
        
        # ── Follow-up detection (if conversation has prior turns) ──
        if len(state["messages"]) > 1:
            follow_ups = [
                "dímelo", "dime", "explica", "explain", "en castellano",
                "en español", "in spanish", "in english", "en inglés",
                "em português", "in portuguese", "en français",
                "más detalle", "more detail", "profundiza", "elaborate",
                "por qué", "why", "cómo", "how", "ejemplo", "example",
                "sí", "yes", "continúa", "continue", "go on",
                "por favor", "please", "sim", "oui",
            ]
            if any(p in lower for p in follow_ups):
                return {"intent": "research"}
        
        # ── Meta / greeting detection (only for first message) ──
        meta_triggers = [
            "hola", "hello", "hi ", "hey", "buenos", "buenas",
            "/help", "/info", "/quit", "qué modelo", "what model",
            "quién eres", "who are you", "qué eres", "what are you",
            "qué puedes", "what can you", "ayuda", "help",
            "gracias", "thanks", "thank you", "adiós", "goodbye",
        ]
        if len(state["messages"]) <= 1:
            for trigger in meta_triggers:
                if lower.startswith(trigger) or trigger in lower:
                    return {"intent": "meta"}
        
        # ── Research detection (skip LLM for obvious queries) ──
        # Questions with scientific keywords → research
        research_patterns = [
            # Direct search commands
            "busca", "buscar", "find", "search", "look for", "papers about",
            "artículos sobre", "papers on", "studies on", "research on",
            # Scientific question patterns
            "species", "especie", "ecology", "ecología", "habitat",
            "population", "conservation", "biodiversity", "ecosystem",
            "climate", "marine", "coral", "forest", "freshwater",
            "what species", "qué especie", "which papers", "qué papers",
            "how does", "cómo", "what is the effect", "cuál es",
            "compare", "comparar", "analyze", "analizar", "summarize",
            "related to", "connection between", "relationship between",
            # Multilingual research triggers
            "qué sabes", "dime sobre", "dime que", "información sobre",
            "basándote", "bibliografía", "base de datos", "tu base",
            "o que você sabe", "informação sobre", "pesquisar",
            "remote sensing", "machine learning", "deep learning",
        ]
        for pattern in research_patterns:
            if pattern in lower:
                return {"intent": "research"}
        
        # Question marks with 5+ words → likely research
        if "?" in user_msg and len(user_msg.split()) >= 5:
            return {"intent": "research"}
        
        # ── LLM fallback for truly ambiguous queries ──
        try:
            response = self.llm_fast.invoke([
                SystemMessage(content=ROUTER_PROMPT),
                HumanMessage(content=user_msg),
            ])
            
            intent = response.content.strip().lower().rstrip(".")
            
            if intent in ("search", "analyze"):
                return {"intent": "research"}
            elif intent == "meta":
                return {"intent": "meta"}
            else:
                return {"intent": "chat"}
                
        except Exception as e:
            logger.warning(f"Router failed, defaulting to research: {e}")
            return {"intent": "research"}
    
    def _meta_node(self, state: AgentState) -> dict:
        """
        Instant response for meta/system queries. No LLM needed.
        """
        user_msg = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_msg = msg.content.lower()
                break
        
        # Greetings
        greetings = ["hola", "hello", "hi", "hey", "buenos", "buenas"]
        if any(g in user_msg for g in greetings):
            response = (
                f"¡Hola! 🌿 Soy EcoloGRAPH, tu asistente de investigación ecológica.\n\n"
                f"Puedo ayudarte a:\n"
                f"• **Buscar papers** en 43 dominios científicos\n"
                f"• **Clasificar textos** por dominio ecológico\n"
                f"• **Buscar info de especies** (FishBase, GBIF, IUCN)\n"
                f"• **Encontrar conexiones** entre dominios\n"
                f"• **Generar hipótesis** de investigación\n\n"
                f"Modelo activo: **{self.model_name}**\n"
                f"¡Pregúntame lo que quieras!"
            )
            return {"messages": [AIMessage(content=response)]}
        
        # Model info
        if any(k in user_msg for k in ["modelo", "model", "llm"]):
            response = (
                f"Estoy usando el modelo **{self.model_name}** "
                f"(detectado de {self.base_url}).\n"
                f"Owner: {self.model_info.get('owned_by', 'N/A')}"
            )
            return {"messages": [AIMessage(content=response)]}
        
        # Capabilities
        if any(k in user_msg for k in ["puedes", "can you", "help", "ayuda", "capab"]):
            tools_desc = get_tool_descriptions()
            response = (
                f"🌿 **EcoloGRAPH** — Asistente de investigación ecológica\n\n"
                f"{tools_desc}\n\n"
                f"Modelo: {self.model_name} | "
                f"Dominios: 43 | Modo: Two-Tier LangGraph"
            )
            return {"messages": [AIMessage(content=response)]}
        
        # Generic meta fallback
        response = (
            f"Soy EcoloGRAPH, asistente de investigación ecológica.\n"
            f"Modelo: {self.model_name}\n"
            f"Tools: {len(self.tools)} disponibles\n"
            f"Pregúntame sobre papers, especies, o dominios ecológicos."
        )
        return {"messages": [AIMessage(content=response)]}
    
    def _research_router_node(self, state: AgentState) -> dict:
        """
        Tier 2: Classify research intent into strategy to limit tools.
        Reduces token usage ~40% by only sending relevant tool schemas.
        """
        user_msg = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_msg = msg.content
                break
        
        lower = user_msg.lower().strip()
        
        # ── Pattern-based strategy detection (~60% of queries) ──
        # Search-focused queries
        if any(kw in lower for kw in ["search", "find papers", "papers about", "buscar", "artículos"]):
            return {"strategy": "search"}
        
        # Graph-focused queries  
        if any(kw in lower for kw in ["graph", "relationships", "connections", "network", "related papers", "similar to"]):
            return {"strategy": "graph"}
        
        # External API queries
        if any(kw in lower for kw in ["species info", "fishbase", "gbif", "iucn", "taxonomy", "conservation status"]):
            return {"strategy": "external"}
        
        # Inference/hypothesis queries
        if any(kw in lower for kw in ["hypothesis", "cross-domain", "connections between", "inference", "predict", "link"]):
            return {"strategy": "inference"}
        
        # ── LLM fallback for ambiguous queries ──
        try:
            response = self.llm_fast.invoke([
                SystemMessage(content=STRATEGY_ROUTER_PROMPT),
                HumanMessage(content=user_msg),
            ])
            
            strategy = response.content.strip().lower().rstrip(".")
            
            if strategy in ("search", "graph", "external", "inference", "full"):
                return {"strategy": strategy}
            else:
                return {"strategy": "full"}  # Default to all tools
                
        except Exception as e:
            logger.warning(f"Strategy router failed, using full toolset: {e}")
            return {"strategy": "full"}
    
    def _chat_node(self, state: AgentState) -> dict:
        """
        Tier 1: Direct LLM response WITHOUT tool schemas.
        Fast — only system prompt (~80 tokens) + user message.
        """
        messages = state["messages"]
        
        response = self.llm_fast.invoke([
            SystemMessage(content=CHAT_SYSTEM),
            *[m for m in messages if isinstance(m, (HumanMessage, AIMessage))],
        ])
        
        return {"messages": [response]}
    
    def _agent_node(self, state: AgentState) -> dict:
        """
        Tier 2/3: Full agent with strategy-specific tool subset.
        Uses only relevant tools based on query classification.
        """
        messages = state["messages"]
        strategy = state.get("strategy", "full")
        
        # Get appropriate tool subset for this strategy
        tools = get_tools_for_strategy(strategy)
        
        # Create LLM instance with limited tools
        llm_with_subset = self.llm_fast.bind_tools(tools)
        
        # Build tool descriptions for this subset only
        tool_desc_lines = ["Available tools:"]
        for t in tools:
            tool_desc_lines.append(f"  • {t.name}: {t.description.split(chr(10))[0].strip()}")
        tool_descriptions = "\n".join(tool_desc_lines)
        
        # Ensure system prompt with subset tools
        if not messages or not isinstance(messages[0], SystemMessage):
            system_msg = SystemMessage(content=AGENT_SYSTEM.format(
                tool_descriptions=tool_descriptions
            ))
            messages = [system_msg] + messages
        
        response = llm_with_subset.invoke(messages)
        return {"messages": [response]}
    
    def _should_continue(self, state: AgentState) -> Literal["tools", "end"]:
        """Decide whether agent needs more tools or can finish."""
        last_message = state["messages"][-1]
        
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "end"
    
    def _route_intent(self, state: AgentState) -> str:
        """Route based on classified intent."""
        return state.get("intent", "chat")
    
    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------
    
    def _build_messages(
        self, question: str, history: list[dict] = None
    ) -> list[BaseMessage]:
        """Convert chat history + current question into LangChain messages."""
        messages = []
        if history:
            # Include last 10 turns for context (avoid token overflow)
            for msg in history[-10:]:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
        messages.append(HumanMessage(content=question))
        return messages
    
    def ask(self, question: str, history: list[dict] = None) -> str:
        """Ask a question and get an answer."""
        initial_state: AgentState = {
            "messages": self._build_messages(question, history),
            "intent": "",
        }
        
        try:
            final_state = self.graph.invoke(
                initial_state,
                config={"recursion_limit": self.max_iterations * 2 + 2}
            )
            
            for msg in reversed(final_state["messages"]):
                if isinstance(msg, AIMessage):
                    return msg.content
            
            return "No pude generar una respuesta. Reformula tu pregunta."
            
        except Exception as e:
            logger.error(f"Agent error: {e}")
            return f"Error: {e}"
    
    def ask_streaming(self, question: str, history: list[dict] = None):
        """
        Ask with streaming. Yields (event_type, content) tuples.
        event_type: 'routing', 'tool_call', 'tool_result', 'answer', 'error'
        """
        initial_state: AgentState = {
            "messages": self._build_messages(question, history),
            "intent": "",
        }
        
        try:
            for event in self.graph.stream(
                initial_state,
                config={"recursion_limit": self.max_iterations * 2 + 2}
            ):
                for node_name, state_update in event.items():
                    # Show routing decision
                    if node_name == "router":
                        intent = state_update.get("intent", "")
                        route_labels = {
                            "meta": "⚡ Respuesta directa",
                            "chat": "💬 Respuesta conversacional",
                            "research": "🔬 Búsqueda en base de datos",
                        }
                        label = route_labels.get(intent, intent)
                        yield ("routing", f"  [{label}]")
                        continue
                    
                    messages = state_update.get("messages", [])
                    for msg in messages:
                        if isinstance(msg, AIMessage):
                            if hasattr(msg, "tool_calls") and msg.tool_calls:
                                for tc in msg.tool_calls:
                                    yield ("tool_call", f"  🔧 {tc['name']}({tc['args']})")
                            elif msg.content:
                                yield ("answer", msg.content)
                        
                        elif isinstance(msg, ToolMessage):
                            preview = msg.content[:150].replace("\n", " ")
                            yield ("tool_result", f"  📊 {preview}...")
                            
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield ("error", f"Error: {e}")
    
    def get_info(self) -> dict:
        """Get agent configuration info."""
        return {
            "model": self.model_name,
            "model_info": self.model_info,
            "base_url": self.base_url,
            "tools": [t.name for t in self.tools],
            "max_iterations": self.max_iterations,
            "architecture": "Two-Tier LangGraph (router → fast/full)",
        }
