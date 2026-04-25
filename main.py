import pprint
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from state import AgentState
from personaAgent import persona_node
from concierge import concierge_node, tools
from watchdog import watchdog_node

def route_concierge_output(state: AgentState):
    """
    Routings for the Concierge.
    If the LLM outputted a tool call (like getMerchant or ask_watchdog), we go to the ToolNode.
    If the LLM did not output a tool call, we assume it's done.
    """
    messages = state.get("messages", [])
    if not messages:
        return END

    last_message = messages[-1]
    
    # Check if the LLM returned any tool calls
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "tools"
    
    # Otherwise, it gave a text response, which means the episode is ending.
    return END

# 1. Initialize the Graph with our AgentState structure
workflow = StateGraph(AgentState)

# 2. Add the custom nodes we scripted
workflow.add_node("persona", persona_node)
workflow.add_node("concierge", concierge_node)

# We use the built-in LangGraph ToolNode to automatically execute any tools 
# the Concierge requests (using the tools array we exported from concierge.py)
tool_executor = ToolNode(tools)
workflow.add_node("tools", tool_executor)

# Our Watchdog agent intercepts the raw API outputs
workflow.add_node("watchdog", watchdog_node)

# 3. Define the Edges (The Flow of the Application)
workflow.add_edge(START, "persona")             # Start by getting the user's constraints
workflow.add_edge("persona", "concierge")       # Pass constraints to the Concierge

# The Concierge decides whether to use a tool or finish
workflow.add_conditional_edges(
    "concierge",
    route_concierge_output,
    {
        "tools": "tools",  # If a tool is called, execute it
        END: END           # If it just replies, we are finished
    }
)

# After the HTTP tool executes, ALWAYS pass the raw data to the Watchdog to check for drift
workflow.add_edge("tools", "watchdog")

# Once the Watchdog finishes its analysis, loop back to the Concierge!
workflow.add_edge("watchdog", "concierge")

# 4. Compile the Graph
app = workflow.compile()

import requests

# --- MAIN EXECUTION SIMULATION ---
if __name__ == "__main__":
    print("\n[SYSTEM] Compiling and starting the LangGraph Moving Target Simulation...")
    print("[SYSTEM] Make sure the OpenEnv server is running on localhost:8000!\n")
    
    EPISODES = 5
    lifelong_memory = {}
    total_lifetime_score = 0
    prev_episode_summary = ""  # Will be built from the last episode's results
    
    for episode in range(EPISODES):
        print(f"\n=======================================================")
        print(f"               STARTING EPISODE {episode + 1}/{EPISODES}")
        print(f"=======================================================\n")
        
        # 1. Scramble the Environment completely before starting
        print("[SYSTEM] Sending /reset to OpenEnv Server to randomize schemas...")
        try:
            requests.post("http://localhost:8000/reset")
        except Exception as e:
            print(f"[SYSTEM ERROR] Could not reach server: {e}")
            break

        # 2. Give the agent a clean chat history, but let it keep its memories AND its score feedback!
        initial_state = {
            "messages": [],
            "current_merchant": "",
            "last_known_schema": lifelong_memory,
            "drift_detected": False,
            "reward_score": 0.0,
            "prev_episode_summary": prev_episode_summary,
            "step_count": 0
        }

        # 3. Run the LangGraph
        episode_score = 0
        for output in app.stream(initial_state, stream_mode="updates"):
            for node_name, state_update in output.items():
                print(f"\n--- Output from Node: {node_name} ---")
                
                if "messages" in state_update:
                    last_msg = state_update["messages"][-1]
                    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                        print(f"[Tool Call Requested]: {last_msg.tool_calls[0]['name']}({last_msg.tool_calls[0].get('args', {})})")
                    elif hasattr(last_msg, "name") and getattr(last_msg, "name") in [t.name for t in tools]:
                        # Extract the reward string
                        content_str = last_msg.content.strip()
                        print(f"[HTTP Tool Result]: {content_str}")
                        
                        import re
                        match = re.search(r"\(Environment Reward: ([-\d.]+)\)", content_str)
                        if match:
                            episode_score += float(match.group(1))
                    else:
                        print(f"[{node_name.capitalize()} Message]: {last_msg.content}")
                
                if "drift_detected" in state_update:
                    if state_update["drift_detected"]:
                        print("--> WARNING: Watchdog flagged a drift!")
                        
                # 4. Save the memory generated from this node so it carries to the next episode
                if "last_known_schema" in state_update:
                    lifelong_memory = state_update["last_known_schema"]
                
        print(f"\n[SYSTEM] Episode {episode + 1} Complete. Episode Score: {episode_score}")
        total_lifetime_score += episode_score
        
        # Build RL feedback summary for the NEXT episode's Concierge
        prev_episode_summary = (
            f"In the last episode (Episode {episode + 1}), you scored {episode_score} points. "
            f"{'Good job — maintain this strategy.' if episode_score >= 0 else 'You lost points. Try checking merchant policies more carefully before placing orders and avoid reduntant watchdog calls.'}"
        )

    print(f"\n=======================================================")
    print(f"[SYSTEM] ALL EPISODES COMPLETE. TOTAL RL SCORE: {total_lifetime_score}")
    print(f"=======================================================\n")
