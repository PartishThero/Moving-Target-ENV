from langchain_core.messages.tool import tool_call
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from state import AgentState
import os
import dotenv
dotenv.load_dotenv()
import requests

def _safe_parse(response):
    """Safely extract data and reward from server response, even if it crashes."""
    try:
        response_data = response.json()
        data = response_data["observation"].get("data", "Server returned empty response.")
        reward = response_data["observation"].get("reward") or response_data.get("reward", 0)
        return data, reward
    except Exception:
        return f"SERVER ERROR: {response.status_code} - {response.text[:200]}", -50.0

@tool
def ask_watchdog(merchant_name: str):
    """ALWAYS use this tool first to check if the API docs and refund policies for a merchant have changed."""
    url = "http://localhost:8000/step"
    payload = {
        "action": {
            "tool": "ask_watchdog", 
            "merchant_name": merchant_name
        }
    }
    response = requests.post(url, json=payload)
    data, reward = _safe_parse(response)
    return f"Observation: {data} \n(Environment Reward: {reward})"

@tool
def getMerchant():
    """Use this tool to get a list of currently available merchants in the environment if you don't want to make one up."""
    url = "http://localhost:8000/step"
    payload = {
        "action": {
            "tool": "get_merchants", 
            "merchant_name": "directory" # placeholder required by model
        }
    }
    response = requests.post(url, json=payload)
    data, reward = _safe_parse(response)
    return f"Observation: {data} \n(Environment Reward: {reward})"

@tool
def place_order(merchant_name: str = "", payload: dict = None):
    """Place an order for a specific merchant. 
    CRITICAL CALL SIGNATURE: You MUST pass TWO separate arguments:
    1. merchant_name (string): The exact name of the merchant (e.g. 'GreenLeaf_Dining'). This is NOT inside the payload dict.
    2. payload (dict): A dictionary containing EXACTLY the required fields from ask_watchdog, no more, no less.
    Example: place_order(merchant_name='GreenLeaf_Dining', payload={'item': 'Salad', 'contact_number': '123-456-7890'})
    """
    url = "http://localhost:8000/step"
    req_body = {
        "action": {
            "tool": "place_order", 
            "merchant_name": merchant_name,
            "payload": payload or {}
        }
    }
    response = requests.post(url, json=req_body)
    data, reward = _safe_parse(response)
    return f"Observation: {data} \n(Environment Reward: {reward})"

tools = [ask_watchdog, getMerchant, place_order]

# 2. Set up the Brain
concierge_llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME"),
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
).bind_tools(tools)

def concierge_node(state: AgentState):
    """
    The Concierge Node: Reads the user's request and decides whether to 
    call a tool or ask a clarifying question.
    """
    # Grab the conversation history (which includes the Persona's request)
    messages = state["messages"]
    
    # Pull in RL feedback from the previous episode (if any)
    prev_summary = state.get("prev_episode_summary", "")
    rl_context = ""
    if prev_summary:
        rl_context = f"PREVIOUS EPISODE PERFORMANCE FEEDBACK: {prev_summary} Use this to do better this episode. "

    # Give the Concierge its system prompt
    system_instruction = SystemMessage(content=(
        "You are an elite E-Commerce AI Concierge. Your job is to fulfill the user's request. "
        + rl_context +
        "CRITICAL RULES: "
        "1. SILENT EXECUTION (DO NOT TALK TO THE USER): You are strictly forbidden from conversing with, asking questions of, or waiting for permission from the user during the search and order process. You must ONLY output tool calls until the task is completely finished. "
        "2. THE ACTION LOOP (NO GUESSING): You are strictly forbidden from guessing a merchant's schema, policies, or prices. You MUST physically execute the 'ask_watchdog' tool to evaluate a merchant. Do NOT summarize or reject a merchant you haven't explicitly called the watchdog for. Before calling ask_watchdog, you MUST read your own previous tool calls. If you see you have already called ask_watchdog for a specific merchant name in this conversation, you are strictly forbidden from calling it again. Skip to the next merchant."
        "3. INVENT MISSING DATA: If the API schema requires fields the user did not explicitly provide (e.g., customer_name, contact_number, delivery_address, discount_code), you MUST completely invent fake details to fill the payload. Do not ask the user for this information. "
        "4. STRICT WATCHDOG USAGE: Call 'ask_watchdog' EXACTLY ONCE per new merchant. Once evaluated, consider it 'known'. "
        "5. MERCHANT COMPLIANCE CHECK (THE PRE-FLIGHT): After receiving the watchdog schema, evaluate it against the  user's ALL requests: "
        "   - 'average_price_for_1': Must be <= the user's budget. "
        "   - 'other_policies' / 'refund_policy': Must satisfy the user's specific dietary or pet constraints. "
        "   - Make sure the dietary conditions are forwarded to the merchant via policy or dietary notes IF in the schema else the match is a FAIL"
        "   - EXCEPTION: If the user states they have NO dietary restrictions, any food policy (Halal, Vegan, Gluten-Free, etc.) is perfectly acceptable. Do not reject a merchant for offering these. "
        "   - MISSING POLICIES: If a specific user constraint is completely missing from the schema, assume it matches and proceed. "
        "6. PASS OR FAIL ROUTING: "
        "   - IF FAIL: Blacklist the merchant. Call 'getMerchant' to see the list, pick a NEW merchant you haven't checked yet, and immediately call 'ask_watchdog' on it. "
        "   - IF PASS: Immediately call the 'place_order' tool. "
        "7. PAYLOAD PRECISION: The 'payload' dictionary in 'place_order' MUST contain EXACTLY the fields listed in the watchdog's 'required_fields' array. No more, no less. "
        "8. TERMINATION CONDITION: 'getMerchant' returns a list of vendors. If you have executed 'ask_watchdog' a number of times equal to the number of vendors on that list, you have exhausted all options. You MUST STOP looping and output a final message to the user."
        "9. ERROR HANDLING: "
        "   [A] 'Missing required field X': add fake field X to the payload and retry place_order immediately. "
        "   [B] 'Unknown field X provided': remove field X from the payload and retry place_order immediately. "
        "   [C] 'VIOLATES USER CONSTRAINTS': Blacklist merchant, use getMerchant, pick a new vendor, and call ask_watchdog. "
        "   [D] 'SERVER ERROR': Retry the exact same place_order call once more."
            ))
    
    # Invoke the LLM with the instruction + the conversation history
    response = concierge_llm.invoke([system_instruction] + messages)
    
    schema_update = {
        "messages": [response],
        "step_count": state.get("step_count", 0) + 1
    }

    # Check if the LLM decided to call a tool
    if hasattr(response,"tool_calls") and len(response.tool_calls)>0:
        tool_call=response.tool_calls[0]
        if 'merchant_name' in tool_call['args']:
            schema_update['current_merchant']=tool_call['args']['merchant_name']
            
    # Return the delta to update the state
    return schema_update

# --- QUICK TEST ---
if __name__ == "__main__":
    from langchain_core.messages import HumanMessage
    
    # We fake the state by pretending the Persona just gave us this request
    fake_state = {
        "messages": [HumanMessage(content="I need a Vegan meal under $40, and it must be flexible/refundable.")]
    }
    
    print("Running Concierge Node Test...\n")
    result = concierge_node(fake_state)
    
    response_message = result["messages"][0]
    
    print("--- LLM Decision ---")
    # If the LLM followed instructions, it won't return text. It will return a Tool Call!
    if hasattr(response_message, 'tool_calls') and len(response_message.tool_calls) > 0:
        tool_call = response_message.tool_calls[0]
        tool_name = tool_call['name']
        tool_args = tool_call['args']
        
        print(f"SUCCESS! The Concierge decided to call a tool: {tool_name}")
        print(f"Arguments: {tool_args}")
        
        print("\n--- Executing HTTP Tool Request ---")
        try:
            if tool_name == 'getMerchant':
                env_reply = getMerchant.invoke(tool_args)
                print(f"Directory response from Localhost: {env_reply}")
            elif tool_name == 'ask_watchdog':
                env_reply = ask_watchdog.invoke(tool_args)
                print(f"Watchdog response from Localhost: {env_reply}")
            elif tool_name == 'place_order':
                env_reply = place_order.invoke(tool_args)
                print(f"Order response from Localhost: {env_reply}")
        except Exception as e:
            print(f"Error calling the server: {e}")
            
    else:
        print("The Concierge just replied with text:")
        print(response_message.content)