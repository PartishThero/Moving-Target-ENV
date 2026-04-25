import json
import os
import random
from typing import Any, Optional

from openenv.core.env_server import Environment
from langchain_openai import ChatOpenAI
import dotenv
dotenv.load_dotenv()

from models import MovingTargetAction, MovingTargetObservation, MovingTargetEnvironmentState

# Potential fields that a merchant might demand
POTENTIAL_FIELDS = [
    "item", "price", "dietary_notes", "delivery_address", 
    "contact_number", "customer_name", "discount_code", 
    "special_instructions", "payment_method", "quantity",
]

class MovingTargetEnv(Environment[MovingTargetAction, MovingTargetObservation, MovingTargetEnvironmentState]):
    # CLASS VARIABLES to persist across FastAPI HTTP request instantiations
    ground_truth = {}
    _global_step_count = 0
    ground_truth_constraint = ""
    _directory_rewarded = False


    def __init__(self):
        super().__init__()
        self.initial_merchants = ["PizzaPalace","VeganBistro", "GreenLeaf_Dining", "SpiceSymphony", "BurgerBar", "SushiStation","CurryHouse","PastaParadise","TacoTown","NoodleNook"]
        # Only initialize once across the whole server
        if not MovingTargetEnv.ground_truth:
            self._initialize_world()

    def _generate_random_schema(self):
        """Creates a completely random API schema."""
        # Pick 2 to 5 required fields randomly
        num_fields = random.randint(2, 5)
        # Always guarantee 'item' is inside since it's food ordering
        fields = {"item",} 
        while len(fields) < num_fields:
            fields.add(random.choice(POTENTIAL_FIELDS))
            
        return {
            "api_version": f"v{random.randint(1, 5)}.{random.randint(0, 9)}",
            "required_fields": list(fields),
            "refund_policy": random.choice([
                "Always fully refundable",
                "Non-refundable for orders under $50",
                "Strictly non-refundable",
                "Flexible within 10 minutes of ordering"
            ]),
            "other_policies":random.choice(["Pet-Friendly", "Not Pet-Friendly", "Vegan Options Available", "Gluten-Free Options Available", "Nut-Free Options Available", "Halal Options Available", "Low-Carb Options Available"]),
            "average_price_for_1": random.choice(["$30", "$50", "$100","$20","$80" ]),
        }

    def _drift_schema(self, merchant_name: str):
        """Mutates an existing schema to simulate 'drift'."""
        schema = MovingTargetEnv.ground_truth[merchant_name]
        
        # 50% chance to add a field, 50% to remove one (if more than 2)
        if random.random() > 0.5 or len(schema["required_fields"]) <= 2:
            # Add a field
            new_field = random.choice([f for f in POTENTIAL_FIELDS if f not in schema["required_fields"]])
            if new_field not in schema["required_fields"]:
                schema["required_fields"].append(new_field)
        else:
            # Remove a field (but never remove 'item')
            removable = [f for f in schema["required_fields"] if f != "item"]
            if removable:
                schema["required_fields"].remove(random.choice(removable))
        
        schema["api_version"] = f"v{random.randint(1, 9)}.{random.randint(0, 9)}-updated"
        MovingTargetEnv.ground_truth[merchant_name] = schema

    def _initialize_world(self):
        """Spawns the initial world."""
        MovingTargetEnv.ground_truth = {}
        for m in self.initial_merchants:
            MovingTargetEnv.ground_truth[m] = self._generate_random_schema()

    def reset(self, seed=None, episode_id=None, **kwargs):
        """Standard OpenEnv reset method."""
        MovingTargetEnv._global_step_count = 0
        MovingTargetEnv._directory_rewarded = False
        self._initialize_world()
        return MovingTargetObservation(
            data="Environment Reset. A new chaotic world of shifting API schemas has been generated.",
            status=200,
        )

    def step(self, action: MovingTargetAction, **kwargs):
        """Route the action to the appropriate tool."""
        MovingTargetEnv._global_step_count += 1

        if action.tool == "get_merchants":
            reward = 0
            if not MovingTargetEnv._directory_rewarded: 
                reward = 5.0
                MovingTargetEnv._directory_rewarded = True
            return MovingTargetObservation(
                data=json.dumps(list(MovingTargetEnv.ground_truth.keys())),
                status=200,
                reward=reward
            )

        elif action.tool == "ask_watchdog":
            return self._ask_watchdog(action.merchant_name)
        
        elif action.tool == "place_order":
            return self._place_order(action.merchant_name, action.payload or {})
            
        else:
            return MovingTargetObservation(
                data=f"Unknown tool: '{action.tool}'.",
                status=400,
                reward=-5.0
            )

    # TOOL 1: The Scout
    def _ask_watchdog(self, merchant_name: str) -> MovingTargetObservation:
        """Check API docs and refund policies for a merchant. Causes drift!"""
        
        # If the agent invents a completely new merchant, "spawn" it natively
        if merchant_name not in MovingTargetEnv.ground_truth:
            MovingTargetEnv.ground_truth[merchant_name] = self._generate_random_schema()
        else:
            # If it already exists, there is a 30% chance the API shifts beneath their feet!
            if random.random() < 0.30:
                self._drift_schema(merchant_name)

        return MovingTargetObservation(
            data=json.dumps(MovingTargetEnv.ground_truth[merchant_name]),
            status=200,
            reward=-2.0
        )

    # TOOL 2: The Executioner
    def _place_order(self, merchant_name: str, payload: dict) -> MovingTargetObservation:
        """Place the order. Fails if the payload doesn't match the current schema."""
        if merchant_name not in MovingTargetEnv.ground_truth:
            return MovingTargetObservation(
                data="Merchant not found in directory. Use ask_watchdog first.", 
                status=404,
                reward=-20.0
            )

        real_rules = MovingTargetEnv.ground_truth[merchant_name]

        # Validate the agent's attempt against Ground Truth
        for field in real_rules["required_fields"]:
            if field not in payload:
                return MovingTargetObservation(
                    data=f"API REJECTED: Missing required field '{field}'. Schema may have drifted!",
                    status=400,
                    reward=-50.0
                )
        
        # Check for unexpected fields (Strict Server Validation)
        for field in payload.keys():
            if field not in real_rules["required_fields"]:
                return MovingTargetObservation(
                    data=f"API REJECTED: Unknown field '{field}' provided. Schema may have drifted!",
                    status=400,
                    reward=-50.0
                )

        # ENVIRONMENT JUDGE: Evaluate Business Logic Compliance
        if MovingTargetEnv.ground_truth_constraint:
            try:
                evaluator = ChatOpenAI(
                    model=os.getenv("MODEL_NAME"),
                    temperature=0.0,
                    base_url="https://openrouter.ai/api/v1",
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                )
                eval_prompt = f"""
                You are the ultimate environment judge.
                User's True Constraints: {MovingTargetEnv.ground_truth_constraint}
                The Agent just placed an order with '{merchant_name}'. 
                
                The Merchant's actual data:
                - Refund Policy: {real_rules.get("refund_policy", "Unknown")}
                - Average Price for 1: {real_rules.get("average_price_for_1", "Unknown")}
                - Other Policies (Pet-Friendly, Halal, etc.): {real_rules.get("other_policies", "Unknown")}
                
                The Agent's payload was: {payload}

                Question: Does this order violate the user's constraints regarding Refund Policy, Budget, or Diet/Special needs (like Pet-Friendly)?
                Respond with EXACTLY 'YES' if it violates (even partially), or 'NO' if it matches perfectly.
                """
                eval_result = evaluator.invoke(eval_prompt)
                if "YES" in eval_result.content.upper():
                    return MovingTargetObservation(
                        data="API Accepted, BUT ORDER VIOLATES USER CONSTRAINTS! Refund policies or dietary needs were ignored.",
                        status=400,
                        reward=-100.0,
                        done=True
                    )
            except Exception as e:
                # Judge failure should not block the episode — log and continue
                print(f"[Environment Judge Error] {e}")

        return MovingTargetObservation(
            data="Order successful! API Schema perfectly matched and constraints were met.", 
            status=200, 
            done=True, 
            reward=50.0
        )

    @property
    def state(self) -> MovingTargetEnvironmentState:
        return MovingTargetEnvironmentState(
            episode_id=None,
            step_count=MovingTargetEnv._global_step_count,
        )