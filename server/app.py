# server/app.py
import dotenv
dotenv.load_dotenv()

from openenv.core.env_server import create_fastapi_app

from models import MovingTargetAction, MovingTargetObservation
from server.Moving_Target_environment import MovingTargetEnv

app = create_fastapi_app(MovingTargetEnv, MovingTargetAction, MovingTargetObservation)
from pydantic import BaseModel

class ConstraintPayload(BaseModel):
    constraint: str

@app.post("/set_constraint")
def set_constraint(payload: ConstraintPayload):
    MovingTargetEnv.ground_truth_constraint = payload.constraint
    return {"status": "success"}

@app.get("/")
def health_check():
    return {"status": "ok"}


def main() -> None:
    """OpenEnv server entry point used by validators and scripts."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
