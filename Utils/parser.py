import json
import os
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "your_api_key_here"))


class ParsedIntent(BaseModel):
    distance_km: float = Field(description="The requested distance in kilometers")
    activity: str = Field(description="The type of activity (run, cycle, etc)")
    terrain: str = Field(description="The preferred terrain style")
    location: str = Field(description="The specific city or area")


def parse_athlete_input(user_input: str) -> ParsedIntent:
    """
    The 'Applied Gen AI' layer that translates natural language to
    structured parameters for the routing engine.
    """
    messages = [
        {"role": "system", "content": "You are a Strava Routing Expert. Extract JSON."},
        {"role": "user", "content": f"Parse this request: {user_input}"},
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        response_format={"type": "json_object"},  # JSON mode [web:10]
    )

    try:
        raw_data = json.loads(response.choices[0].message.content)
        return ParsedIntent(**raw_data)
    except (ValidationError, json.JSONDecodeError) as e:
        print(f"Error parsing AI response: {e}")
        return ParsedIntent(
            distance_km=5.0, activity="run", terrain="balanced", location="Berlin"
        )


if __name__ == "__main__":
    athlete_request = "Find me a hilly 12k trail run near Grunewald, Berlin"
    structured_params = parse_athlete_input(athlete_request)
    print(structured_params.model_dump())
