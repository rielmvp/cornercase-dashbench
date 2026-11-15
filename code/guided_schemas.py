"""JSON schemas for vLLM guided decoding (optional, --guided-json)."""
Q4_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "unusual_or_dangerous_event_detected": {"type": "boolean"},
    },
    "required": ["reasoning", "unusual_or_dangerous_event_detected"],
    "additionalProperties": False,
}

PROMPT_TYPE_SCHEMAS = {"Q4": Q4_SCHEMA}
