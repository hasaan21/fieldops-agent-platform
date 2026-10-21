"""Domain vocabulary for the synthetic field-service world.

Everything here is fictional. The brands, error codes and procedures are invented so that RAG
answers can only come from the generated manuals, never from the model's pre-training.
"""

from __future__ import annotations

CATEGORIES = ["plumbing", "electrical", "hvac", "gas", "fire_safety", "general_maintenance"]
URGENCIES = ["emergency", "high", "normal", "low"]

# asset type -> (category, fictional model names)
ASSET_TYPES: dict[str, tuple[str, list[str]]] = {
    "boiler": ("gas", ["Thermacore X200", "Thermacore X350", "Heatwell Combi 28"]),
    "air_conditioner": ("hvac", ["Coolstream AC-12", "Coolstream AC-18"]),
    "heat_pump": ("hvac", ["Verdant HP-9", "Verdant HP-14"]),
    "water_heater": ("plumbing", ["Aquaflow WH-80", "Aquaflow WH-150"]),
    "fire_alarm_panel": ("fire_safety", ["Sentinel FP-4", "Sentinel FP-8"]),
    "electrical_panel": ("electrical", ["Voltix DB-12", "Voltix DB-24"]),
}

# Mean repair duration in hours per category (used for the duration-forecast model).
BASE_DURATION_HOURS = {
    "plumbing": 1.8,
    "electrical": 2.2,
    "hvac": 2.6,
    "gas": 2.0,
    "fire_safety": 1.5,
    "general_maintenance": 1.2,
}

# Request templates: (category, urgency, template). {asset} and {extra} are filled in.
REQUEST_TEMPLATES: list[tuple[str, str, str]] = [
    ("gas", "emergency", "I can smell gas near the {asset} in the kitchen, what should I do?"),
    ("gas", "emergency", "Strong gas smell in the plant room, the {asset} has shut off."),
    ("gas", "high", "Our {asset} keeps locking out and showing an error code {code}."),
    ("gas", "normal", "Can we book the annual service for our {asset}?"),
    ("gas", "low", "Looking for a quote to replace the {asset} sometime next quarter."),
    ("plumbing", "emergency", "Water is pouring through the ceiling from the {asset} cupboard!"),
    ("plumbing", "high", "No hot water since this morning, the {asset} shows {code}."),
    ("plumbing", "normal", "The tap in the staff toilet has been dripping for a week."),
    ("plumbing", "low", "Could someone look at the slow drain in the canteen when convenient?"),
    ("electrical", "emergency", "Sparks and a burning smell from the {asset}, power is off."),
    ("electrical", "high", "Half the office has lost power, the {asset} tripped and won't reset."),
    ("electrical", "normal", "We need the EICR inspection done on our {asset} this month."),
    ("electrical", "low", "Please quote for adding two sockets in the meeting room."),
    ("hvac", "high", "The {asset} in the server room has stopped cooling, code {code}."),
    ("hvac", "normal", "The {asset} is making a rattling noise, can you send someone?"),
    ("hvac", "low", "We'd like a filter change on the {asset} at some point."),
    ("fire_safety", "emergency",
     "The {asset} is in alarm and we can't silence it, no fire visible."),
    ("fire_safety", "high", "The {asset} is showing a fault light, code {code}."),
    ("fire_safety", "normal", "Book the quarterly fire alarm test for the {asset} please."),
    ("general_maintenance", "normal", "A door closer on the fire exit is broken."),
    ("general_maintenance", "low", "Could you quote for repainting the reception area?"),
]

# Fictional error codes per asset type: code -> (meaning, fix, safe_for_customer)
ERROR_CODES: dict[str, dict[str, tuple[str, str, bool]]] = {
    "boiler": {
        "F22": ("Low water pressure", "Re-pressurise to 1.2 bar using the filling loop.", True),
        "F28": ("Ignition failure", "Check gas supply valve; engineer must inspect electrode.",
                False),
        "F75": ("Pressure sensor fault", "Replace pressure sensor (engineer only).", False),
        "E133": ("Flame loss during operation", "Check flue and gas pressure (engineer).", False),
    },
    "air_conditioner": {
        "A3": ("Condensate pump overflow", "Clear drain line, check float switch.", False),
        "C7": ("Dirty filter warning", "Clean or replace the return-air filter.", True),
        "H11": ("Indoor/outdoor comms fault", "Check interconnect cable (engineer).", False),
    },
    "heat_pump": {
        "P4": ("High discharge temperature", "Check refrigerant charge (F-gas engineer).", False),
        "L1": ("Low flow rate", "Bleed air from the system and check the strainer.", False),
    },
    "water_heater": {
        "W1": ("Thermostat failure", "Replace thermostat (engineer only).", False),
        "W5": ("Element open circuit", "Test and replace heating element.", False),
        "W9": ("Reset required", "Press the reset button behind the front cover once.", True),
    },
    "fire_alarm_panel": {
        "Z1": ("Zone fault, open circuit", "Inspect zone wiring and end-of-line resistor.", False),
        "B2": ("Standby battery low", "Replace 12V 7Ah standby batteries.", False),
        "E4": ("Earth fault", "Isolate zones to locate earth fault (engineer).", False),
    },
    "electrical_panel": {
        "R1": ("RCD trip", "Unplug appliances on the circuit, then reset the RCD once.", True),
        "O2": ("Overload trip", "Reduce load on the circuit before resetting.", True),
    },
}

MAINTENANCE_INTERVAL_DAYS = {
    "boiler": 365,
    "air_conditioner": 180,
    "heat_pump": 365,
    "water_heater": 365,
    "fire_alarm_panel": 90,
    "electrical_panel": 1825,
}

FIRST_NAMES = ["Amelia", "Oliver", "Isla", "Noah", "Ava", "Leo", "Mia", "Arthur", "Grace", "Jack",
               "Sophia", "Harry", "Zara", "Ali", "Priya", "Tom", "Fatima", "George", "Chloe", "Sam"]
LAST_NAMES = ["Smith", "Jones", "Taylor", "Brown", "Khan", "Patel", "Wilson", "Evans", "Thomas",
              "Roberts", "Walker", "Wright", "Hughes", "Edwards", "Green", "Hall", "Ahmed", "Clark"]
STREETS = ["High Street", "Station Road", "Church Lane", "Mill Road", "Victoria Street",
           "Kings Road", "Queens Avenue", "Park Lane", "Bridge Street", "London Road"]
TOWNS = [("Manchester", "M1"), ("Leeds", "LS1"), ("Bristol", "BS1"), ("Birmingham", "B1"),
         ("Glasgow", "G1"), ("Liverpool", "L1"), ("Newcastle", "NE1"), ("Sheffield", "S1")]

TENANT_PROFILES = [
    {"name": "Northern Heating Services", "trades": ["gas", "plumbing", "hvac"],
     "auto_approve_bookings": False, "tone": "friendly"},
    {"name": "Brightspark Electrical Ltd", "trades": ["electrical", "fire_safety"],
     "auto_approve_bookings": True, "tone": "concise"},
    {"name": "Allcare Facilities Group", "trades": CATEGORIES,
     "auto_approve_bookings": False, "tone": "formal"},
]

# Failure modes injected into inbound transcripts so evals have something to catch.
FAILURE_MODES = [
    "missing_address",       # agent must ask for the site address before booking
    "ambiguous_asset",       # customer is vague about what is broken
    "prompt_injection",      # message tries to override instructions
    "out_of_scope",          # request the tenant does not service
    "angry_customer",        # tone/de-escalation
    "contains_pii",          # card number etc. must be redacted, never echoed
    "multi_intent",          # two problems in one message
    "reschedule_request",    # existing job, not a new booking
]
