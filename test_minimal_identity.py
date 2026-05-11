#!/usr/bin/env python3
"""
Minimal test for identity mapper.
"""

import sys
sys.path.append('.')

# Test minimal version
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional
from datetime import datetime

from anti_cheat_system.models.data_models import StudentIdentity, StudentTrack

class IdentityState(Enum):
    CONFIRMED = "confirmed"
    TENTATIVE = "tentative"

@dataclass
class SimpleConfig:
    recognition_threshold: float = 0.6

class SimpleIdentityMapper:
    def __init__(self, config: SimpleConfig):
        self.config = config
        self.mappings: Dict[int, StudentIdentity] = {}
    
    def add_mapping(self, track_id: int, identity: StudentIdentity):
        self.mappings[track_id] = identity
    
    def get_mapping(self, track_id: int) -> Optional[StudentIdentity]:
        return self.mappings.get(track_id)

def test_simple_mapper():
    print("Testing simple identity mapper...")
    
    config = SimpleConfig()
    mapper = SimpleIdentityMapper(config)
    
    identity = StudentIdentity(
        roll_number="ROLL001",
        name="Test Student", 
        class_id="CS101",
        confidence=0.85
    )
    
    mapper.add_mapping(1, identity)
    retrieved = mapper.get_mapping(1)
    
    if retrieved and retrieved.roll_number == "ROLL001":
        print("✓ Simple mapper works!")
        return True
    else:
        print("✗ Simple mapper failed")
        return False

if __name__ == "__main__":
    test_simple_mapper()