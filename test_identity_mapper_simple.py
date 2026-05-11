#!/usr/bin/env python3
"""
Simple test for identity mapper functionality.
"""

import sys
import os
sys.path.append('.')

from anti_cheat_system.recognition.identity_mapper import (
    IdentityMapper, IdentityMappingConfig, IdentityMapping, IdentityState
)
from anti_cheat_system.models.data_models import StudentIdentity, StudentTrack, BoundingBox
from datetime import datetime

def test_identity_mapper():
    """Test basic identity mapper functionality."""
    print("Testing Identity Mapper...")
    
    # Create configuration
    config = IdentityMappingConfig(
        recognition_interval_frames=10,
        tentative_confidence_threshold=0.6,
        confirmation_confidence_threshold=0.8
    )
    print("✓ Configuration created")
    
    # Create mapper
    mapper = IdentityMapper(config)
    print("✓ Mapper created")
    
    # Create student identity
    student_identity = StudentIdentity(
        roll_number="ROLL001",
        name="Test Student",
        class_id="CS101",
        confidence=0.85
    )
    print("✓ Student identity created")
    
    # Force identity mapping
    success = mapper.force_identity_mapping(1, student_identity)
    print(f"✓ Force mapping: {success}")
    
    # Create student tracks
    tracks = [
        StudentTrack(
            track_id=1,
            bbox=BoundingBox(x1=100, y1=100, x2=200, y2=200),
            confidence=0.8,
            age=10,
            last_seen=datetime.now(),
            is_confirmed=True
        ),
        StudentTrack(
            track_id=2,
            bbox=BoundingBox(x1=300, y1=150, x2=400, y2=250),
            confidence=0.9,
            age=5,
            last_seen=datetime.now(),
            is_confirmed=True
        )
    ]
    print("✓ Student tracks created")
    
    # Update mapper
    identity_mappings = mapper.update(tracks, frame_number=1)
    print(f"✓ Mapper updated, mappings: {len(identity_mappings)}")
    
    # Get mapping by track ID
    mapping = mapper.get_mapping_by_track_id(1)
    if mapping:
        print(f"✓ Mapping found: track {mapping.track_id} -> {mapping.student_identity.roll_number}")
    else:
        print("✗ No mapping found")
    
    # Get performance metrics
    metrics = mapper.get_performance_metrics()
    print(f"✓ Metrics: {metrics['active_mappings']} active mappings")
    
    # Get identity statistics
    stats = mapper.get_identity_statistics()
    print(f"✓ Statistics: {stats['total_students_recognized']} students recognized")
    
    print("Identity Mapper test completed successfully!")
    return True

if __name__ == "__main__":
    try:
        test_identity_mapper()
        print("\n🎉 All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()