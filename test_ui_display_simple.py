"""
Unit tests for the UI Display module (simplified version)
Tests basic visualization components and alert display functionality
"""

import pytest
import numpy as np
from datetime import datetime

from anti_cheat_system.ui_display_simple import UIDisplay
from anti_cheat_system.models import (
    SystemConfig,
    SystemResult,
    AlertLevel
)


class TestUIDisplaySimple:
    """Test cases for simplified UIDisplay class"""
    
    def test_ui_display_init(self):
        """Test UIDisplay initialization"""
        config = SystemConfig()
        ui_display = UIDisplay(config)
        
        assert ui_display.config == config
        assert 'normal' in ui_display.colors
        assert 'amber' in ui_display.colors
        assert 'red' in ui_display.colors
        assert ui_display.colors['normal'] == (0, 255, 0)
        assert ui_display.colors['red'] == (0, 0, 255)
    
    def test_create_simple_display_none_frame(self):
        """Test simple display creation with None frame"""
        config = SystemConfig()
        ui_display = UIDisplay(config)
        
        system_result = SystemResult(
            object_score=0.1,
            gaze_score=0.2,
            posture_score=0.1,
            composite_score=0.13,
            alert_level=AlertLevel.NORMAL,
            timestamp=datetime.now()
        )
        
        result = ui_display.create_simple_display(None, system_result)
        
        assert result is not None
        assert result.shape == (480, 640, 3)
        assert result.dtype == np.uint8
    
    def test_create_simple_display_basic(self):
        """Test basic simple display creation"""
        config = SystemConfig()
        ui_display = UIDisplay(config)
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        system_result = SystemResult(
            object_score=0.1,
            gaze_score=0.2,
            posture_score=0.1,
            composite_score=0.13,
            alert_level=AlertLevel.NORMAL,
            timestamp=datetime.now()
        )
        
        result = ui_display.create_simple_display(frame, system_result)
        
        assert result is not None
        assert result.shape == frame.shape
        assert result.dtype == np.uint8
    
    def test_get_alert_color(self):
        """Test alert color mapping"""
        config = SystemConfig()
        ui_display = UIDisplay(config)
        
        # Test all alert levels
        normal_color = ui_display._get_alert_color(AlertLevel.NORMAL)
        amber_color = ui_display._get_alert_color(AlertLevel.AMBER)
        red_color = ui_display._get_alert_color(AlertLevel.RED)
        
        assert normal_color == ui_display.colors['normal']
        assert amber_color == ui_display.colors['amber']
        assert red_color == ui_display.colors['red']
    
    def test_create_simple_display_different_alerts(self):
        """Test simple display with different alert levels"""
        config = SystemConfig()
        ui_display = UIDisplay(config)
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Test with RED alert
        red_result = SystemResult(
            object_score=0.9,
            gaze_score=0.8,
            posture_score=0.7,
            composite_score=0.86,
            alert_level=AlertLevel.RED,
            timestamp=datetime.now()
        )
        
        red_display = ui_display.create_simple_display(frame, red_result)
        assert red_display is not None
        
        # Test with AMBER alert
        amber_result = SystemResult(
            object_score=0.7,
            gaze_score=0.6,
            posture_score=0.5,
            composite_score=0.63,
            alert_level=AlertLevel.AMBER,
            timestamp=datetime.now()
        )
        
        amber_display = ui_display.create_simple_display(frame, amber_result)
        assert amber_display is not None