"""
CyberGuard AI - Test Configuration
Pytest fixtures and configuration
"""

import pytest
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def api_client():
    """Create test client for API"""
    from fastapi.testclient import TestClient
    from app.main import app

    return TestClient(app)


@pytest.fixture
def sample_attack():
    """Sample attack data for testing"""
    return {
        "id": "TEST-001",
        "source": {
            "country": "CN",
            "name": "China",
            "lat": 35.86,
            "lng": 104.19,
            "ip": "185.220.101.1",
        },
        "target": {
            "country": "TR",
            "name": "Turkey",
            "lat": 38.96,
            "lng": 35.24,
            "ip": "192.168.1.100",
        },
        "attack_type": "DDoS",
        "severity": "high",
        "timestamp": "2026-01-13T10:00:00",
        "blocked": True,
    }


@pytest.fixture
def sample_log_entry():
    """Sample log entry for testing"""
    return {
        "timestamp": "2026-01-13T10:00:00",
        "level": "WARNING",
        "source": "ids",
        "message": "Suspicious activity detected",
        "details": {"source_ip": "185.220.101.1", "target_port": 22, "protocol": "TCP"},
    }


@pytest.fixture
def sample_prediction():
    """Sample ML prediction for testing"""
    return {
        "is_threat": True,
        "confidence": 0.92,
        "severity": "high",
        "suggested_action": "block",
        "model_version": "1.0",
    }


# Test markers
def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "ml: marks tests that require ML models")
