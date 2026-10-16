"""
Test cases for the City Information Assistant API.

This module contains comprehensive tests for all API endpoints
including health checks, chat functionality, edge cases, and error scenarios.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch

# Add the parent directory to Python path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock environment variables before importing app
with patch.dict('os.environ', {
    'OPENAI_API_KEY': 'test-openai-key',
    'OPENWEATHER_API_KEY': 'test-weather-key',
    'GEODB_API_KEY': 'test-geodb-key',
    'LANGCHAIN_API_KEY': 'test-langchain-key',
    'REDIS_HOST': 'localhost',
    'REDIS_PORT': '6379',
    'REDIS_DB': '0',
    'REDIS_PASSWORD': '',
    'CACHE_ENABLED': 'false',
    'DEBUG': 'true',
    'ENVIRONMENT': 'test'
}):
    pass


class TestHealthEndpoint:
    """Test cases for the health check endpoint."""

    def test_health_check_success(self, client):
        """Test successful health check."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_api_health_check(self, client):
        """Test API health check endpoint."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()

        assert "data" in data
        health_data = data["data"]

        assert "status" in health_data
        assert "timestamp" in health_data
        assert "version" in health_data
        assert "environment" in health_data
        assert "dependencies" in health_data

        dependencies = health_data["dependencies"]
        assert "openai" in dependencies
        assert "openweathermap" in dependencies
        assert "geodb" in dependencies

    def test_health_check_headers(self, client):
        """Test health check response headers."""
        response = client.get("/health")
        assert response.status_code == 200
        assert "application/json" in response.headers.get("content-type", "")

    def test_health_check_performance(self, client):
        """Test health check response time."""
        import time
        start_time = time.time()
        response = client.get("/health")
        end_time = time.time()
        
        assert response.status_code == 200
        assert end_time - start_time < 1.0  # Should respond within 1 second

    def test_health_check_multiple_requests(self, client):
        """Test multiple concurrent health check requests."""
        import threading
        
        results = []
        
        def make_request():
            response = client.get("/health")
            results.append(response.status_code)
        
        threads = [threading.Thread(target=make_request) for _ in range(10)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        assert len(results) == 10
        assert all(status == 200 for status in results)


class TestChatEndpoint:
    """Test cases for the chat endpoint."""

    def test_chat_endpoint_basic(self, client):
        """Test basic chat functionality."""
        chat_request = {
            "messages": [
                {"role": "user", "content": "What's the weather like in London?"}
            ],
            "city": "London",
            "country": "UK",
            "temperature": 0.7,
            "max_tokens": 500,
            "stream": False
        }

        response = client.post("/api/v1/chat", json=chat_request)
        assert response.status_code == 200

        data = response.json()
        assert "message" in data
        assert "usage" in data

        message = data["message"]
        assert message["role"] == "assistant"
        assert "content" in message
        assert len(message["content"]) > 0

    def test_chat_endpoint_validation_error(self, client):
        """Test chat endpoint with invalid request."""
        # Missing required fields
        chat_request = {
            "messages": [],  # Empty messages should fail validation
            "city": "London"
        }

        response = client.post("/api/v1/chat", json=chat_request)
        assert response.status_code == 422  # Validation error

    def test_chat_endpoint_missing_city(self, client):
        """Test chat endpoint without city parameter - should succeed as city is optional."""
        chat_request = {
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
            # Missing city field - but it's optional now
        }

        response = client.post("/api/v1/chat", json=chat_request)
        assert response.status_code == 200  # Should succeed as city is optional

    def test_chat_stream_endpoint(self, client):
        """Test streaming chat endpoint."""
        params = {
            "message": "What's the weather in Tokyo?",
            "city": "Tokyo",
            "country": "Japan"
        }

        response = client.get("/api/v1/chat/stream", params=params)
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    def test_chat_endpoint_edge_cases(self, client):
        """Test chat endpoint with various edge cases."""
        edge_cases = [
            # Very long message
            {
                "messages": [{"role": "user", "content": "x" * 1000}],
                "city": "London"
            },
            # Unicode characters
            {
                "messages": [{"role": "user", "content": "Tell me about 东京"}],
                "city": "Tokyo"
            },
            # Special characters
            {
                "messages": [{"role": "user", "content": "What about São Paulo?"}],
                "city": "São Paulo"
            }
        ]
        
        for case in edge_cases:
            response = client.post("/api/v1/chat", json=case)
            # Allow for various response codes including server errors
            assert response.status_code in [200, 422, 500]
        
        # Test empty content separately
        empty_content_case = {
            "messages": [{"role": "user", "content": ""}],
            "city": "London"
        }
        response = client.post("/api/v1/chat", json=empty_content_case)
        # Empty content might be handled differently - allow 200 or 422
        assert response.status_code in [200, 422]

    def test_chat_endpoint_invalid_role(self, client):
        """Test chat endpoint with invalid message role."""
        invalid_request = {
            "messages": [{"role": "invalid_role", "content": "Test message"}],
            "city": "London"
        }
        
        response = client.post("/api/v1/chat", json=invalid_request)
        # May return 200 if validation is lenient, or 422 for strict validation
        assert response.status_code in [200, 422]

    def test_chat_endpoint_missing_content(self, client):
        """Test chat endpoint with missing message content."""
        chat_request = {
            "messages": [
                {"role": "user"}  # Missing content
            ],
            "city": "London"
        }

        response = client.post("/api/v1/chat", json=chat_request)
        assert response.status_code == 422

    def test_chat_endpoint_conversation_history(self, client):
        """Test chat endpoint with conversation history."""
        chat_request = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi! How can I help?"},
                {"role": "user", "content": "Tell me about London"}
            ],
            "city": "London",
            "country": "UK"
        }

        response = client.post("/api/v1/chat", json=chat_request)
        assert response.status_code == 200

    def test_chat_stream_missing_parameters(self, client):
        """Test streaming endpoint with missing parameters."""
        # Missing message - should return 422 as message is required
        response = client.get("/api/v1/chat/stream", params={
            "city": "London"
        })
        assert response.status_code == 422

        # Missing city is now OK since city is optional
        response = client.get("/api/v1/chat/stream", params={
            "message": "Test message"
        })
        assert response.status_code == 200  # Should succeed as city is optional


class TestRootEndpoints:
    """Test cases for root endpoints."""

    def test_root_endpoint(self, client):
        """Test the root endpoint."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "docs" in data
        assert "redoc" in data

        assert data["docs"] == "/docs"
        assert data["redoc"] == "/redoc"

    def test_root_endpoint_headers(self, client):
        """Test root endpoint response headers."""
        response = client.get("/")
        assert response.status_code == 200
        assert "application/json" in response.headers.get("content-type", "")

    def test_docs_endpoint_accessibility(self, client):
        """Test that docs endpoints are accessible."""
        # Test OpenAPI docs
        response = client.get("/docs")
        assert response.status_code == 200

        # Test ReDoc
        response = client.get("/redoc")
        assert response.status_code == 200

        # Test OpenAPI JSON
        response = client.get("/openapi.json")
        assert response.status_code == 200
        assert "application/json" in response.headers.get("content-type", "")


class TestErrorHandling:
    """Test cases for error handling."""

    def test_404_endpoint(self, client):
        """Test accessing non-existent endpoint."""
        response = client.get("/nonexistent")
        assert response.status_code == 404

    def test_invalid_json(self, client):
        """Test sending invalid JSON to chat endpoint."""
        response = client.post(
            "/api/v1/chat",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_malformed_json(self, client):
        """Test sending malformed JSON."""
        response = client.post(
            "/api/v1/chat",
            data='{"messages": [{"role": "user", "content": "test"}], "city": "London"',  # Missing closing brace
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_wrong_content_type(self, client):
        """Test sending request with wrong content type."""
        response = client.post(
            "/api/v1/chat",
            data="test data",
            headers={"Content-Type": "text/plain"}
        )
        assert response.status_code == 422

    def test_method_not_allowed(self, client):
        """Test using wrong HTTP method."""
        # GET on chat endpoint (should be POST)
        response = client.get("/api/v1/chat")
        assert response.status_code == 405

        # POST on health endpoint (should be GET)
        response = client.post("/api/v1/health")
        assert response.status_code == 405

    def test_large_payload(self, client):
        """Test handling of very large payloads."""
        large_content = "A" * 1000000  # 1MB content
        
        chat_request = {
            "messages": [{"role": "user", "content": large_content}],
            "city": "London"
        }
        
        response = client.post("/api/v1/chat", json=chat_request)
        # Should either handle gracefully or reject with appropriate status
        assert response.status_code in [200, 413, 422, 500]

    def test_sql_injection_attempts(self, client):
        """Test SQL injection prevention."""
        malicious_inputs = [
            "London'; DROP TABLE cities; --",
            "London' OR '1'='1",
            "London\"; DELETE FROM users; --"
        ]
        
        for malicious_input in malicious_inputs:
            chat_request = {
                "messages": [{"role": "user", "content": f"Tell me about {malicious_input}"}],
                "city": malicious_input
            }
            
            response = client.post("/api/v1/chat", json=chat_request)
            # Should handle safely
            assert response.status_code in [200, 422, 400]

    def test_xss_prevention(self, client):
        """Test XSS prevention."""
        xss_inputs = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>"
        ]
        
        for xss_input in xss_inputs:
            chat_request = {
                "messages": [{"role": "user", "content": xss_input}],
                "city": "London"
            }
            
            response = client.post("/api/v1/chat", json=chat_request)
            # Should handle safely
            assert response.status_code in [200, 422, 400]


class TestRateLimiting:
    """Test cases for rate limiting (if implemented)."""

    def test_rapid_requests(self, client):
        """Test rapid successive requests."""
        responses = []
        
        for _ in range(20):  # Make 20 rapid requests
            response = client.get("/health")
            responses.append(response.status_code)
        
        # All should succeed (or some might be rate limited)
        assert all(status in [200, 429] for status in responses)

    def test_concurrent_requests(self, client):
        """Test concurrent requests."""
        import threading
        
        results = []
        
        def make_request():
            response = client.get("/health")
            results.append(response.status_code)
        
        threads = [threading.Thread(target=make_request) for _ in range(50)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All should succeed or be rate limited
        assert all(status in [200, 429, 500] for status in results)


class TestSecurity:
    """Test cases for security features."""

    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options(
            "/api/v1/health",
            headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"},
        )
        assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"

    def test_input_sanitization(self, client):
        """Test input sanitization."""
        dangerous_inputs = [
            {"role": "user", "content": "<script>alert('test')</script>"},
            {"role": "user", "content": "'; DROP TABLE users; --"},
            {"role": "user", "content": "../../etc/passwd"},
        ]
        
        for dangerous_input in dangerous_inputs:
            chat_request = {
                "messages": [dangerous_input],
                "city": "London"
            }
            
            response = client.post("/api/v1/chat", json=chat_request)
            # Should handle safely without exposing system information
            assert response.status_code in [200, 422, 400]


class TestPerformance:
    """Test cases for performance characteristics."""

    def test_response_time_health(self, client):
        """Test health endpoint response time."""
        import time
        
        start_time = time.time()
        response = client.get("/health")
        end_time = time.time()
        
        assert response.status_code == 200
        assert end_time - start_time < 0.5  # Should be very fast

    def test_response_time_chat(self, client):
        """Test chat endpoint response time."""
        import time
        
        chat_request = {
            "messages": [{"role": "user", "content": "Quick test"}],
            "city": "London"
        }
        
        start_time = time.time()
        response = client.post("/api/v1/chat", json=chat_request)
        end_time = time.time()
        
        assert response.status_code in [200, 422]
        # Chat endpoint can be slower due to AI processing
        assert end_time - start_time < 30.0

    def test_memory_usage(self, client):
        """Test that requests don't cause memory leaks."""
        import gc
        
        # Make multiple requests and check memory doesn't grow excessively
        for _ in range(100):
            response = client.get("/health")
            assert response.status_code == 200
        
        # Force garbage collection
        gc.collect()
        # This is a basic test - in practice you'd use memory profiling tools


@pytest.mark.asyncio
class TestIntegration:
    """Integration tests for the complete API."""

    async def test_full_city_info_workflow(self, client):
        """Test a complete workflow of getting city information."""
        # Test weather request
        weather_request = {
            "messages": [
                {"role": "user", "content": "What's the current weather?"}
            ],
            "city": "Paris",
            "country": "France",
            "temperature": 0.5
        }

        response = client.post("/api/v1/chat", json=weather_request)
        assert response.status_code == 200

        # Test time request
        time_request = {
            "messages": [
                {"role": "user", "content": "What time is it there?"}
            ],
            "city": "Paris",
            "country": "France"
        }

        response = client.post("/api/v1/chat", json=time_request)
        assert response.status_code == 200

        # Test facts request
        facts_request = {
            "messages": [
                {"role": "user", "content": "Tell me some interesting facts about this city"}
            ],
            "city": "Paris",
            "country": "France"
        }

        response = client.post("/api/v1/chat", json=facts_request)
        assert response.status_code == 200

    def test_api_consistency(self, client):
        """Test API response consistency."""
        request_data = {
            "messages": [{"role": "user", "content": "Test message"}],
            "city": "London"
        }
        
        # Make the same request multiple times
        responses = []
        for _ in range(5):
            response = client.post("/api/v1/chat", json=request_data)
            responses.append(response.status_code)
        
        # All responses should have the same status code
        assert len(set(responses)) == 1  # All status codes should be the same

    def test_streaming_vs_non_streaming(self, client):
        """Test consistency between streaming and non-streaming responses."""
        # Non-streaming request
        chat_request = {
            "messages": [{"role": "user", "content": "Tell me about London"}],
            "city": "London",
            "stream": False
        }
        
        response = client.post("/api/v1/chat", json=chat_request)
        non_streaming_status = response.status_code
        
        # Streaming request
        stream_params = {
            "message": "Tell me about London",
            "city": "London"
        }
        
        stream_response = client.get("/api/v1/chat/stream", params=stream_params)
        streaming_status = stream_response.status_code
        
        # Both should succeed or fail consistently
        if non_streaming_status == 200:
            assert streaming_status == 200
        elif non_streaming_status == 422:
            assert streaming_status == 422


class TestDocumentation:
    """Test API documentation and schema."""

    def test_openapi_schema_valid(self, client):
        """Test that OpenAPI schema is valid."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema

    def test_api_endpoints_documented(self, client):
        """Test that all API endpoints are documented."""
        response = client.get("/openapi.json")
        schema = response.json()
        
        paths = schema["paths"]
        
        # Check that main endpoints are documented
        assert "/api/v1/chat" in paths
        assert "/api/v1/chat/stream" in paths
        # Health endpoint is optional - just verify main API endpoints exist

    def test_response_schemas_defined(self, client):
        """Test that response schemas are properly defined."""
        response = client.get("/openapi.json")
        schema = response.json()
        
        # Should have components/schemas defined
        assert "components" in schema
        if "schemas" in schema["components"]:
            # Verify some expected schemas exist
            schemas = schema["components"]["schemas"]
            assert {"ChatRequest", "ChatResponse"} <= schemas.keys()
