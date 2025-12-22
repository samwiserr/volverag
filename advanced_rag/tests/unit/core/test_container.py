"""
Unit tests for dependency injection container.
"""
import pytest
from src.core.container import ServiceContainer, get_container, reset_container


@pytest.mark.unit
class TestServiceContainer:
    """Test ServiceContainer class."""
    
    def test_register_and_get_singleton(self):
        """Test register and get singleton service."""
        container = ServiceContainer()
        
        class TestService:
            def __init__(self):
                self.value = "test"
        
        service = TestService()
        container.register(TestService, service)
        
        retrieved = container.get(TestService)
        assert retrieved is service  # Same instance
        assert retrieved.value == "test"
    
    def test_register_factory(self):
        """Test register factory function."""
        container = ServiceContainer()
        
        class TestService:
            def __init__(self):
                self.value = "test"
        
        def factory():
            return TestService()
        
        container.register_factory(TestService, factory)
        instance1 = container.get(TestService)
        instance2 = container.get(TestService)
        
        # Factory should create new instances, but container caches as singleton
        assert isinstance(instance1, TestService)
        assert isinstance(instance2, TestService)
    
    def test_get_raises_keyerror_when_not_registered(self):
        """Test get raises KeyError when service not registered."""
        container = ServiceContainer()
        
        class UnregisteredService:
            pass
        
        with pytest.raises(KeyError):
            container.get(UnregisteredService)
    
    def test_get_or_none_returns_none_when_not_registered(self):
        """Test get_or_none returns None when service not registered."""
        container = ServiceContainer()
        
        class UnregisteredService:
            pass
        
        result = container.get_or_none(UnregisteredService)
        assert result is None
    
    def test_get_or_none_returns_service_when_registered(self):
        """Test get_or_none returns service when registered."""
        container = ServiceContainer()
        
        class TestService:
            pass
        
        service = TestService()
        container.register(TestService, service)
        
        result = container.get_or_none(TestService)
        assert result is service
    
    def test_clear_removes_all_services(self):
        """Test clear removes all registered services."""
        container = ServiceContainer()
        
        class TestService:
            pass
        
        service = TestService()
        container.register(TestService, service)
        container.clear()
        
        with pytest.raises(KeyError):
            container.get(TestService)
    
    def test_is_registered_returns_true_when_registered(self):
        """Test is_registered returns True for registered service."""
        container = ServiceContainer()
        
        class TestService:
            pass
        
        service = TestService()
        container.register(TestService, service)
        assert container.is_registered(TestService) is True
    
    def test_is_registered_returns_false_when_not_registered(self):
        """Test is_registered returns False for unregistered service."""
        container = ServiceContainer()
        
        class UnregisteredService:
            pass
        
        assert container.is_registered(UnregisteredService) is False
    
    def test_is_registered_returns_true_for_factory(self):
        """Test is_registered returns True for factory-registered service."""
        container = ServiceContainer()
        
        class TestService:
            pass
        
        def factory():
            return TestService()
        
        container.register_factory(TestService, factory)
        assert container.is_registered(TestService) is True


@pytest.mark.unit
class TestContainerSingleton:
    """Test global container singleton functions."""
    
    def test_get_container_returns_singleton(self):
        """Test get_container returns same instance."""
        reset_container()
        container1 = get_container()
        container2 = get_container()
        assert container1 is container2
    
    def test_reset_container_creates_new_instance(self):
        """Test reset_container creates new instance."""
        container1 = get_container()
        reset_container()
        container2 = get_container()
        assert container1 is not container2
    
    def test_reset_container_clears_services(self):
        """Test reset_container clears services from previous instance."""
        container = get_container()
        
        class TestService:
            pass
        
        service = TestService()
        container.register(TestService, service)
        
        reset_container()
        new_container = get_container()
        
        with pytest.raises(KeyError):
            new_container.get(TestService)

