# training_infra/config/base.py - FIXED VERSION
"""
Base Configuration System

Provides the foundation for all configuration classes with:
- Dataclass-based structure
- YAML/JSON serialization with nested object support
- Validation support
- Type checking
- Default value management
"""

import os
import json
import yaml
from dataclasses import dataclass, asdict, field, is_dataclass
from typing import Any, Dict, Optional, Union, Type
from pathlib import Path
import warnings

class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass

@dataclass
class BaseConfig:
    """
    Base configuration class that all other configs inherit from.
    
    Provides common functionality:
    - Serialization to/from YAML and JSON with nested object support
    - Validation
    - Dictionary conversion
    - Environment variable support
    """
    
    # Basic metadata
    config_name: str = "base_config"
    config_version: str = "0.1.0"
    
    def __post_init__(self):
        """Called after initialization to perform validation."""
        self.validate()
    
    def validate(self) -> bool:
        """
        Validate the configuration.
        
        Returns:
            bool: True if valid
            
        Raises:
            ConfigValidationError: If validation fails
        """
        try:
            # Basic validation - subclasses can override
            self._validate_types()
            return True
        except Exception as e:
            raise ConfigValidationError(f"Configuration validation failed: {e}")
    
    def _validate_types(self):
        """Validate that all fields have correct types."""
        # This is a placeholder - real validation in Phase 1.2
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary with nested object support."""
        def convert_value(value):
            if is_dataclass(value) and hasattr(value, 'to_dict'):
                return value.to_dict()
            elif is_dataclass(value):
                return asdict(value)
            elif isinstance(value, (list, tuple)):
                return [convert_value(item) for item in value]
            elif isinstance(value, dict):
                return {k: convert_value(v) for k, v in value.items()}
            else:
                return value
        
        result = {}
        for key, value in asdict(self).items():
            result[key] = convert_value(value)
        
        return result
    
    def to_yaml(self) -> str:
        """Convert configuration to YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)
    
    def to_json(self) -> str:
        """Convert configuration to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def save_yaml(self, filepath: Union[str, Path]):
        """Save configuration to YAML file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_yaml())
        
        print(f"💾 Configuration saved to {filepath}")
    
    def save_json(self, filepath: Union[str, Path]):
        """Save configuration to JSON file.""" 
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
        
        print(f"💾 Configuration saved to {filepath}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BaseConfig':
        """Create configuration from dictionary with nested object reconstruction."""
        # Get the field types from the dataclass
        field_types = {}
        if hasattr(cls, '__dataclass_fields__'):
            for field_name, field_info in cls.__dataclass_fields__.items():
                field_types[field_name] = field_info.type
        
        # Process the config dict to reconstruct nested objects
        processed_dict = {}
        for key, value in config_dict.items():
            if key in field_types:
                field_type = field_types[key]
                
                # Check if this field should be a dataclass
                if (isinstance(value, dict) and 
                    hasattr(field_type, '__dataclass_fields__')):
                    # Reconstruct the nested object
                    processed_dict[key] = field_type.from_dict(value)
                else:
                    processed_dict[key] = value
            else:
                processed_dict[key] = value
        
        try:
            return cls(**processed_dict)
        except TypeError as e:
            raise ConfigValidationError(f"Invalid configuration dictionary: {e}")
    
    @classmethod
    def from_yaml(cls, filepath: Union[str, Path]) -> 'BaseConfig':
        """Load configuration from YAML file."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        print(f"📂 Configuration loaded from {filepath}")
        return cls.from_dict(config_dict)
    
    @classmethod  
    def from_json(cls, filepath: Union[str, Path]) -> 'BaseConfig':
        """Load configuration from JSON file."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        print(f"📂 Configuration loaded from {filepath}")
        return cls.from_dict(config_dict)
    
    def update_from_dict(self, updates: Dict[str, Any]) -> 'BaseConfig':
        """Update configuration with values from dictionary."""
        current_dict = self.to_dict()
        current_dict.update(updates)
        return self.__class__.from_dict(current_dict)
    
    def update_from_env(self, prefix: str = "") -> 'BaseConfig':
        """
        Update configuration from environment variables.
        
        Args:
            prefix: Environment variable prefix (e.g., "TRAINING_")
        
        Example:
            TRAINING_BATCH_SIZE=64 -> batch_size = 64
        """
        updates = {}
        
        for key, value in os.environ.items():
            if prefix and not key.startswith(prefix):
                continue
            
            # Remove prefix and convert to lowercase
            config_key = key[len(prefix):].lower() if prefix else key.lower()
            
            # Try to convert to appropriate type
            try:
                # Try int first
                if value.isdigit():
                    updates[config_key] = int(value)
                # Try float
                elif '.' in value and value.replace('.', '').isdigit():
                    updates[config_key] = float(value)
                # Try boolean
                elif value.lower() in ('true', 'false'):
                    updates[config_key] = value.lower() == 'true'
                # Keep as string
                else:
                    updates[config_key] = value
            except:
                updates[config_key] = value
        
        if updates:
            print(f"🔄 Updated config from environment: {list(updates.keys())}")
            return self.update_from_dict(updates)
        
        return self
    
    def merge_with(self, other: 'BaseConfig') -> 'BaseConfig':
        """Merge this configuration with another configuration."""
        if not isinstance(other, BaseConfig):
            raise TypeError("Can only merge with another BaseConfig instance")
        
        current_dict = self.to_dict()
        other_dict = other.to_dict()
        
        # Merge dictionaries (other takes precedence)
        merged_dict = {**current_dict, **other_dict}
        
        return self.__class__.from_dict(merged_dict)
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"{self.__class__.__name__}({self.config_name})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"{self.__class__.__name__}(config_name='{self.config_name}', version='{self.config_version}')"

# Utility functions for working with configs

def save_config(config: BaseConfig, filepath: Union[str, Path], format: str = "yaml"):
    """
    Save any configuration to file.
    
    Args:
        config: Configuration instance to save
        filepath: Path to save file
        format: File format ("yaml" or "json")
    """
    if format.lower() == "yaml":
        config.save_yaml(filepath)
    elif format.lower() == "json":
        config.save_json(filepath)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'yaml' or 'json'.")

def load_config(config_class: Type[BaseConfig], filepath: Union[str, Path]) -> BaseConfig:
    """
    Load configuration from file.
    
    Args:
        config_class: Configuration class to instantiate
        filepath: Path to configuration file
    
    Returns:
        Configuration instance
    """
    filepath = Path(filepath)
    
    if filepath.suffix.lower() in ['.yaml', '.yml']:
        return config_class.from_yaml(filepath)
    elif filepath.suffix.lower() == '.json':
        return config_class.from_json(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")

# Example configuration for testing
@dataclass
class ExampleConfig(BaseConfig):
    """Example configuration class for testing."""
    
    config_name: str = "example_config"
    
    # Example fields
    name: str = "example"
    value: int = 42
    enabled: bool = True
    items: list = field(default_factory=list)
    
    def _validate_types(self):
        """Custom validation for example config."""
        if self.value < 0:
            raise ValueError("value must be non-negative")
        
        if not isinstance(self.name, str):
            raise TypeError("name must be a string")

# Test function
def test_base_config():
    """Test the base configuration system."""
    print("🧪 Testing Base Configuration System")
    
    # Test basic config
    config = BaseConfig()
    print(f"✅ Basic config created: {config}")
    
    # Test example config
    example = ExampleConfig(name="test", value=100)
    print(f"✅ Example config created: {example}")
    
    # Test serialization
    yaml_str = example.to_yaml()
    print(f"✅ YAML serialization: {len(yaml_str)} chars")
    
    json_str = example.to_json()
    print(f"✅ JSON serialization: {len(json_str)} chars")
    
    # Test dictionary conversion
    config_dict = example.to_dict()
    print(f"✅ Dictionary conversion: {len(config_dict)} keys")
    
    # Test from_dict
    restored = ExampleConfig.from_dict(config_dict)
    print(f"✅ Restored from dict: {restored.name}")
    
    print(f"🎉 Base configuration system working!")
    
    return True

if __name__ == "__main__":
    test_base_config()