# training_infra/config/validation.py
"""
Configuration Validation System

Provides comprehensive validation for configuration objects:
- Type checking
- Value range validation  
- Dependency validation
- Custom validation rules
- Detailed error reporting
"""

import re
from typing import Any, Callable, Dict, List, Optional, Type, Union
from dataclasses import fields, is_dataclass
from pathlib import Path

from .base import BaseConfig, ConfigValidationError

class ValidationRule:
    """
    A validation rule that can be applied to configuration fields.
    """
    
    def __init__(
        self,
        field_name: str,
        validator: Callable[[Any], bool],
        error_message: str,
        required: bool = True
    ):
        self.field_name = field_name
        self.validator = validator
        self.error_message = error_message
        self.required = required
    
    def validate(self, config: BaseConfig) -> Optional[str]:
        """
        Validate a configuration against this rule.
        
        Returns:
            None if valid, error message if invalid
        """
        try:
            value = getattr(config, self.field_name, None)
            
            # Check if field is required
            if value is None and self.required:
                return f"Required field '{self.field_name}' is missing"
            
            # Skip validation if field is optional and None
            if value is None and not self.required:
                return None
            
            # Apply validator
            if not self.validator(value):
                return f"Field '{self.field_name}': {self.error_message}"
            
            return None
            
        except Exception as e:
            return f"Error validating '{self.field_name}': {e}"

class ConfigValidator:
    """
    Configuration validator that applies multiple validation rules.
    """
    
    def __init__(self):
        self.rules: List[ValidationRule] = []
        self.custom_validators: List[Callable[[BaseConfig], Optional[str]]] = []
    
    def add_rule(self, rule: ValidationRule) -> 'ConfigValidator':
        """Add a validation rule."""
        self.rules.append(rule)
        return self
    
    def add_custom_validator(self, validator: Callable[[BaseConfig], Optional[str]]) -> 'ConfigValidator':
        """Add a custom validator function."""
        self.custom_validators.append(validator)
        return self
    
    def validate(self, config: BaseConfig) -> List[str]:
        """
        Validate a configuration object.
        
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        
        # Apply validation rules
        for rule in self.rules:
            error = rule.validate(config)
            if error:
                errors.append(error)
        
        # Apply custom validators
        for validator in self.custom_validators:
            try:
                error = validator(config)
                if error:
                    errors.append(error)
            except Exception as e:
                errors.append(f"Custom validator error: {e}")
        
        return errors
    
    def validate_strict(self, config: BaseConfig) -> bool:
        """
        Validate configuration and raise exception if invalid.
        
        Returns:
            True if valid
            
        Raises:
            ConfigValidationError: If validation fails
        """
        errors = self.validate(config)
        if errors:
            error_msg = "\n".join([f"  - {error}" for error in errors])
            raise ConfigValidationError(f"Configuration validation failed:\n{error_msg}")
        
        return True

# Common validation functions

def is_positive(value: Union[int, float]) -> bool:
    """Check if value is positive."""
    return isinstance(value, (int, float)) and value > 0

def is_non_negative(value: Union[int, float]) -> bool:
    """Check if value is non-negative."""
    return isinstance(value, (int, float)) and value >= 0

def is_in_range(min_val: float, max_val: float) -> Callable[[float], bool]:
    """Create a validator for value in range."""
    def validator(value: Union[int, float]) -> bool:
        return isinstance(value, (int, float)) and min_val <= value <= max_val
    return validator

def is_one_of(valid_values: List[Any]) -> Callable[[Any], bool]:
    """Create a validator for value in list."""
    def validator(value: Any) -> bool:
        return value in valid_values
    return validator

def is_valid_path(value: Union[str, Path]) -> bool:
    """Check if path is valid (doesn't need to exist)."""
    try:
        Path(value)
        return True
    except Exception:
        return False

def is_existing_path(value: Union[str, Path]) -> bool:
    """Check if path exists."""
    try:
        return Path(value).exists()
    except Exception:
        return False

def is_valid_email(value: str) -> bool:
    """Check if string is a valid email."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return isinstance(value, str) and bool(re.match(pattern, value))

def has_min_length(min_len: int) -> Callable[[str], bool]:
    """Create a validator for minimum string length."""
    def validator(value: str) -> bool:
        return isinstance(value, str) and len(value) >= min_len
    return validator

def matches_pattern(pattern: str) -> Callable[[str], bool]:
    """Create a validator for regex pattern matching."""
    compiled_pattern = re.compile(pattern)
    def validator(value: str) -> bool:
        return isinstance(value, str) and bool(compiled_pattern.match(value))
    return validator

# Specialized validators for training configurations

def is_valid_device(value: str) -> bool:
    """Check if device string is valid."""
    valid_devices = ["cpu", "cuda", "auto"]
    if value in valid_devices:
        return True
    
    # Check for specific CUDA devices (e.g., "cuda:0")
    cuda_pattern = r'^cuda:\d+$'
    return bool(re.match(cuda_pattern, value))

def is_valid_optimizer(value: str) -> bool:
    """Check if optimizer name is valid."""
    valid_optimizers = ["adam", "adamw", "sgd", "rmsprop", "adagrad"]
    return value.lower() in valid_optimizers

def is_valid_scheduler(value: str) -> bool:
    """Check if scheduler name is valid."""
    valid_schedulers = ["linear", "cosine", "exponential", "step", "plateau"]
    return value.lower() in valid_schedulers

def is_power_of_two(value: int) -> bool:
    """Check if value is a power of two."""
    return isinstance(value, int) and value > 0 and (value & (value - 1)) == 0

# Main validation function

def validate_config(config: BaseConfig, validator: Optional[ConfigValidator] = None) -> bool:
    """
    Validate a configuration object.
    
    Args:
        config: Configuration to validate
        validator: Custom validator (optional)
    
    Returns:
        True if valid
        
    Raises:
        ConfigValidationError: If validation fails
    """
    if validator is None:
        # Use basic validation
        config.validate()
        return True
    else:
        # Use custom validator
        return validator.validate_strict(config)

# Pre-built validators for common configurations

def create_training_validator() -> ConfigValidator:
    """Create a validator for training configurations."""
    validator = ConfigValidator()
    
    # Add common training validation rules
    validator.add_rule(ValidationRule(
        "batch_size", 
        is_positive,
        "must be positive integer"
    ))
    
    validator.add_rule(ValidationRule(
        "learning_rate",
        lambda x: is_positive(x) and x < 1.0,
        "must be positive and less than 1.0"
    ))
    
    validator.add_rule(ValidationRule(
        "epochs",
        is_positive,
        "must be positive integer"
    ))
    
    return validator

def create_model_validator() -> ConfigValidator:
    """Create a validator for model configurations."""
    validator = ConfigValidator()
    
    # Add common model validation rules
    validator.add_rule(ValidationRule(
        "hidden_size",
        lambda x: is_positive(x) and x % 64 == 0,
        "must be positive and divisible by 64"
    ))
    
    validator.add_rule(ValidationRule(
        "num_layers",
        is_positive,
        "must be positive integer"
    ))
    
    validator.add_rule(ValidationRule(
        "num_attention_heads",
        is_positive,
        "must be positive integer"
    ))
    
    return validator

# Test function

def test_validation_system():
    """Test the validation system."""
    print("🧪 Testing Configuration Validation System")
    
    from .base import ExampleConfig
    
    # Test basic validation
    config = ExampleConfig(name="test", value=42)
    
    try:
        validate_config(config)
        print("✅ Basic validation passed")
    except ConfigValidationError as e:
        print(f"❌ Basic validation failed: {e}")
        return False
    
    # Test custom validator
    validator = ConfigValidator()
    validator.add_rule(ValidationRule(
        "value",
        lambda x: x > 0,
        "must be positive"
    ))
    
    try:
        validator.validate_strict(config)
        print("✅ Custom validation passed")
    except ConfigValidationError as e:
        print(f"❌ Custom validation failed: {e}")
        return False
    
    # Test validation failure
    bad_config = ExampleConfig(name="test", value=-1)
    
    try:
        validator.validate_strict(bad_config)
        print("❌ Should have failed validation")
        return False
    except ConfigValidationError:
        print("✅ Validation correctly failed for invalid config")
    
    # Test common validators
    assert is_positive(5)
    assert not is_positive(-1)
    assert is_in_range(0, 10)(5)
    assert not is_in_range(0, 10)(15)
    assert is_valid_device("cuda:0")
    assert not is_valid_device("invalid")
    
    print("✅ Common validators working")
    
    print("🎉 Validation system working!")
    return True

if __name__ == "__main__":
    test_validation_system()