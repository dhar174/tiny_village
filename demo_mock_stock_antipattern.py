#!/usr/bin/env python3
"""
Demo script showing the difference between problematic type() mocks and proper mocks.
This demonstrates why using type() to create MockStock objects can mask real issues.
"""

from unittest.mock import Mock
import traceback


def demonstrate_type_mock_problems():
    """Show how type() mocks can mask real issues."""
    print("🚨 ANTIPATTERN: Using type() to create MockStock")
    print("=" * 60)
    
    # BAD: Using type() to create a fake MockStock
    MockStock = type("MockStock", (object,), {
        "name": "AAPL",
        "value": 150.0,
        "quantity": 10
    })
    
    fake_stock = MockStock()
    print(f"✅ Created fake stock: {fake_stock.name}")
    print(f"✅ Stock value: {fake_stock.value}")
    print(f"✅ Stock quantity: {fake_stock.quantity}")
    print("\n⚠️  This looks like it works, but it's masking problems:")
    
    # Show what happens when we try to call methods that should exist on Stock
    print("\n🔍 Testing real Stock interface methods:")
    methods_to_test = [
        ("get_value", []),
        ("set_value", [175.0]), 
        ("get_quantity", []),
        ("increase_quantity", [5]),
        ("decrease_quantity", [3]),
        ("to_dict", [])
    ]
    
    for method, args in methods_to_test:
        try:
            getattr(fake_stock, method)(*args)
            print(f"✅ {method}({', '.join(map(str, args))}) - OK")
        except AttributeError:
            print(f"❌ {method}({', '.join(map(str, args))}) - MISSING!")
        except Exception as e:
            print(f"⚠️  {method}({', '.join(map(str, args))}) - ERROR: {e}")
    
    print(f"\n❌ The type() mock is missing {len(methods_to_test)} critical methods!")
    print("❌ Tests using this mock would pass even if the real Stock class is broken!")


def demonstrate_proper_mock():
    """Show how proper mocks catch real issues."""
    print("\n\n✅ CORRECT APPROACH: Proper Mock with realistic interface")
    print("=" * 60)
    
    # GOOD: Using Mock with spec to match real Stock interface
    proper_mock = Mock(spec_set=[
        'name', 'value', 'quantity', 'stock_description', 'uuid',
        'get_name', 'set_name', 'get_value', 'set_value', 
        'get_quantity', 'set_quantity', 'increase_quantity', 
        'decrease_quantity', 'increase_value', 'decrease_value', 'to_dict'
    ])
    
    # Configure the mock with realistic behavior
    proper_mock.name = "AAPL"
    proper_mock.value = 150.0
    proper_mock.quantity = 10
    proper_mock.get_value.return_value = 150.0
    proper_mock.get_quantity.return_value = 10
    proper_mock.set_value.return_value = 175.0
    proper_mock.increase_quantity.return_value = 15
    proper_mock.to_dict.return_value = {"name": "AAPL", "value": 150.0, "quantity": 10}
    
    print(f"✅ Created proper mock: {proper_mock.name}")
    print(f"✅ Stock value: {proper_mock.value}")
    print(f"✅ Stock quantity: {proper_mock.quantity}")
    
    print("\n🔍 Testing real Stock interface methods:")
    methods_to_test = [
        ("get_value", []),
        ("set_value", [175.0]), 
        ("get_quantity", []),
        ("increase_quantity", [5]),
        ("to_dict", [])
    ]
    
    for method, args in methods_to_test:
        try:
            result = getattr(proper_mock, method)(*args)
            print(f"✅ {method}({', '.join(map(str, args))}) = {result}")
        except Exception as e:
            print(f"❌ {method}() - ERROR: {e}")
    
    print("\n✅ The proper mock has all required methods!")
    print("✅ Tests using this mock will fail if the real Stock interface changes!")
    
    # Show that it prevents calling non-existent methods
    print("\n🔍 Testing protection against non-existent methods:")
    try:
        proper_mock.non_existent_method()
        print("❌ This shouldn't happen!")
    except AttributeError:
        print("✅ Properly prevents calls to non-existent methods!")


def demonstrate_investment_validation():
    """Show how proper mocks enable proper validation."""
    print("\n\n🧪 INVESTMENT VALIDATION DEMO")
    print("=" * 60)
    
    def validate_stock(stock):
        """Simulate validation logic that would be in the real system."""
        required_attributes = ['name', 'value', 'quantity']
        required_methods = ['get_value', 'get_quantity', 'to_dict']
        
        validation_results = []
        
        # Check attributes
        for attr in required_attributes:
            if hasattr(stock, attr):
                validation_results.append(f"✅ Has {attr}")
            else:
                validation_results.append(f"❌ Missing {attr}")
        
        # Check methods
        for method in required_methods:
            if hasattr(stock, method):
                validation_results.append(f"✅ Has {method}()")
            else:
                validation_results.append(f"❌ Missing {method}()")
        
        # Check value constraints
        try:
            if hasattr(stock, 'get_value'):
                value = stock.get_value()
                if value > 0:
                    validation_results.append(f"✅ Value > 0: {value}")
                else:
                    validation_results.append(f"❌ Invalid value: {value}")
            elif hasattr(stock, 'value'):
                if stock.value > 0:
                    validation_results.append(f"✅ Value > 0: {stock.value}")
                else:
                    validation_results.append(f"❌ Invalid value: {stock.value}")
        except Exception as e:
            validation_results.append(f"❌ Value check failed: {e}")
        
        return validation_results
    
    # Test with type() mock
    print("📊 Validating type() mock:")
    MockStock = type("MockStock", (object,), {"name": "AAPL", "value": 150.0, "quantity": 10})
    type_mock = MockStock()
    
    type_results = validate_stock(type_mock)
    for result in type_results:
        print(f"  {result}")
    
    # Test with proper mock
    print("\n📊 Validating proper mock:")
    proper_mock = Mock(spec_set=['name', 'value', 'quantity', 'get_value', 'get_quantity', 'to_dict'])
    proper_mock.name = "AAPL"
    proper_mock.value = 150.0
    proper_mock.quantity = 10
    proper_mock.get_value.return_value = 150.0
    proper_mock.get_quantity.return_value = 10
    proper_mock.to_dict.return_value = {"name": "AAPL", "value": 150.0, "quantity": 10}
    
    proper_results = validate_stock(proper_mock)
    for result in proper_results:
        print(f"  {result}")
    
    # Count failures
    type_failures = sum(1 for r in type_results if "❌" in r)
    proper_failures = sum(1 for r in proper_results if "❌" in r)
    
    print(f"\n📈 Validation Summary:")
    print(f"  type() mock failures: {type_failures}")
    print(f"  Proper mock failures: {proper_failures}")
    
    if type_failures > proper_failures:
        print("✅ Proper mock catches more issues than type() mock!")
    else:
        print("⚠️  Both mocks have similar failure rates")


def main():
    """Run all demonstrations."""
    print("🎯 MockStock Testing Antipattern Demonstration")
    print("=" * 80)
    print("This demo shows why using type() to create MockStock objects")
    print("can mask real issues with stock creation or management.")
    print("=" * 80)
    
    demonstrate_type_mock_problems()
    demonstrate_proper_mock()
    demonstrate_investment_validation()
    
    print("\n\n🎯 CONCLUSION")
    print("=" * 40)
    print("✅ Use unittest.mock.Mock with spec_set for realistic mocks")
    print("❌ Avoid type() for creating mock objects")
    print("✅ Proper mocks will fail if the real system changes")
    print("❌ type() mocks mask interface changes and real bugs")
    print("✅ Proper mocks enable better validation and testing")


if __name__ == "__main__":
    main()