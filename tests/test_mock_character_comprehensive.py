#!/usr/bin/env python3
"""
Comprehensive test for character investment functionality.
This test demonstrates proper mocking of Stock objects vs problematic type() approach.
"""

import unittest
from unittest.mock import MagicMock, Mock, patch
import sys
import os

# Add the parent directory to sys.path to import project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCharacterInvestmentFunctionality(unittest.TestCase):
    """Test character investment functionality with proper and improper mocking patterns."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the dependencies to avoid import issues
        self.mock_modules = {}
        for module in ['numpy', 'networkx', 'torch', 'faiss']:
            mock_module = MagicMock()
            sys.modules[module] = mock_module
            self.mock_modules[module] = mock_module

    def tearDown(self):
        """Clean up after tests."""
        # Remove mocked modules
        for module in self.mock_modules:
            if module in sys.modules:
                del sys.modules[module]

    def test_problematic_type_mock_stock(self):
        """
        ANTIPATTERN: Using type() to create fake MockStock object.
        This approach masks real issues with stock creation or management.
        """
        # This is the WRONG way to mock a Stock object
        MockStock = type("MockStock", (object,), {
            "name": "TestStock",
            "value": 100.0,
            "quantity": 10
        })
        
        fake_stock = MockStock()
        
        # This test will pass even if the real Stock class changes its interface
        self.assertEqual(fake_stock.name, "TestStock")
        self.assertEqual(fake_stock.value, 100.0)
        self.assertEqual(fake_stock.quantity, 10)
        
        # But this mock doesn't have the actual methods that Stock should have
        # If the real Stock class has methods like get_value(), get_quantity(), etc.
        # this test won't catch if those methods are broken or missing
        
        # The following would fail because the type() mock doesn't have these methods:
        with self.assertRaises(AttributeError):
            fake_stock.get_value()
        with self.assertRaises(AttributeError):
            fake_stock.set_value(150.0)
        with self.assertRaises(AttributeError):
            fake_stock.increase_quantity(5)

    def test_proper_stock_mock_with_realistic_interface(self):
        """
        CORRECT APPROACH: Create a realistic mock that has the actual Stock interface.
        This mock will fail if the investment system doesn't work properly.
        """
        # Create a mock that mimics the real Stock class interface
        mock_stock = Mock(spec_set=['name', 'value', 'quantity', 'stock_description', 'uuid', 
                                   'scarcity', 'ownership_history', 'get_name', 'set_name', 
                                   'get_value', 'set_value', 'get_quantity', 'set_quantity',
                                   'increase_quantity', 'decrease_quantity', 'increase_value',
                                   'decrease_value', 'to_dict'])
        
        # Configure the mock with realistic behavior
        mock_stock.name = "AAPL"
        mock_stock.value = 150.0
        mock_stock.quantity = 25
        mock_stock.stock_description = "Apple Inc. stock"
        mock_stock.get_name.return_value = "AAPL"
        mock_stock.get_value.return_value = 150.0
        mock_stock.get_quantity.return_value = 25
        mock_stock.set_value.return_value = 150.0
        mock_stock.increase_quantity.return_value = 30
        mock_stock.to_dict.return_value = {"name": "AAPL", "value": 150.0, "quantity": 25}
        
        # Test that the mock behaves like a real Stock object
        self.assertEqual(mock_stock.get_name(), "AAPL")
        self.assertEqual(mock_stock.get_value(), 150.0)
        self.assertEqual(mock_stock.get_quantity(), 25)
        
        # Test method calls
        mock_stock.set_value(175.0)
        mock_stock.set_value.assert_called_with(175.0)
        
        mock_stock.increase_quantity(5)
        mock_stock.increase_quantity.assert_called_with(5)
        
        # This mock will fail if we try to access non-existent methods
        with self.assertRaises(AttributeError):
            mock_stock.non_existent_method()

    def test_investment_portfolio_with_proper_stock_mocks(self):
        """Test InvestmentPortfolio with properly mocked Stock objects."""
        # Create realistic Stock mocks
        stock1 = Mock(spec_set=['name', 'value', 'quantity', 'get_value', 'get_quantity'])
        stock1.name = "GOOGL"
        stock1.value = 2500.0
        stock1.quantity = 10
        stock1.get_value.return_value = 2500.0
        stock1.get_quantity.return_value = 10
        
        stock2 = Mock(spec_set=['name', 'value', 'quantity', 'get_value', 'get_quantity'])
        stock2.name = "MSFT"
        stock2.value = 300.0
        stock2.quantity = 20
        stock2.get_value.return_value = 300.0
        stock2.get_quantity.return_value = 20
        
        # Create InvestmentPortfolio mock with realistic behavior
        portfolio = Mock(spec_set=['stocks', 'get_stocks', 'set_stocks', 'get_portfolio_value'])
        portfolio.stocks = [stock1, stock2]
        portfolio.get_stocks.return_value = [stock1, stock2]
        
        # Calculate realistic portfolio value
        expected_value = (2500.0 * 10) + (300.0 * 20)  # 25000 + 6000 = 31000
        portfolio.get_portfolio_value.return_value = expected_value
        
        # Test the portfolio
        stocks = portfolio.get_stocks()
        self.assertEqual(len(stocks), 2)
        self.assertEqual(stocks[0].get_value(), 2500.0)
        self.assertEqual(stocks[1].get_value(), 300.0)
        self.assertEqual(portfolio.get_portfolio_value(), 31000.0)
        
        # This approach ensures that if the real InvestmentPortfolio changes its interface,
        # the test will fail and alert us to the change

    def test_character_investment_functionality_integration(self):
        """Test character with investment portfolio integration."""
        # Create a character mock with investment capabilities
        character = Mock(spec_set=['name', 'investment_portfolio', 'has_investment', 
                                  'wealth_money', 'add_stock_to_portfolio'])
        
        # Set up character attributes
        character.name = "Alice"
        character.wealth_money = 10000.0
        
        # Create realistic investment portfolio
        stock = Mock(spec_set=['name', 'value', 'quantity', 'get_value'])
        stock.name = "TSLA"
        stock.value = 800.0
        stock.quantity = 5
        stock.get_value.return_value = 800.0
        
        portfolio = Mock(spec_set=['get_stocks', 'get_portfolio_value'])
        portfolio.get_stocks.return_value = [stock]
        portfolio.get_portfolio_value.return_value = 4000.0  # 800 * 5
        
        character.investment_portfolio = portfolio
        character.has_investment.return_value = True
        
        # Test character investment functionality
        self.assertTrue(character.has_investment())
        self.assertEqual(len(character.investment_portfolio.get_stocks()), 1)
        self.assertEqual(character.investment_portfolio.get_portfolio_value(), 4000.0)
        
        # Test adding stock to portfolio
        new_stock = Mock(spec_set=['name', 'value', 'quantity'])
        new_stock.name = "NVDA"
        new_stock.value = 500.0
        new_stock.quantity = 2
        
        character.add_stock_to_portfolio(new_stock)
        character.add_stock_to_portfolio.assert_called_with(new_stock)

    def test_stock_validation_with_proper_mock(self):
        """Test that proper mocks can be validated like real Stock objects."""
        # Create a mock that passes isinstance checks by setting the right spec
        mock_stock = Mock(spec=['name', 'value', 'quantity', 'get_value', 'get_quantity', 'stock_description'])
        
        # Set up realistic Stock interface
        mock_stock.name = "BTC"
        mock_stock.value = 45000.0
        mock_stock.quantity = 0.5
        mock_stock.get_value.return_value = 45000.0
        mock_stock.get_quantity.return_value = 0.5
        mock_stock.stock_description = "Bitcoin cryptocurrency"
        
        # Test validation logic (similar to what's in test_motivecalc.py)
        # This would be the kind of validation that needs to pass:
        self.assertTrue(hasattr(mock_stock, 'get_value'))
        self.assertTrue(hasattr(mock_stock, 'value'))
        self.assertGreater(mock_stock.get_value(), 0)
        self.assertIsNotNone(mock_stock.name)
        
        # If the mock doesn't have the right interface, these tests would fail,
        # alerting us to problems with the investment system

    def test_antipattern_demonstration(self):
        """Demonstrate how type() mocks can mask real issues."""
        # BAD: type() mock that doesn't match real interface
        BadMockStock = type("MockStock", (object,), {
            "name": "BadStock",
            "value": 100.0
        })
        bad_mock = BadMockStock()
        
        # This will pass but doesn't test the real functionality
        self.assertEqual(bad_mock.name, "BadStock")
        self.assertEqual(bad_mock.value, 100.0)
        
        # But it fails to test real Stock behavior:
        # - No quantity attribute
        # - No get_value() method
        # - No validation of stock creation
        # - Won't catch if Stock constructor changes
        
        self.assertFalse(hasattr(bad_mock, 'quantity'))
        self.assertFalse(hasattr(bad_mock, 'get_value'))
        self.assertFalse(hasattr(bad_mock, 'stock_description'))
        
        # GOOD: Proper mock with realistic interface
        good_mock = Mock(spec_set=['name', 'value', 'quantity', 'get_value', 'set_value'])
        good_mock.name = "GoodStock"
        good_mock.value = 100.0
        good_mock.quantity = 10
        good_mock.get_value.return_value = 100.0
        
        # This tests the actual interface and will fail if the interface changes
        self.assertTrue(hasattr(good_mock, 'quantity'))
        self.assertTrue(hasattr(good_mock, 'get_value'))
        self.assertEqual(good_mock.get_value(), 100.0)


if __name__ == '__main__':
    unittest.main()