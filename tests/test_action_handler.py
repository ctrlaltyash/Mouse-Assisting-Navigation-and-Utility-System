"""
Unit tests for Action Handler module.
"""

import unittest
from unittest.mock import patch, MagicMock
from actions.action_handler import ActionHandler


class TestActionHandler(unittest.TestCase):
    """Test action handler execution and safety."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.handler = ActionHandler()
    
    @patch('pyautogui.moveTo')
    def test_move_mouse(self, mock_move):
        """Test mouse movement."""
        result = self.handler.move_mouse(100, 200)
        
        self.assertTrue(result)
        mock_move.assert_called_once()
    
    @patch('pyautogui.click')
    def test_left_click(self, mock_click):
        """Test left mouse click."""
        result = self.handler.click_left()
        
        self.assertTrue(result)
        mock_click.assert_called_once_with(button='left')
        self.assertEqual(self.handler.action_count, 1)
    
    @patch('pyautogui.click')
    def test_right_click(self, mock_click):
        """Test right mouse click."""
        result = self.handler.click_right()
        
        self.assertTrue(result)
        mock_click.assert_called_once_with(button='right')
        self.assertEqual(self.handler.action_count, 1)
    
    @patch('pyautogui.mouseDown')
    @patch('pyautogui.mouseUp')
    def test_drag_sequence(self, mock_up, mock_down):
        """Test drag (mouse down/up) sequence."""
        down_result = self.handler.mouse_down()
        up_result = self.handler.mouse_up()
        
        self.assertTrue(down_result)
        self.assertTrue(up_result)
        mock_down.assert_called_once()
        mock_up.assert_called_once()
        self.assertEqual(self.handler.action_count, 1)  # Up counts as action
    
    @patch('pyautogui.scroll')
    def test_scroll(self, mock_scroll):
        """Test scroll action."""
        result = self.handler.scroll('up', 3)
        
        self.assertTrue(result)
        mock_scroll.assert_called_once()
        self.assertEqual(self.handler.action_count, 1)
    
    @patch('pyautogui.press')
    def test_key_press(self, mock_press):
        """Test keyboard key press."""
        result = self.handler.press_key('enter')
        
        self.assertTrue(result)
        mock_press.assert_called_once_with('enter')
        self.assertEqual(self.handler.action_count, 1)
    
    @patch('pyautogui.typewrite')
    def test_type_text(self, mock_type):
        """Test text typing."""
        result = self.handler.type_text("Hello")
        
        self.assertTrue(result)
        mock_type.assert_called_once()
        self.assertEqual(self.handler.action_count, 1)
    
    @patch('webbrowser.open')
    def test_open_browser(self, mock_browser):
        """Test browser open."""
        result = self.handler.open_browser("https://example.com")
        
        self.assertTrue(result)
        self.assertEqual(self.handler.action_count, 1)
    
    @patch('pyautogui.size')
    def test_get_screen_size(self, mock_size):
        """Test get screen size."""
        mock_size.return_value = (1920, 1080)
        
        size = self.handler.get_screen_size()
        
        self.assertEqual(size, (1920, 1080))
    
    def test_action_counter(self):
        """Test action counter tracking."""
        initial_count = self.handler.get_action_count()
        
        self.handler.action_count += 5
        self.assertEqual(self.handler.get_action_count(), initial_count + 5)
        
        self.handler.reset_action_count()
        self.assertEqual(self.handler.get_action_count(), 0)


if __name__ == '__main__':
    unittest.main()
