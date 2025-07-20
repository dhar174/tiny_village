import unittest
from tiny_time_manager import GameCalendar
import datetime
import time

class TestGameCalendar(unittest.TestCase):
    def setUp(self):
        self.calendar = GameCalendar()

    def test_set_real_time_begin(self):
        self.assertIsNotNone(self.calendar.set_real_time_begin())
        self.assertIsInstance(self.calendar.get_real_time_begin(), float)

    def test_get_game_time(self):
        self.calendar.test_seconds(70000)
        game_time = self.calendar.get_game_time()
        print(game_time)
        self.assertIsNotNone(game_time)
        self.assertIsInstance(game_time, datetime.datetime)

    def test_get_game_time_string(self):
        game_time_string = self.calendar.get_game_time_string()
        self.assertIsNotNone(game_time_string)
        self.assertIsInstance(game_time_string, str)

    def test_game_started(self):
        session_begin = self.calendar.game_started()
        self.assertIsNotNone(session_begin)
        self.assertIsInstance(session_begin, float)

    def test_get_session_time(self):
        session_time = self.calendar.get_session_time()
        self.assertIsNotNone(session_time)
        self.assertIsInstance(session_time, float)

    def test_game_end(self):
        accumulated_timer = self.calendar.game_end()
        self.assertIsNotNone(accumulated_timer)
        self.assertIsInstance(accumulated_timer, float)

if __name__ == '__main__':
    unittest.main()