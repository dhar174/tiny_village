# Description: A tiny time manager for a game engine
import calendar
import datetime

import time
from typing import Callable

class GameCalendar:
    def __init__(self, start_year=2023, start_month=1, start_day=1, start_hours=0, start_minutes=0, start_seconds=0):
        self.years = [year for year in range(2023, 9999)]
        self.months = [month for month in range(1, 13)]
        self.days = [day for day in range(1, 31)]
        self.hours = [hour for hour in range(0, 24)]
        self.minutes = [minute for minute in range(0, 60)]
        self.seconds = [second for second in range(0, 60)]
        self.real_time_begin = self.set_real_time_begin()
        self.accumulated_timer = 0
        self.session_begin = self.game_started()
        self.session_end = None
        self.game_time = None
        self.session_time = None
        
        self.unit_conversion_seconds = {
            "year": 950400,
            "month": 79200,
            "day": 2640,
            "hour": 220,
            "minute": 10,
        }
        self.current_year = start_year
        self.current_month = start_month
        self.current_day = start_day
        self.current_hour = start_hours
        self.current_minute = start_minutes
        self.current_second = start_seconds


    def set_real_time_begin(self):
        self.real_time_begin = time.time()
        return self.real_time_begin

    def get_real_time_begin(self):
        return self.real_time_begin
    
    def get_game_time(self):
        t = self.accumulated_timer + self.get_session_time()
        self.current_year += int(t / self.unit_conversion_seconds["year"])
        t = t % self.unit_conversion_seconds["year"]
        self.current_month += int(t / self.unit_conversion_seconds["month"])
        t = t % self.unit_conversion_seconds["month"]
        self.current_day += int(t / self.unit_conversion_seconds["day"])
        t = t % self.unit_conversion_seconds["day"]
        self.current_hour += int(t // self.unit_conversion_seconds["hour"])
        t = t % self.unit_conversion_seconds["hour"]
        self.current_minute += int(t / self.unit_conversion_seconds["minute"])
        t = t % self.unit_conversion_seconds["minute"]
        self.current_second += int(t)

        # Ensure the time units are within their valid ranges
        self.current_second %= len(self.seconds)
        self.current_minute %= len(self.minutes)
        self.current_hour %= len(self.hours)
        self.current_day %= len(self.days)
        self.current_month %= len(self.months)
        self.current_year %= len(self.years)

        self.game_time = datetime.datetime(self.current_year, self.current_month, self.current_day, self.current_hour, self.current_minute, self.current_second)
        return self.game_time
    
    def get_game_time_string(self):
        return self.get_game_time().strftime("%Y-%m-%d %H:%M:%S")
    
    def game_started(self):
        self.session_begin = time.time()
        return self.session_begin
    
    def get_session_time(self):
        self.session_time = time.time() - self.session_begin
        return self.session_time
    
    def game_end(self):
        self.session_end = time.time()
        self.accumulated_timer += self.session_end - self.session_begin
        return self.accumulated_timer
    
    def test_seconds(self, seconds):
        self.accumulated_timer += seconds
        return self.accumulated_timer

class ScheduledBehavior:
    def __init__(self, name, cycle_length, cycle_unit, behavior: Callable, calendar: GameCalendar):
        self.name = name
        self.cycle_length = cycle_length
        self.cycle_unit = cycle_unit
        self.last_cycle = calendar.get_game_time().strftime(f"%{self.cycle_unit}")
        self.behavior = behavior

    def check_calendar(self):
        if calendar.get_game_time().strftime(f"%{self.cycle_unit}") == self.last_cycle + self.cycle_length:
            self.last_cycle = calendar.get_game_time().strftime(f"%{self.cycle_unit}")
            self.behavior()

    def get_name(self):
        return self.name
    
    def get_cycle_length(self):
        return self.cycle_length
    
    def get_cycle_unit(self):
        return self.cycle_unit
    
    def get_last_cycle(self):
        return self.last_cycle
    
    def get_behavior(self):
        return self.behavior
    
    def to_dict(self):
        return {
            "name": self.name,
            "cycle_length": self.cycle_length,
            "cycle_unit": self.cycle_unit,
            "last_cycle": self.last_cycle,
            "behavior": self.behavior
        }

class GameTimeManager:
    def __init__(self, calendar: GameCalendar):
        self.calendar = calendar
        self.scheduled_behaviors = []
    
    def add_scheduled_behavior(self, scheduled_behavior: ScheduledBehavior):
        self.scheduled_behaviors.append(scheduled_behavior)
    
    def remove_scheduled_behavior(self, scheduled_behavior: ScheduledBehavior):
        self.scheduled_behaviors.remove(scheduled_behavior)
    
    def get_scheduled_behaviors(self):
        return self.scheduled_behaviors
    
    def get_calendar(self):
        return self.calendar
    
    def to_dict(self):
        return {
            "calendar": self.calendar.to_dict(),
            "scheduled_behaviors": [scheduled_behavior.to_dict() for scheduled_behavior in self.scheduled_behaviors]
        }
    
    def run(self):
        while True:
            for scheduled_behavior in self.scheduled_behaviors:
                scheduled_behavior.check_calendar()
            time.sleep(1)