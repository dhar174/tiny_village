# -*- coding: utf-8 -*-
# It appears there was confusion in calling the function due to a reset in the execution state.
# Let's redefine the function and call it correctly with the updated doses.


def calculate_daily_levels(doses, days_until_test, half_life_hours=12):
    """
    Calculate the estimated blood levels of Adderall for each day up until the urinalysis.

    :param doses: List of daily doses of Adderall in mg.
    :param days_until_test: Number of days after the first dose when the urinalysis is performed.
    :param half_life_hours: Half-life of Adderall in hours. Default is 12 hours.
    :return: List of estimated blood levels for each day.
    """
    # Initialize a list to hold the blood levels for each day
    daily_levels = [0] * days_until_test

    # Loop through each day up to the day of the urine test
    for day in range(days_until_test):
        # Start with the dose taken on the current day (if any)
        daily_level = doses[day] if day < len(doses) else 0

        # Add the diminished amounts of the doses from previous days
        for past_day in range(day):
            # Calculate how many half-lives have passed since the dose was taken
            half_lives_passed = (day - past_day) * 24 / half_life_hours
            # Add the remaining amount of this past dose to the current day's level
            daily_level += doses[past_day] * (0.5**half_lives_passed)

        # Store the calculated level for the current day
        daily_levels[day] = daily_level

    return daily_levels


# Recalculate the daily levels with the updated dose for day 4 (25 mg instead of 15 mg)
doses_updated = [40, 40, 0, 25, 10, 5]  # Updated doses
days_until_test = 6  # The urinalysis is performed on day 6

daily_levels_updated = calculate_daily_levels(doses_updated, days_until_test)
daily_levels_updated


# Given the information that urinary pH can significantly affect the clearance of Adderall,
# and considering the rough guide that each unit increase or decrease in urinary pH produces a respective 7-hour increase or decrease in plasma half-life,
# let's adjust the code to account for urinary pH effects on the half-life of Adderall.


def calculate_daily_levels_ph_adjusted(
    doses, days_until_test, base_half_life_hours=12, urinary_pH=7
):
    """
    Calculate the estimated blood levels of Adderall for each day up until the urinalysis, adjusting for urinary pH.

    :param doses: List of daily doses of Adderall in mg.
    :param days_until_test: Number of days after the first dose when the urinalysis is performed.
    :param base_half_life_hours: Base half-life of Adderall in hours, before adjustment for urinary pH.
    :param urinary_pH: Urinary pH, used to adjust the half-life of Adderall.
    :return: List of estimated blood levels for each day.
    """
    # Adjust half-life based on urinary pH: each unit change from neutral (pH 7) alters half-life by 7 hours
    pH_adjustment = (urinary_pH - 7) * 7
    adjusted_half_life_hours = base_half_life_hours + pH_adjustment

    # Initialize a list to hold the blood levels for each day
    daily_levels = [0] * days_until_test

    # Loop through each day up to the day of the urine test
    for day in range(days_until_test):
        # Start with the dose taken on the current day (if any)
        daily_level = doses[day] if day < len(doses) else 0

        # Add the diminished amounts of the doses from previous days
        for past_day in range(day):
            # Calculate how many half-lives have passed since the dose was taken
            half_lives_passed = (day - past_day) * 24 / adjusted_half_life_hours
            # Add the remaining amount of this past dose to the current day's level
            daily_level += doses[past_day] * (0.5**half_lives_passed)

        # Store the calculated level for the current day
        daily_levels[day] = daily_level

    return daily_levels
