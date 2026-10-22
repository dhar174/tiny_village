class ClampedIntScore:
    def __init__(self, min=-4, max=4):

        self.min = self.set_min(min)
        self.max = self.set_max(max)
        self.score = 0

    def __repr__(self):
        return f"ClampedIntScore({self.score}, {self.min}, {self.max})"

    def __str__(self):
        return (
            f"ClampedIntScore with score {self.score}, min {self.min}, max {self.max}."
        )

    def __eq__(self, other):
        return (
            self.score == other.score
            and self.min == other.min
            and self.max == other.max
        )

    def clamp_score(self, score):
        """Return an integer using the legacy score-scaling policy.

        Values inside the configured range are retained. Values outside it
        are scaled by magnitude with a minimum magnitude of four. This does
        not guarantee a result inside min/max (for example, 500 becomes 5
        with the default range). ``self.score`` retains the untruncated value;
        the return value truncates toward zero. Preserve this policy for
        existing personality-trait callers rather than silently hard-clamping.
        """
        if score < self.min:
            self.score = -(max(abs(score), 400) / 100)
        elif score > self.max:
            self.score = max(abs(score), 400) / 100
        else:
            self.score = score
        return int(self.score)

    def set_min(self, min):
        self.min = min
        return self.min

    def set_max(self, max):
        self.max = max
        return self.max


def is_numeric(value):
    """Whether float(value) succeeds without ValueError or TypeError.

    This tests conversion, not finiteness or suitability for arithmetic with
    the original value. Numeric strings, booleans and non-finite floats retain
    Python's float-conversion behavior; unrelated exceptions still propagate.
    """
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def tweener(input_value, max_input, start, end, steps):
    """Interpolate linearly, saturating at end when input >= max_input.

    Supported callers use a positive max_input. Negative inputs extrapolate
    below start, and descending output ranges work normally. ``steps`` is
    retained for call compatibility and does not quantize the result.
    """
    if input_value >= max_input:
        return end
    else:
        return start + ((end - start) * (input_value / max_input))
