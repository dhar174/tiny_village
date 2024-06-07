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
    try:
        float(value)
        return True
    except ValueError:
        return False


def tweener(input_value, max_input, start, end, steps):
    if input_value >= max_input:
        return end
    else:
        return start + ((end - start) * (input_value / max_input))
