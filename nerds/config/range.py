from nerds.util.logging import get_logger

log = get_logger()


class ValidRange(object):
    def contains(self, value):
        raise NotImplementedError


class ValidSetRange(ValidRange):
    def __init__(self, valid_range, default_value):
        self.valid_range = valid_range
        self.default_value = default_value

    def contains(self, value):
        return value in self.valid_range


class ValidFloatRange(ValidRange):
    # excluding start_value and end_value
    def __init__(self, start_value, end_value, default_value):
        self.start_value = start_value
        self.end_value = end_value
        self.default_value = default_value

    def contains(self, value):
        return self.start_value < value < self.end_value
