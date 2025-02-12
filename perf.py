import time

class PerfExec:
    def __init__(self, name, perf_counter):
        self.start_time = time.perf_counter_ns()
        self.end_time = 0
        self.perf_counter = perf_counter
        self.name = name

    def end(self):
        self.end_time = time.perf_counter_ns()
        self.perf_counter._end(self)
        return self.end_time - self.start_time

    def total(self):
        return self.end_time - self.start_time

    def print(self):
        print(f"Profile {self.name} took {self.total() / 1000000} milliseconds")

    def get_name(self):
        return self.name


def convert_time(time_in_ns):
    if time_in_ns < 1000:
        return str(time_in_ns) + " nanoseconds"
    elif time_in_ns < 1000000000:
        return str(time_in_ns / 1000000) + " milliseconds"
    else:
        return str(time_in_ns / 1000000000) + " seconds"


class PerfCounter:
    def __init__(self):
        self.profiles = {}

    def start(self, name):
        return PerfExec(name, self)

    def _end(self, perf):
        if perf.name not in self.profiles:
            self.profiles[perf.name] = []
        self.profiles[perf.name].append(perf)

    def print_profiles(self):
        for name, events in self.profiles.items():
            total = 0
            for event in events:
                total += event.total()
            print(f"Profile {name} took on average {convert_time(total / len(events))} over {len(events)} events, total time was {convert_time(total)} milliseconds")