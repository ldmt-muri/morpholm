import math
import random
from prob import mult_sample

class PYP:
    def __init__(self, theta, d, base):
        self.theta = theta
        self.d = d
        self.base = base
        self.tables = {}
        self.ntables = 0
        self.ncustomers = {}
        self.total_customers = 0

    def increment(self, k):
        i = mult_sample(self.dish_tables(k))
        self.seat_at(k, i)

    def decrement(self, k):
        i = random.randint(0, self.ncustomers[k]-1)
        tables = self.tables[k]
        for t in xrange(len(tables)):
            if i < tables[t]:
                tables[t] -= 1
                self.ncustomers[k] -= 1
                self.total_customers -= 1
                if tables[t] == 0: # cleanup empty table
                    del tables[t]
                    self.ntables -= 1
                    self.base.decrement(k)
                    if len(tables) == 0: # cleanup dish
                        del self.tables[k]
                        del self.ncustomers[k]
                break
            i -= tables[t]

    def dish_tables(self, k): # all the tables labeled with dish k
        # existing tables
        if k in self.tables:
            for i, n in enumerate(self.tables[k]):
                yield i, n-self.d
        # new table
        yield -1, (self.theta + self.d * self.ntables) * math.exp(self.base.prob(k))
    
    def prob(self, k): # total prob for dish k
        # new table
        w = (self.theta + self.d * self.ntables) * math.exp(self.base.prob(k))
        # existing tables
        if k in self.tables:
            w += self.ncustomers[k] - self.d * len(self.tables[k])
        return math.log(w / (self.theta + self.total_customers))
    
    def seat_at(self, k, i): # seat at (dish, table)
        if not k in self.tables:
            self.tables[k] = []
            self.ncustomers[k] = 0
        tables = self.tables[k]
        if i == -1: # new table
            self.ntables += 1
            tables.append(1)
            self.base.increment(k)
        else: # existing table
            tables[i] += 1
        self.ncustomers[k] += 1
        self.total_customers += 1

    def log_likelihood(self):
        return sum(count * self.prob(k)
                for k, count in self.ncustomers.iteritems())
