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

    def _dish_tables(self, k): # all the tables labeled with dish k
        if k in self.tables:
            # existing tables
            for i, n in enumerate(self.tables[k]):
                yield i, n-self.d
            # new table
            yield -1, (self.theta + self.d * self.ntables) * math.exp(self.base.prob(k))
        else:
            yield -1, 1

    def _seat_at(self, k, i): # seat at (dish, table)
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

    def increment(self, k):
        i = mult_sample(self._dish_tables(k))
        self._seat_at(k, i)

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
                    """
                    if len(tables) == 0: # cleanup dish (useless: dish will be re-incremented)
                        del self.tables[k]
                        del self.ncustomers[k]
                    """
                break
            i -= tables[t]
    
    def prob(self, k): # total prob for dish k
        # new table
        w = (self.theta + self.d * self.ntables) * math.exp(self.base.prob(k))
        # existing tables
        if k in self.tables:
            w += self.ncustomers[k] - self.d * len(self.tables[k])
        return math.log(w / (self.theta + self.total_customers))
    
    def pred_weight(self, k):
        return self.prob(k) # TODO: check

    def log_likelihood(self):
        return sum(count * self.prob(k)
                for k, count in self.ncustomers.iteritems())

    def __str__(self):
        return 'PYPLM(d={self.d}, theta={self.theta}, #customers={self.total_customers}, #tables={self.ntables}, #dishes={V}, Base={self.base})'.format(self=self, V=len(self.tables))
