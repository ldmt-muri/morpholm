import math
import random
from prob import mult_sample

class CRP(object):
    def __init__(self):
        self.tables = {}
        self.ntables = 0
        self.ncustomers = {}
        self.total_customers = 0

    def _seat_to(self, k, i):
        if not k in self.tables: # add new dish
            self.tables[k] = []
            self.ncustomers[k] = 0
        self.ncustomers[k] += 1
        self.total_customers += 1
        tables = self.tables[k]
        if i == -1: # add new table
            self.ntables += 1
            tables.append(1)
        else: # existing table
            tables[i] += 1
        return (i == -1)

    def _unseat_from(self, k, i):
        self.ncustomers[k] -= 1
        self.total_customers -= 1
        tables = self.tables[k]
        tables[i] -= 1
        if tables[i] == 0: # cleanup empty table
            del tables[i]
            self.ntables -= 1
            if len(tables) == 0: # cleanup dish
                del self.tables[k]
                del self.ncustomers[k]
            return True
        return False

class PYP(CRP):
    def __init__(self, theta, d, base):
        super(PYP, self).__init__()
        self.theta = theta
        self.d = d
        self.base = base

    def _dish_tables(self, k): # all the tables labeled with dish k
        if k in self.tables:
            # existing tables
            for i, n in enumerate(self.tables[k]):
                yield i, n-self.d
            # new table
            yield -1, (self.theta + self.d * self.ntables) * math.exp(self.base.prob(k))
        else:
            yield -1, 1

    def _customer_table(self, k, n): # find table index of nth customer with dish k
        tables = self.tables[k]
        for i in xrange(len(tables)):
            if n < tables[i]: return i
            n -= tables[i]

    def increment(self, k):
        i = mult_sample(self._dish_tables(k))
        if self._seat_to(k, i):
            self.base.increment(k)

    def decrement(self, k):
        i = self._customer_table(k, random.randint(0, self.ncustomers[k]-1))
        if self._unseat_from(k, i):
            self.base.decrement(k)
    
    def prob(self, k): # total prob for dish k
        # new table
        w = (self.theta + self.d * self.ntables) * math.exp(self.base.prob(k))
        # existing tables
        if k in self.tables:
            w += self.ncustomers[k] - self.d * len(self.tables[k])
        return math.log(w / (self.theta + self.total_customers))

    def pseudo_log_likelihood(self):
        return sum(count * self.prob(k)
                for k, count in self.ncustomers.iteritems())

    def log_likelihood(self, base=True):
        ll = (math.lgamma(self.theta) - math.lgamma(self.theta + self.total_customers)
                + math.lgamma(self.theta / self.d + self.ntables)
                - math.lgamma(self.theta / self.d)
                + self.ntables * (math.log(self.d) - math.lgamma(1 - self.d))
                + sum(math.lgamma(n - self.d) for tables in self.tables.itervalues()
                    for n in tables))
        if base:
            ll += self.base.log_likelihood()
        return ll

    def decode(self, k):
        p0, dec = self.base.decode(k)
        w = (self.theta + self.d * self.ntables) * math.exp(p0)
        if k in self.tables:
            w += self.ncustomers[k] - self.d * len(self.tables[k])
        return math.log(w / (self.theta + self.total_customers)), dec

    def __repr__(self):
        return 'PYP(d={self.d}, theta={self.theta}, #customers={self.total_customers}, #tables={self.ntables}, #dishes={V}, Base={self.base})'.format(self=self, V=len(self.tables))

class BackoffBase:
    def __init__(self, model, ctx):
        self.model = model
        self.ctx = ctx

    def increment(self, k):
        return self.model.increment(self.ctx+(k,))

    def decrement(self, k):
        return self.model.decrement(self.ctx+(k,))

    def prob(self, k):
        return self.model.prob(self.ctx+(k,))

class PYPLM:
    def __init__(self, theta, d, order, initial_base):
        self.theta = theta
        self.d = d
        self.order = order
        self.backoff = initial_base if order == 1 else PYPLM(theta, d, order-1, initial_base)
        self.models = {}

    def __getitem__(self, ctx):
        if ctx not in self.models:
            self.models[ctx] = self._get(ctx)
        return self.models[ctx]

    def _get(self, ctx):
        if ctx not in self.models:
            base = (self.backoff if self.order == 1 else BackoffBase(self.backoff, ctx[1:]))
            return PYP(self.theta, self.d, base)
        return self.models[ctx]

    def increment(self, k):
        self[k[:-1]].increment(k[-1])

    def decrement(self, k):
        self.models[k[:-1]].decrement(k[-1])

    def prob(self, k):
        return self._get(k[:-1]).prob(k[-1])

    def log_likelihood(self):
        return (sum(m.log_likelihood(base=False) for m in self.models.itervalues())
                + self.backoff.log_likelihood())

    def __repr__(self):
        return 'PYPLM(d={self.d}, theta={self.theta}, #ctx={C}, backoff={self.backoff})'.format(self=self, C=len(self.models))
