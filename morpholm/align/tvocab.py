#!/usr/bin/env python
import sys
import re

nt_re = re.compile('\[X,\d+\]')

def main():
    for line in sys.stdin:
        _, _, e, _, _ = line.split(' ||| ') # [X] ||| f ||| e ||| features ||| alignment
        for w in e.split():
            if nt_re.match(w): continue
            print w

if __name__ == '__main__':
    main()
