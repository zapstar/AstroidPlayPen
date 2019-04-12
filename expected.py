#!/usr/bin/env python
"""
Some Test class
"""

class Boo:
    """Some class Boo"""
    def __init__(self):
        """Constructor for Boo"""
        self.coo = 0
        self.doo = 0

    def moo(self):
        """Method M"""
        return self.coo

    def noo(self):
        """Method N"""
        return self.doo


class Aoo:
    """Class Aoo"""
    def __init__(self):
        """Constructor for A"""
        self.eoo = 0
        self.hoo = 1
        self.goo = 2

    def xoo(self):
        """Method xoo"""
        return self.eoo

    def yoo(self):
        """Method yoo"""
        return self.hoo

    def zoo(self):
        """Method zoo"""
        return self.goo


def main():
    """Main method"""
    obj_a = Aoo()
    obj_b = Boo()
    # Should raise a `attribute-defined-outside-init`
    obj_b.too = 1
    # Should raise a `no-member`
    obj_a.voo.woo = 2

if __name__ == '__main__':
    main()
