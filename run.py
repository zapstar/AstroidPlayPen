"""
Some more docstring
"""
import foo_pb2


def main():
    """
    Some docstring
    :return: Nothing
    """
    out = foo_pb2.Outer()
    # Expecting to raise `attribute-defined-outside-init`
    out.my_foobar = False
    # Expecting to NOT raise `no-member`
    out.my_inner2.my2_inner1.my1_name = "Hello"
    print(out)


if __name__ == '__main__':
    main()
