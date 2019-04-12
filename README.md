Steps
=====

Please make sure you have `Python 3.x` with `venv` support.
* First run `make venv` to create a virtual environment.
* Next run `. venv/bin/activate` to activate the virtual environment.
* Now install dependencies using `make deps`.
* Please install the protobuf compiler from [here](https://github.com/protocolbuffers/protobuf/releases/). Now compile the protobuf with `make compile`.
  `OR`
  Download the generated protobuf file from [here](https://gist.github.com/zapstar/c874336c5b0d70e8ce5ce92486f81e74/raw/91dca4ebbd0e437ba854983a51b417e0f2cd5e0e/foo_pb2.py).
* See what I was expecting by running `make expected`.
* See what my code generates via `make check`.

Expected:
---------
```
$ make expected
pylint expected.py
************* Module expected
expected.py:48:4: W0201: Attribute 'too' defined outside __init__ (attribute-defined-outside-init)
expected.py:50:4: E1101: Instance of 'Aoo' has no 'voo' member; maybe 'eoo'? (no-member)

------------------------------------------------------------------
Your code has been rated at 7.69/10 (previous run: 7.69/10, +0.00)

make: *** [expected] Error 6
```

My Output
---------
```
$ make check
pylint --init-hook='import sys; sys.path.insert(0, ".")' --load-plugins=checker run.py
import enum


class Corpus(enum.Enum, metaclass=EnumMeta):
    UNIVERSAL = 0
    WEB = 1
    IMAGES = 2
    LOCAL = 3
    NEWS = 4
    PRODUCTS = 5
    VIDEO = 6



class Color(enum.Enum, metaclass=EnumMeta):
    RED = 0
    GREEN = 1
    BLUE = 2



class Inner1(object):
    
    def __init__(self):
        self.my1_name = ''



class Inner2(object):
    
    def __init__(self):
        self.my2_name = ''
        self.my2_inner1 = Inner1()



class Outer(object):
    
    def __init__(self):
        self.my_double = 0.0
        self.my_float = 0.0
        self.my_int64 = 0
        self.my_uint64 = 0
        self.my_int32 = 0
        self.my_fixed64 = 0
        self.my_fixed32 = 0
        self.my_bool = False
        self.my_string = ''
        self.my_inner2 = Inner2()
        self.my_bytes = b''
        self.my_uint32 = 0
        self.my_enum_color = Color.RED
        self.my_sfixed32 = 0
        self.my_sfixed64 = 0
        self.my_sint32 = 0
        self.my_sint64 = 0
        self.my_repeated_int32 = [0]
        self.my_repeated_corpus = [Corpus.UNIVERSAL]
        self.my_repeated_inner1 = [Inner1()]



************* Module run
run.py:16:4: E1101: Instance of 'Outer' has no 'my_inner2' member (no-member)

------------------------------------------------------------------
Your code has been rated at 3.75/10 (previous run: 3.75/10, +0.00)

make: *** [check] Error 2
```
