Steps
=====

Please make sure you have `Python 3.x` with `venv` support.
* First run `make venv` to create a virtual environment
* Next run `. venv/bin/activate` to activate the virtual environment
* Please install the protobuf compiler from [here](https://github.com/protocolbuffers/protobuf/releases/). Now compile the protobuf with `make compile`
  `OR`
  Download the generated protobuf file from [here](https://gist.github.com/zapstar/c874336c5b0d70e8ce5ce92486f81e74/raw/91dca4ebbd0e437ba854983a51b417e0f2cd5e0e/foo_pb2.py)
* See what I was expecting by running `make expected`
* See what my code generates via `make check`
