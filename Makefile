# Install virtual environment
venv:
	python3 -m venv venv

# Install dependencies
deps:
	pip3 install -r requirements.txt

# Compile the protobuf if you have a Protobuf compiler
compile:
	protoc --python_out=. *.proto

# Expected output (what I was expecting)
expected:
	pylint expected.py

# Use my plugin to test transformation
check:
	pylint --init-hook='import sys; sys.path.insert(0, ".")' --load-plugins=checker run.py

# Don't use this unless you have a protobuf compiler
clean:
	rm -f *_pb2.py
	rm -rf __pycache__
	rm -r venv
