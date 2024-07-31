import pykeops
# Clear ~/.cache/pykeops2.1/...
pykeops.clean_pykeops()
# Rebuild from scratch the required binaries
pykeops.test_torch_bindings()