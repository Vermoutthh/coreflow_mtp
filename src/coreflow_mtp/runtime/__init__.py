"""V1 execution adapter, independent of CoreFlow's vendored vLLM 0.9.2.

Importing this package does not import torch, CUDA or vLLM.  Those dependencies
are loaded only when a real instance starts.
"""

