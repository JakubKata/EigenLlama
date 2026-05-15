# LLM Engine (C++ & Python Hybrid)

LLM is a lightweight, high-performance project combining a **C++17 core** with a **Python orchestration layer**. It implements a compact, dependency-light LLaMA-like runtime suitable for running small models (e.g. TinyLlama) without heavy frameworks such as PyTorch.

Core design goals:

- High-performance matrix/normalization kernels implemented in C++ (Eigen)
- Transparent Python implementation of transformer layers (RoPE, Attention, MLP)
- Fast, zero-copy weight loading using `safetensors`
- Minimal dependencies: `pybind11` for bindings and NumPy for numerics

---

## Features

- Custom C++ core for heavy math (Eigen-backed)
- Python-side layer implementations for quick experimentation
- Hybrid workflow: compute-heavy code in C++ (pybind11), orchestration in Python
- Zero-copy weight loading via `safetensors`
- Tokenization support via `transformers` (or other tokenizer libraries)

---

## Requirements

- Python 3.10+
- C++17-capable compiler (MSVC, GCC, Clang)
- Recommended Python packages: `numpy`, `pybind11`, `safetensors`, `transformers`, `ml-dtypes`

Note: The Python installation builds a native extension via `pybind11` (see `setup.py`). CMake is optional for Python users — use it only when you want standalone native artifacts or IDE project files.

---

## Quick Start

1) Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Linux / macOS:

```bash
python -m venv venv
source venv/bin/activate
```

2) Install Python dependencies and build the extension

```bash
pip install -e .
# or, equivalently:
python setup.py build_ext --inplace
```

The `setup.py` defines a `Pybind11Extension` (the extension target is named `tiny_math`) built from `core/LLMEngine.cpp`. Installing the package this way compiles the native extension and makes the `model` package available for import.

3) Run the interactive CLI

```bash
python model/main.py
```

Or call the Python API directly in your own scripts:

```python
from model.engine import TinyLlamaEngine

# Create an engine pointing at the `tinyllama_files` directory
engine = TinyLlamaEngine("tinyllama_files")
engine.generate("Hello, world!", max_tokens=20)
```

Note: Running `model/main.py` launches a simple interactive REPL that constructs `TinyLlamaEngine` and prompts for input in an infinite loop.

If you need direct access to the native core from C++, build the `core` directory with CMake:

```bash
cmake -S core -B core/build
cmake --build core/build --config Release
```

---

## Notes on building

- The recommended way for Python users is `pip install -e .` which triggers compilation via setuptools + pybind11.
- Use CMake only when you need standalone native artifacts, custom compile flags, or IDE integration.
- Check `setup.py` for the exact extension name and source files.

---

## Project structure

```text
LLM/
├── .gitmodules         # Git submodules (e.g., Eigen)
├── LICENSE
├── pyproject.toml      # Build system requirements (PEP 518)
├── setup.py            # Builds pybind11 extension from core/LLMEngine.cpp
├── tiny_math.*.pyd     # Compiled native C++ extension (generated after build)
├── core/               # C++ core (optional CMake workflow)
│   ├── lib/            # External C++ libraries (e.g., Eigen)
│   ├── CMakeLists.txt
│   └── LLMEngine.cpp
├── model/              # Python package (engine, layers, CLI)
│   ├── __init__.py
│   ├── engine.py       
│   ├── layers.py       
│   └── main.py         # Interactive REPL script
├── tinyllama_files/    # Model/tokenizer files (safetensors, tokenizer.json, etc.)
└── tools/              # Helper scripts
    └── huggingface_hub_download.py
```

---

## License

MIT