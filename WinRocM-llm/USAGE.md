# vllm-windows — How to build & run

A Windows-native C++ engine that dispatches vLLM-style attention to AMD
ROCm kernels (gfx1201 = RDNA4 / RX 9070 XT). It runs as a standalone
`vllm_engine.exe` and uses a C-ABI bridge into a libtorch/ROCm shared object.

## Prerequisites

| Component | Version tested | How to get |
|---|---|---|
| AMD GPU | RX 9070 XT (gfx1201) | physical card |
| ROCm / HIP SDK | **HIP 7.16.26323** | https://rocmdocs.amd.com (ROCm for Windows) |
| Python | 3.12 | `python.org` or `winget install Python.Python.3.12` |
| PyTorch (ROCm Windows) | `2.15.0a0+rocm10.1.0` | `pip install torch==2.15.0a0+rocm10.1.0a20260816 -f https://download.pytorch.org/whl/torch_stable.html` **and** the AMD ROCm wheel index — see AMD's "Install PyTorch (Windows)" guide |
| Model checkpoint | any HF/vLLM-format | your `.safetensors` dir |

> `cl.exe` is **not** used. All compilation goes through
> `G:\ROCM10RT-gfx1201\lib\llvm\bin\clang++`/`hipcc` + GNU `ld.lld`.

## 1. Install ROCm

Install the HIP SDK to `G:\ROCM10RT-gfx1201` (or any path — edit
`ROCM_HOME` in the next step). Required components:

- `bin/hipcc`, `bin/clang++`, `bin/lld-link` (or `ld.lld`)
- `include/hip/`, `include/hip/hcc`, `lib/hip/hipblas/lib`, `lib/amdhip64.lib`
- the GCN bitcode dir `lib/llvm/amdgcn/bitcode`

## 2. Install PyTorch (ROCm build, Windows)

```powershell
pip install torch==2.15.0a0+rocm10.1.0a20260816  # from AMD's ROCm wheel index
python -c "import torch; print(torch.__version__, torch.version.hip)"
```

The wheel ships `torch/include/` (the C++ headers) and
`torch/lib/*.lib` (import libs: `c10.lib`, `torch.lib`, `c10_hip.lib`, …).
The path to `torch/include/` becomes `TORCH_INC` in the build script.

## 3. Configure build paths

Edit the three variables at the top of
[`cmake/build_host.cmd`](WinRocM-llm/cmake/build_host.cmd):

```bat
set "ROCM_HOME=G:\ROCM10RT-gfx1201"
set "TORCH_INC=<your-python>/Lib/site-packages/torch/include"
set "MODEL_DIR=<your-checkpoint>"     (optional, only for full --run)
```

## 4. Build

From a **Developer PowerShell** / cmd prompt at the repo root:

```bat
cmake\build_host.cmd --self-check     :: build + run smoke test
cmake\build_host.cmd --attention     :: build + launch --attention (needs rocm_ops.dll)
```

What it does:

1. compiles the engine .cpp TUs (clang++, GNU triple + `_MSC_VER` shim);
2. compiles the `attention_gfx1201.cu` device kernel (hipcc, gfx1201);
3. links `vllm_engine.exe` with `ld.lld` + `amdhip64.lib` (no cl.exe objects);
4. if `BUILD_ROCM_OPS_DLL=1` is set, **also** builds `rocm_ops.pyd`.

Artifacts land in `WinRocM-llm\build-host\`.

## 5. Smoke test (no model needed)

```bat
WinRocM-llm\build-host\vllm_engine.exe --self-check
```

Expected:
```
GPU 0: AMD Radeon RX 9070 XT (gfx1201) PCI=xx:xx:x VRAM=xMB
self-check: 1 GPU(s) visible
  ...
  KV round-trip: ok
  prefix-cache re-shared: ok
  child prefix-cache shared: ok
  post-eviction alloc: ok
self-check PASS
```

## 6. Run a model (attention)

```bat
set BUILD_ROCM_OPS_DLL=1 && cmake\build_host.cmd --attention
WinRocM-llm\build-host\vllm_engine.exe --attention --model %MODEL_DIR%
```

If `BUILD_ROCM_OPS_DLL` is unset (default), `--attention` fails closed:
```
FAIL: execute threw: unable to load rocm_ops.dll (C-ABI bridge D017)
```
i.e. the engine builds and runs, but the torch extension isn't present
yet — flip the env var and rebuild.

## Project layout

```
WinRocM-llm/
  cmake/build_host.cmd          build driver (clang++/hipcc/ld.lld)
  src/engine/...                  C++ model runner + scheduler + block table
  src/engine/include/...          public headers
  src/hip_wrap/...                HIP/ROCm minimal shims
  src/cli/main.cpp                vllm_engine.exe entry (--self-check / --attention)
csrc/rocm/                        kernel sources (attention_gfx1201.cu, …)
csrc/hip_wrap/                    cuda<->hip runtime alias shim headers
tests/hip/                        standalone HIP smoke kernels
knowledge/                        design notes (decisions.log.md, toolchain.dox.md …)
```

## Troubleshooting

- `__stddef_wchar_t.h` errors → the kernel compile omits `-fshort-wchar`
  (fixed upstream in `build_host.cmd`); if it recurs, ensure `-fshort-wchar`
  is present with `-D_MSC_VER=1900`.
- `hipMalloc` returns `hipErrorOutOfMemory` → the GPU is seen but VRAM init
  failed; re-plug the card and confirm in Device Manager it is *not*
  running a compatibility driver.
- `unable to load rocm_ops.dll` → set `BUILD_ROCM_OPS_DLL=1` and rebuild;
  the host exe deliberately does not statically link torch headers (D014).
