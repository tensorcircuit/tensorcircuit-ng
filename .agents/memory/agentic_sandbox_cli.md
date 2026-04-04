# Agentic Sandbox & CLI Protocols

1.  **Environment Variables for Read-Only Runtimes**:
    - **Context**: The agentic sandbox environment often has restricted write permissions to default configuration and cache directories (e.g., `~/.cache`, `~/.config`).
    - **Protocol**: Always prepend critical cache redirection variables when running scripts that use Numba, Matplotlib, or JAX.
    - **Required Variables**:
      - `NUMBA_CACHE_DIR=/tmp/numba_cache`: Prevents `PermissionError` when Numba (used by `quimb`, `scipy`) tries to write specialized kernels.
      - `MPLCONFIGDIR=/tmp/matplotlib_cache`: Prevents `PermissionError` when `matplotlib` attempts to write font caches or configuration.
      - `mypy --cache-dir=/tmp/.mypy_cache`: Prevents `PermissionError` when `mypy` tries to write to the default `.mypy_cache` directory in the sandbox.
    - **Execution Template**:
      ```bash
      # For Python scripts
      NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib_cache conda run -n <env> python3 <script.py>

      # For Mypy checks
      conda run -n <env> mypy --cache-dir=/tmp/.mypy_cache <path>
      ```

2.  **Output Redirection and Paging**:
    - The sandbox terminal often handles large outputs poorly or requires non-interactive execution.
    - **Protocol**: Use `PAGER=cat` (often default in agent tools) to ensure logs are fully captured without getting stuck in a `less` prompt.
