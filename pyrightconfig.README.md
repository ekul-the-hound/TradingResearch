# pyrightconfig.json -- why each setting exists

JSON has no comment syntax, so these explanations live here instead of as
"//" keys inside the config (those keys made pyright reject the whole file,
which silently disabled the `exclude` list and flooded the report with errors
from generated code).

- **venvPath / venv**: Points Pylance at the quant2 conda environment. Without
  this it falls back to a base interpreter with no pandas, numpy, or backtrader,
  and reports every import as unresolved. Pylance reads this file directly, so
  it survives VS Code settings being reset or the project opening on another
  machine.

- **typeCheckingMode: basic**: Catches real errors without the strict-mode noise.

- **reportMissingModuleSource: none**: pandas ships type stubs separately. When
  only the stubs resolve, Pylance warns the source is missing -- harmless noise
  once the interpreter is correct, so it is silenced here.

- **exclude**: Restores Pylance's default excludes (node_modules, __pycache__,
  dotfiles) alongside project-specific ones. `discovered_strategies` and
  `strategies/variants` are AUTO-GENERATED and regenerated every run -- type
  errors in them are noise, not bugs, so they are excluded to keep the report
  focused on hand-maintained code.
