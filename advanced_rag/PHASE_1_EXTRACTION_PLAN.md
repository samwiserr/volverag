# Phase 1 Extraction Plan for generate_answer

## Current Status
- `generate_answer` function is ~1350 lines in `nodes.py` (lines 932-2281)
- Function has complex dependencies on:
  - `_latest_user_question`, `_infer_recent_context` (from `utils/message_utils.py`)
  - `_get_response_model()`, `_get_registry()` (from `nodes.py`)
  - `GENERATE_PROMPT` (from `nodes.py`)
  - `extract_well`, `normalize_formation` (from `normalize.query_normalizer`)
  - `resolve_property_deterministic`, `choose_property_with_agent` (from normalize modules)
  - `logger` (from `nodes.py`)
  - Standard library: `re`, `Path`, `json`, `Optional`

## Extraction Strategy

### Step 1: Create `generation/answer.py`
- Extract the entire `generate_answer` function
- Import dependencies carefully to avoid circular imports:
  - Use `from ..utils.message_utils import _latest_user_question, _infer_recent_context`
  - Use function-level imports for `nodes.py` dependencies to avoid circular imports:
    ```python
    def generate_answer(state: MessagesState):
        # Import from nodes at function level to avoid circular import
        from ..nodes import _get_response_model, _get_registry, GENERATE_PROMPT, logger
        # ... rest of function
    ```

### Step 2: Update `nodes.py`
- Replace the function definition with an import and re-export:
  ```python
  from .generation.answer import generate_answer
  ```

### Step 3: Test
- Run all tests to ensure no regression
- Verify Streamlit app works
- Check GitHub Actions CI passes

## Risk Mitigation
- Use function-level imports to avoid circular dependencies
- Maintain exact function signature
- Test after each step
- Can rollback if issues arise

