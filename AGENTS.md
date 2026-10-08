## Code language

All generated or modified code must use English only, including identifiers, comments, docstrings, logs, errors, tests, configuration, and embedded documentation. Explanations outside code may use Chinese.

## Backward compatibility

Backward compatibility with legacy artifacts is not required. New code may use the current schemas and formats directly without adding migrations, fallbacks, adapters, or compatibility branches unless explicitly requested.

## Test and Validation Code Rules

- Every file containing test, verification, or offline validation code must have a filename beginning with `test_`.
- All test and validation files must be placed under the top-level `test/` directory.
## Retained automated test scope

Keep only startup-to-order trading main-flow tests in the permanent automated
suite. Prefer updating an existing main-flow test when that flow changes.
Do not accumulate permanent bug-specific, timing, dashboard, concurrency or
other isolated regression tests after fixes. Use temporary verification for
such changes and remove it after validation. Explicit manual exchange and
offline model-validation tools may remain outside default pytest discovery.
