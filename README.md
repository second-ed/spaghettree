# Spaghettree
Software complexity directly affects the maintainability of modern codebases.
Most of the software lifecycle is spent maintaining production systems. High complexity leads to harder maintenance, slower feature delivery, and longer onboarding for new engineers.

### What this tool does

This is a prototype tool for simplifying structural complexity of a codebase. It works by optimising the call-graph and is intended for integration as a CI/CD pipeline stage.

### Why bother?
This tool hopes to:
- Help manage and limit complexity growth during development.

- Complements traditional linters and formatters by addressing architectural issues.

- And also:
    - reduce technical debt
    - lower maintenance costs
    - speed up engineer onboarding


### Notes
As this is a prototype and not ready for production use, the defaults are set to just report the current structures directed weighted modularity and the current call tree for the repo as it stands.

# Args
| Argument           | Type                  | Required | Default | Description                                          |
| ------------------ | --------------------- | -------- | ------- | ---------------------------------------------------- |
| positional src_root      | `str`                 | ✅       |  | Path to the root of the repository to scan           |
| `--new-root`    | `str`                 | ❌       | `''` | Optional new root path for output (default: empty, meaning same as src_root if optimisation is enabled).        |
| `--optimise-src-code`  | Flag (no value)       | ❌       |  | Enable optimisation of the source code. |



# How it works
- All `py` files in the given directory are read in as strings
- Each of those strings are parsed into `libcst` CST objects
    - This is so comments and other things are retained otherwise useful info would be lost
- A list of locations of each of the entities (name, original module, line no) is collected and stored.
- The CSTs are transformed into custom objects:
    - ModuleCST
    - ClassCST
    - FuncCST
    - GlobalCST
    - ImportCST
- A ClassCST can have `0-n` FuncCST methods on it, and each FuncCST has a list of fully qualified calls that the function calls. 
= With these structures we can create a call-graph. e.g. `ClassA.method_a -> some_func`
- To ensure any refactoring is possible, a call from a classes methods is counted as a call to that class (so you don't split classes into separate methods).
- From the call-graph, the non-native calls are filtered out, that means that only entities defined in the repo are considered for moving.
- An adjacency matrix is created from the call graph where the x and y axes are the entities and then the co-ordinates are counts of calls from x to y
- Each of the entities is considered as a single module at first, so that means you could have a single constant in a file by itself.
- Then each pair-wise combination is considered to be merged
    - If the merge of the entities would result in a gain of the repo's directed weighted modularity then its added as a possible merge to consider.
- All the possible merges are sorted by the largest gain it'd bring to the overall system, then each non-overlapping merge is applied 
     - e.g. merge `[(mod_a, mod_b), (mod_c, mod_d)]` 
     - merge for `(mod_b, mod_c)` is not considered as the mod_c and mod_d merge would result in a higher directed weighted modularity.
- This is repeated until there are no more valid merges
- Once this is done some extra modification is done, for example if you were writing a library of validators that didn't call eachother but all sat in the same module, then they are combined.
- When writing the entities to their new files, the imports are updated, and the location of each of the entities are kept as close as they can be to where they were before.


```python
# some_original_mod

T = TypeVar("T")

class SomeClass:
    def method(self, item: T) -> T:
        return item

class SomeOtherClass:
    def method(self, item: T) -> T:
        return item
    
SomeType = SomeClass | SomeOtherClass
```


- This is to ensure that for an example like above the result is still valid, an initial idea was to always write globals, classes, funcs, but that would result in `some_broken_mod`

```python
# some_broken_mod

T = TypeVar("T")
SomeType = SomeClass | SomeOtherClass # BROKEN as the classes aren't defined yet

class SomeClass:
    def method(self, item: T) -> T:
        return item

class SomeOtherClass:
    def method(self, item: T) -> T:
        return item
    
```

- Lastly when the entities are all written to their new module location, `ruff` is called on the files to fix any formatting, because of how ruff is set up, it means it would respect the users own `ruff.toml` so would include or exclude rules they were interested in.

# Repo map
```
├── .github
│   └── workflows
│       ├── ci_tests.yaml
│       └── publish.yaml
├── src
│   └── spaghettree
│       ├── adapters
│       │   ├── __init__.py
│       │   └── io_wrapper.py
│       ├── domain
│       │   ├── __init__.py
│       │   ├── adj_mat.py
│       │   ├── entities.py
│       │   ├── optimisation.py
│       │   ├── parsing.py
│       │   ├── processing.py
│       │   └── visitors.py
│       ├── logger
│       │   └── __init__.py
│       ├── __init__.py
│       └── __main__.py
├── tests
│   ├── adapters
│   │   ├── __init__.py
│   │   └── test_adapter_apis.py
│   ├── domain
│   │   ├── __init__.py
│   │   ├── test_entities.py
│   │   ├── test_optimisation.py
│   │   └── test_processing.py
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_main.py
│   └── test_result.py
├── .pre-commit-config.yaml
├── README.md
├── pyproject.toml
├── ruff.toml
└── uv.lock
::
```