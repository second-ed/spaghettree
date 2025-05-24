# Setup

## Install uv
To set up a virtual environment to run this code firstly ensure that `uv` is install with:

### Linux / MacOS:
```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows:
```shell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Setup environment

Traverse to the root directory of the codebase and run the following command depending on the usecase:

#### To recreate the research in the paper:
```shell
uv sync --all-groups
```

#### To use as a standalone tool on a personal library:
```shell
uv sync
```

#### To install the development dependencies:
```shell
uv sync --group dev
```

## Run the code

### To recreate the research in the paper:
Run the following command:
- Note that this can take a long time due to the size of the search space for the larger repositories.
```shell
uv run -m spaghettree
```

### To run the code on a different package:
Using all three of the optimisation methods: 
- `--use-hc`: use the hill climbing method
- `--use-sa`: use the simulated annealing method
- `--use-gen`: use the genetic algorithm method

```shell
uv run -m spaghettree --process "path/to/your/package" --use-hc --use-sa --use-gen 
```