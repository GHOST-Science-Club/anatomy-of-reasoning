# Anatomy of Reasoning

## Setup

1. Install [Git](https://git-scm.com/downloads) and [uv](https://docs.astral.sh/uv/getting-started/installation/).

2. Clone the repo (the `--recurse-submodules` flag also downloads ThinkARM):

   ```bash
   git clone --recurse-submodules <repo-url>
   cd anatomy-of-reasoning
   ```

   Already cloned without it? Run `git submodule update --init --recursive`.

3. Install the Python packages:

   ```bash
   uv sync
   ```
