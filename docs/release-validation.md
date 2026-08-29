# v1.0.0 release validation

This record separates observed release evidence from documentation claims. It is
completed from a fresh GitHub clone and the configured local Ollama stack before
the `v1.0.0` tag is published.

## Platform coverage

- **Windows / Python 3.12:** local fresh-clone runtime install, readiness, ingestion,
  grounded-answer smoke, live Ollama test, evaluation, and complete quality gate.
- **Ubuntu / Python 3.11 and 3.12:** clean GitHub Actions checkout, pinned dependency
  install, formatting, linting, unit/UI tests, coverage, and dependency validation.
- **Windows / Python 3.11:** the same clean GitHub Actions quality workflow.
- **macOS:** commands are documented but were not executed for this release.

## Commands and observed results

Validation date: **2026-08-29**.

### Fresh clone and default quality gate

Tested from default-branch commit `637698f2b95f4c9e730cddf38064ed5bbe006009`
on Windows NT 10.0.26200 with Python 3.12.6:

```powershell
git clone --depth 1 https://github.com/akrambelajouza/local-agentic-rag-with-ollama.git
Set-Location local-agentic-rag-with-ollama
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.lock
Copy-Item .env.example .env
.\venv\Scripts\python.exe -m local_rag.readiness
.\venv\Scripts\python.exe -m pip install -r requirements-dev.lock
.\venv\Scripts\python.exe scripts\quality.py
```

Observed results:

- Pinned runtime installation and import smoke: **PASS**.
- Pre-Ollama readiness: configuration and dataset passed; Ollama, models, and the
  not-yet-created collection failed with the documented corrective actions.
- Formatting and linting: **PASS**.
- Tests: **62 passed, 1 live test skipped by default**.
- Branch coverage: **86%** (required floor: 85%).
- Dependency consistency: **PASS**.

### Live local stack

The official Windows installer for Ollama `0.33.2` was verified before execution:

- Authenticode status: **Valid**, signer: **Ollama Inc.**
- Installer SHA-256: `5a91c1cf92480e28a84cd99e437219be719df5a50d5fa0fd5fe5b5c4a122f506`
- Published SHA-256: exact match

The configured models were pulled before ingestion:

- `mxbai-embed-large:latest` (`468836162de7`, 669 MB)
- `llama3.2:3b` (`a80c4f17acd5`, 2.0 GB)

Readiness then passed configuration, dataset, Ollama, and model checks while correctly
reporting that the fresh clone had no vector collection. The first ingestion attempt
exposed an incompatible CUDA kernel on this host; the atomic rebuild failed without
publishing a partial index. Ollama was restarted with `OLLAMA_LLM_LIBRARY=cpu` and
`OLLAMA_NO_CLOUD=1`, after which ingestion completed:

- Documents: **21**
- Chunks: **21**
- Failures: **0**
- Duration: **28.94s**

Post-ingestion readiness passed all five checks and reported 21 chunks. The opt-in
live test answered `Who created Python?`, returned a stored python.org citation, and
passed in **13.16s** with warm CPU models.

The release candidate fixed defects discovered by the first live evaluation without
allowing similarity to override a semantic insufficiency decision. The judge now sees
the two strongest excerpts and can perform one genuine rewrite; the positive-label
citation metric is honestly named annotated-source coverage; and proposed unsupported
claims must be copied from the answer and independently confirmed by a second semantic
decision. Number-contradiction and near-match decline regressions cover the safety
boundaries. The final real-model run is committed as a
[human summary](../evaluation/results/v1.0.0.md) and
[machine report](../evaluation/results/v1.0.0.json):

- Status: **PASS**
- Retrieval hit rate: **100%**
- Answer correctness: **100%**
- Unsupported claims: **0**
- Annotated-source coverage: **100%**
- Cases: **6** (four answerable and two out-of-corpus decline cases)
- Duration: **316.57s** on the CPU backend
- Evaluated source revision: `cc7b5cab81a1867432af6f1fb99ccb58df1eaf91`
- Runtime provenance: Ollama `0.33.2`, both full model digests, retrieval limit `4`,
  relevance threshold `0.25`, and generation bound `512` in the machine report

### Final release-candidate quality gate

- Formatting and linting: **PASS**
- Tests: **74 passed, 1 live test skipped by default**
- Branch coverage: **86%** (required floor: 85%)
- Dependency consistency: **PASS**

GitHub Actions repeats this gate on Windows and Ubuntu with Python 3.11 and 3.12.
macOS remains explicitly unverified for v1.0.0.

### Post-merge publication sequence

The release URL is intentionally pending while this candidate is under review. After
the release PR merges and all required checks remain green:

1. Update local `main` and verify the merge commit is the intended release target.
2. Create annotated tag `v1.0.0` at that merge commit and push the tag.
3. Publish GitHub release `v1.0.0 — Portfolio release` from the tag using this
   changelog and validation record as release notes.
4. Verify the release URL, README links, closed issue, clean default branch, and that
   no release PR remains open.

The tag will not be created from the pre-merge branch head; this prevents the release
from omitting the merge commit or diverging from the default branch.
