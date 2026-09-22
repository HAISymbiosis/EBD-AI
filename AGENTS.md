# Agent instructions

Read [LLM/DEVELOPMENT.md](LLM/DEVELOPMENT.md) before changing this repository.
It contains the project conventions, code map, and validation requirements.

Before changing genetic fitness evaluation or firing kernels, also read
[the speedup record](LLM/performance/SPEED_UP.md) and
[the decision review](LLM/performance/SPEED_UP_REVIEW.md). Preserve bit-for-bit objective parity.

Before delegating, read [the delegation guidance](LLM/DELEGATION.md).

For Demo Studio, keep the original `Demos/` notebooks and scripts as the source
of truth. Implement GUI integration under `ebdai`, without modifying the rebased
`ex_fuzzy` library. Follow the demo validation notes in `LLM/DEVELOPMENT.md`.
