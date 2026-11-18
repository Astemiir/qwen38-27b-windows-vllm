# Messages for the upstream authors

One file per addressee, self-contained and ready to post. Each starts with an HTML comment giving the
place and a suggested title; the body below it is the message.

| file                                   | addressee            | where                                      | what it asks for                                                                           |
| -------------------------------------- | -------------------- | ------------------------------------------ | ------------------------------------------------------------------------------------------ |
| `01-jorge-vllm-quantpack.md`           | Jorge Ramírez Espejo | new issue in `jramirez-uab/vllm-quantpack` | the encoding fix that re-enables his own int4 fast kernel; offer of a PR                   |
| `02-vllm-project-triton-fp8-int4.md`   | vLLM maintainers     | new issue (two sections)                   | one guard predicate too wide for `fp8_per_token_head`; a question about int4 + speculation |
| `03-aivrar-windows-build.md`           | Chris (aivrar)       | comment in #20                             | the two doc lines that still say FlashInfer has no Windows wheel; what now exists          |
| `04-flashinfer-ai-windows-jit.md`      | FlashInfer upstream  | new issue                                  | five mechanical Windows blockers, with the artifact-suffix and `spec.build()` notes        |
| `05-systempanic-flashinfer-windows.md` | SystemPanic          | comment in #14                             | the delta from `v0.7.0.post1` to `v0.7.2.post1`, offered as a patch                        |

Order matters only in one respect: 01 is the message that changes what a Windows user downloads, so send
it first. Everything else is independent.

Before posting, replace every `https://github.com/Astemiir/qwen38-27b-windows-vllm` with the published repository URL, and attach nothing — the
messages point at `patches/`, `PERFORMANCE.md` and `bundle/` inside it.

`../UPSTREAM-FIXES.md` is the audit view: the same fixes with the full context and the code around each
line. These files are the posting view: shorter, addressed to one person, no cross-references to the other
authors.
