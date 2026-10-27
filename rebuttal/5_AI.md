**Operating regime.** "Tight" means tight relative to the source (1,024 ≈ 13%) yet large enough for evidence pairs to compete. At 256 tokens one repaired pair nearly fills the budget; as pairs can compete, BridgePack shows the largest 256→1,024 gain (Fig. 2) and the best macro at 1,024.

**Robustness.** Mechanism claims rest on paired instance-level tests with Holm correction (Table 3): joint selection beats independent and MMR selection significantly.

**Related work.** We will cite CPC and CompAct. Both rely on trained encoders/compressors, whereas BridgePack is training-free, extractive, pairs query/support evidence explicitly and enforces an exact downstream-token budget.

**Compression baselines.** Official LongLLMLingua (Llama-2-7B compressor) under the same exact 1,024-token budget, wrapper and scorer trails BridgePack by 6.81 [4.65, 9.04] on Qwen and 6.31 [4.12, 8.59] on Llama-3. RECOMP requires trained compressors.

**Repair.** Repair and no-repair variants share the same selector and exact ceiling; repair restores qualifiers and answer phrases cut by fixed boundaries. We will report packed lengths.

**Selector specification and minor issues.** We will move the full algorithm (pool sizes, pair admission, tie-breaking) into the main text, define common-success inputs and BridgePack vs. Lexical-BridgePack, and state the paired test and Holm family in Table 3.
