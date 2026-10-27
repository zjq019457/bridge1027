# External asset artifacts

These large artifacts are intentionally separate from the code/runtime bundle and must not be uploaded to GitHub.

| asset | path | SHA256 |
|---|---|---|
| Qwen model/tokenizer snapshot | `/home/yu-bai/Desktop/jiaqizhang/BRIDGEPACK_MODEL_ASSET.tar` | `ae94c46c29ca6a6547e66a336bca8336d4550f1e01af34dfa08b0fc80564763b` |
| six-task LongBench data | `/home/yu-bai/Desktop/jiaqizhang/BRIDGEPACK_DATA_ASSET.tar` | `40e241a1362869910ca05640cee6327cd83a6d361fbc149dedc803ee45924072` |

The model asset tar was created by dereferencing the frozen HF snapshot symlinks. After extraction, verify the extracted directory against `MODEL_TOKENIZER_SHA256SUMS.txt`.
