# Task 5 Production Recommendation

## Instruction 3 Completion (Latency + Memory + F1)
Combined benchmark with memory columns:
- `experiments/task5_production/quant_benchmark_with_memory.csv`

Full-set (`sample_count=3453`) results:
- `Q4_K_M_LORA_full`: F1 `0.3573`, `1922.74 ms/sample`, model size `940.37 MB`
- `Q5_K_M_LORA_full`: F1 `0.4119`, `2646.65 ms/sample`, model size `1072.93 MB`
- `Q8_0_LORA_full`: F1 `0.4476`, `2108.27 ms/sample`, model size `1570.29 MB`

## Memory Bottleneck
Evidence:
- `experiments/task5_production/memory_profile.csv`
- `experiments/task5_production/memory_profile.json`
- `experiments/task5_production/quant_benchmark_with_memory.csv`

Findings:
- Mean activation memory per profiled layer is ~`0.3304 MB`.
- Model weights are GB-scale (`~940 MB` to `~1570 MB` quantized; `~2944 MB` fp16 params).
- Dominant memory bottleneck is **model weights**, not activation memory.

## Concurrency Results (SLA-focused)
Source:
- `experiments/task5_production/concurrency.csv`

Key tuned rows:
- `Q4_K_M_server_tuned, concurrency=1`: p95 `283.74 ms`, throughput `3.96 rps`
- `Q4_K_M_server_tuned, concurrency=2`: p95 `427.01 ms`, throughput `5.65 rps`
- `Q5_K_M_server_tuned, concurrency=1`: p95 `363.74 ms`, throughput `3.05 rps`
- `Q5_K_M_server_tuned, concurrency=2`: p95 `583.88 ms` (SLA miss)

Observations:
- Increasing concurrency improves throughput, but can increase tail latency.
- Q4 tuned config is currently the strongest for strict p95 SLA.

## Instruction 7 Completion (Production Config Recommendation)
Target SLA: `<500 ms` p95 latency.

### Recommended SLA config (meets target)
- Quantization: `Q4_K_M`
- Batch size: `1` (single-request path)
- Concurrency: `2`
- Result: p95 `427.01 ms`, throughput `5.65 rps`

### Quality-first alternative
- Quantization: `Q8_0`
- Use when accuracy is prioritized over strict SLA (`Q8_0` has best F1: `0.4476`)
- For strict `<500 ms` p95 target, Q8 needs additional optimization/tuning.

## Final Takeaway
Task 5 requirements are completed end-to-end:
- Memory profiling, quantization, full benchmark, concurrency testing, bottleneck analysis, and SLA-ready recommendation are all produced and documented.
