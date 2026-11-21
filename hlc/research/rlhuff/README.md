# rlhuff — (run class, value class) + raw mantissas + Huffman by the previous symbol's group, against llrice RLGR

`dump.cpp` (built inside an llrice `lane-rlgr` checkout with a recording encoder in `Coders<rlgrf::PageBuf>`): the values
every llrice RLGR coder receives (CODER 3: g, r, b) + the llrice file size. `concept.cpp dir 3` [-DHALF=1]: the model
coder on these streams: token = (zero run before a value, the value), symbol = (bit width of the run, value class),
raw mantissas; value class = bit width, or with HALF=1 bit width + the next bit (half octaves). Semi-static Huffman per
image per (stream, context), header estimate 8 + 14 bits per used symbol.

2026-11-16, HALF=1, context = prev value class x prev run 0/1/2+ (48):
| set | llrice CODER 3 (RLGR) | model | |
|---|---|---|---|
| 12 full screens | 1.6013 | 1.2553 | −21.6% (RLK v9 1.2561) |
| 12 crops test | 2.0156 | 1.5830 (best 1.5643 with 4 contexts: header) | −21.5% |
| 16 CLIC | 8.4151 | 8.4301 | +0.2% |

`mant.cpp dir 3`: what is left in the raw mantissas (half octaves): the empirical entropy of the value mantissa given
the class and of the run mantissa given the run class, per image and pooled over the set.
| set | value mantissa raw / entropy pooled | run mantissa raw / entropy pooled |
|---|---|---|
| 16 CLIC | 1.6124 / 1.5697 bpp | 0.2537 / 0.2325 bpp |
| 12 full screens | 0.3266 / 0.3218 bpp | 0.1577 / 0.1449 bpp |
Most of it is in the first mantissa bit; `-DHALF=2` (quarter octaves, 2 bits in the symbol), 48 contexts:
16 CLIC 8.3855 (−0.4% to llrice), 12 screens 1.2553 (=; the header grows), crops 1.5961 (4 contexts: 1.5642).

`jpeg.cpp dir 3`: "pure JPEG" tokens on the same streams — symbol (run 0..15, size = bw(v)), ZRL per 16 zeros, raw size
bits; one table or 11 contexts of the previous symbol; semi-static per image or pooled over the set (optimistic static).
| set | llrice | JPEG 1 table semi / pooled | JPEG + prev ctx semi / pooled | RLH v2 (static, real) |
|---|---|---|---|---|
| 12 full screens | 1.6013 | 1.5619 / 1.5994 | 1.3824 / 1.4167 | 1.2994 |
| crops test | 2.0156 | 1.8759 / 1.9594 | 1.7074 / 1.7293 | 1.5979 |
| 16 CLIC | 8.4151 | 8.9501 / 9.3223 | 8.6156 / 8.7932 | (98 CLIC: 8.7257 vs llrice 8.7583) |

Richer contexts (HALF=1, semi-static; payload / with header): 12 full screens — prev k x run 0/1/2+ (48) 1.2386 / 1.2553,
prev k x run octave (150) 1.2226 / 1.2514, prev full class x run3 (120) 1.2290 / 1.2546, two previous k-groups x run (50)
1.2291 / 1.2486; 16 CLIC: 8.4160 / 8.4301, 8.4125 / 8.4357, 8.3939 / 8.4160, 8.3720 / 8.3952. Payload gains 1–1.5%,
the per-image header eats most of it; matters for static tables.

Context = the llrice RLGR state at the token start (Vladimir's idea: k = run window, kr = Rice parameter, simulated
causally on the same stream), HALF=1, semi-static; payload / with header:
| context | 12 full screens | crops test | 16 CLIC |
|---|---|---|---|
| prev k x run 0/1/2+ (48) | 1.2386 / 1.2553 | 1.4967 / 1.5830 | 8.4160 / 8.4301 |
| RLGR k x kr (256) | 1.2244 / 1.2611 | 1.4883 / 1.6317 | 8.2983 / 8.3120 (−1.2% to llrice) |
| k x kr (8x8) x prev k-group (320) | **1.2045** / 1.2590 | 1.4606 / 1.6636 | **8.2654 / 8.2955 (−1.4%)** |
The RLGR state is a smoothed long-history estimate (like the A/N counters of JPEG-LS): −2.8% payload on screens and
−1.8% on photos over the previous-symbol context; for static tables the payload numbers are what counts.

`tune.cpp full clic16 crops [params...]`: coordinate descent over the RLGR-state rule used only as the context
(k x kr buckets x prev k-group), cost = pooled entropy per set (optimistic static: trained on the set itself, no header,
no Huffman loss); crops = holdout. Log: `tune.log`. Result UP=6 DOWN=3 UPD=6 DOWND=1 DR=1 (slower decay = longer
memory), buckets of 1/2 bit, 12x12:
| params (UP,DOWN,UPD,DOWND,DR,KG,QK,QR,NK,NR) | ctx | 12 full | 16 CLIC | crops (holdout) |
|---|---|---|---|---|
| llrice RLGR rule, 1-bit buckets 8x8 | 321 | 1.2309 | 8.2774 | 1.5231 |
| tuned, 1/2-bit 12x12 | 721 | 1.2258 | 8.2137 | 1.5123 |
| tuned, 1-bit 6x6 | 181 | 1.2353 | 8.2522 | 1.5318 |
| tuned, 2-bit 4x4 | 81 | 1.2413 | 8.3375 | 1.5359 |
| tuned, 2-bit 3x3 | 46 | 1.2518 | 8.3435 | 1.5547 |
| llrice rule, 2-bit 4x4 | 81 | 1.2446 | 8.3778 | 1.5436 |
llrice RLGR itself: 1.6013 / 8.4151 / 2.0156.

`rlgrtune.cpp` (+ `_eval`): the llrice RLGR constants tuned on RLGR itself (exact bit model = llrice engine 1 file size
to 4 decimals). Params UP,DOWN,UPD,DOWND,DR,KG (1/8 bit):
| rule | 12 full | 16 CLIC | crops |
|---|---|---|---|
| llrice (tuned by Vladimir on CLIC) 5,9,2,3,2 | 1.6013 | 8.4150 | 2.0154 |
| screens-optimal 3,6,8,8,2 | **1.4363 (−10.3%)** | 8.6103 | **1.7852 (−11.4%)** |
| photos-optimal 4,9,2,2,2 | 1.5994 | 8.4100 | 2.0091 |
| both 3,4,4,1,2 | 1.4665 | 8.5056 | 1.8296 |
| step 1/4 bit 4,8,2,2,2 | 1.5983 | 8.4153 | 2.0115 |
| step 1/2 bit 4,8,4,4,4 | 1.5390 | 8.5691 (+1.8%) | 1.9252 |
| step 1 bit 8,8,8,8,8 | 1.5446 | 9.1808 (+9.1%) | 1.9403 |
Transferred into the Huffman context (k x kr 6x6 1-bit / 12x12 1/2-bit x prev k-group, pooled entropy):
llrice rule 1.2355 / 8.2787 / 1.5340; screens rule 1.2320 / 8.2767 / 1.5295 (12x12: 1.2218 / 8.2368 / 1.5105);
photos rule 1.2351 / 8.2802; context-tuned rule 1.2353 / 8.2522 / 1.5318. The context barely depends on the rule (±0.3%).

Bucket shapes for k / kr (screens RLGR rule 3,6,8,8,2; QK/QR > 0 uniform step, < 0 exponential bw(v/|Q|), 0 quantile =
equal token counts over full+CLIC):
| buckets | ctx | 12 full | 16 CLIC | crops |
|---|---|---|---|---|
| uniform 1 bit 6x6 | 181 | **1.2320** | 8.2767 | **1.5295** |
| exponential from 1/2 bit 6x6 | 181 | 1.2375 | 8.3037 | 1.5362 |
| exponential from 1 bit 6x6 | 181 | 1.2401 | 8.3142 | 1.5392 |
| quantile 6x6 | 181 | 1.2528 | **8.2704** | 1.5435 |
| uniform 1 bit 4x4 | 81 | 1.2473 | 8.3155 | 1.5524 |
| exponential from 1 bit 4x4 | 81 | **1.2419** | 8.3144 | **1.5454** |
| quantile 4x4 | 81 | 1.2617 | 8.3140 | 1.5535 |
| quantile 12x12 | 721 | 1.2352 | 8.2347 | 1.5192 |
| uniform 1/2 bit 12x12 | 721 | 1.2218 | 8.2368 | 1.5105 |
k and kr are already logarithms (window 2^k, Rice 2^kr): exponential buckets on top are log-of-log; they help only with
very few buckets (4x4).

`ema.cpp`: linear filters as the context instead of the RLGR state: per token EMA of bw(value) and of bw(run), weight
2^-s, 1-bit buckets 6x6 x prev k-group (181 ctx), pooled entropy. Best: value 1/8, run 1/4 — 1.2306 / 8.2994 / 1.5263
(RLGR state, screens rule 6x6: 1.2320 / 8.2767 / 1.5295). value 1/4: 1.2316 / 8.3217; value 1/16: 1.2332 / 8.3174;
run 1/16..1/32 worse. Full grid: `ema.log`.

`fsm.cpp`: the EMA context as a finite automaton — internal precision P units per bit (rounded EMA, value 1/8, run 1/4),
buckets 1 bit 6x6 x prev k-group: P=16: 1.2297 / 8.2902 / 1.5267; P=8 (48 states per filter): 1.2293 / 8.2812 / 1.5263;
P=4: 1.2354 / 8.3285; P=2: 1.2431 / 8.4650; P=1 (state = bucket, 6 per filter): 1.2629 / 8.5236 / 1.5657.
Two independent automata of 48 states (value, run) with a transition table per token class lose nothing.

Qwen task TASK_v5_rlh_fsm (Drive claude-phone/2026-11-16_rlh_fsm/): token files from `tok.cpp` (crops train/test,
2+2 CLIC photos, <= 500k tokens per stream), `fsm_eval.py` = joint train tables, test cross-entropy bits/token.
Baselines: one table 6.3576 / 4.2512; prev symbol 5.5533 / 3.9466; EMA automata 48+48 (181 ctx) 5.2723 / 3.7735.

## Qwen TASK v5 automaton in the real (semi-static, per image) coder — 2026-11-17
`qwen_fsm_v5.h` = C++ port of Qwen's `ctx_fsm.py` (decision tree over 5 causal registers + previous token, 23 contexts;
port checked token-for-token against the Python on screens_test and photos_test). `concept.cpp` modes 13 (Qwen) and
14 (our EMA 48+48), `-DHALF=1`. bpp incl. mantissas and header estimate:

| set | llrice | k x kr(64) x prev | EMA 48+48 | Qwen FSM v5 |
|---|---|---|---|---|
| 12 full screens* | 1.6013 | 1.2590 | 1.2432 | **1.2367** |
| 12 crops* (Qwen's train images) | 2.0156 | 1.6636 | 1.6153 | **1.5579** |
| 12 test crops (clean) | 1.5504 | 1.4495 | 1.4198 | **1.3815** |
| 12 CLIC (clean, without the first 4) | 8.4645 | 8.3395 | 8.3447 | **8.3000** |

\* Qwen's screens_train are crops of these 12 images, so these rows are not clean. On the clean sets Qwen's automaton
wins by 2.7% (crops) and 0.5% (CLIC) against EMA 48+48. On screens most of the gain is the header (fewer contexts);
the payload alone is about equal to EMA; on CLIC the payload is better too.
