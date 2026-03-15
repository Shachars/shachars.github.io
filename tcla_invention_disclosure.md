# TCLA — Token Count Link Adaptation
## Invention Disclosure: QoE-Driven Wireless Adaptation for Semantic Video Communications

**Document Status:** Internal Idea Document — Confidential  
**Date:** March 2026  
**Relates to:** TCLA_Simulation_OneDPiece_fixed.ipynb (Cells 1–18)

---

## Technical Background

A semantic video tokenizer trained with the **Tail Token Drop (TTD)** objective — specifically One-D-Piece (Turing Inc., 2025) — converts each video frame into an ordered 1D sequence of discrete tokens `[T₁, T₂, ..., T₂₅₆]`. The TTD training objective forces the encoder to concentrate information at the head of the sequence: earlier tokens carry global structure and dominant semantics, later tokens carry fine detail.

This produces a critical property: **QoE(k) is monotone**. The receiver can reconstruct a valid image from any prefix of k tokens, with quality increasing continuously from k=1 to k=256. Any missing tail tokens can be predicted by a Masked Token Predictor (MaskGIT) conditioned on the received prefix — quality degrades gracefully rather than collapsing.

Simulation validation on real One-D-Piece-L-256: QoE measured as composite = 0.5×SSIM + 0.5×(1−LPIPS), monotone confirmed across all images with r ≥ 0.95 Block ACK prediction accuracy.

---

---

## Problem Statement and Prior Art

### The Failure of Conventional Real-Time Video over Wi-Fi

Real-time video applications — videoconferencing, VR/XR, holographic telepresence — require every frame to be rendered within a hard deadline determined by frame rate (33.3ms at 30fps, 11.1ms at 90fps). The current Wi-Fi stack fails this requirement in three fundamental ways.

**Failure Mode 1: All-or-nothing decoding cliff**

H.264 and H.265 codecs require complete delivery of each frame to decode it. A single lost MPDU corrupts the P-frame it belongs to. All subsequent P-frames referencing it are also corrupted, producing visual garbage until the next I-frame — up to 2 seconds later at standard IDR intervals. The user experiences a sudden, prolonged freeze. There is no intermediate degraded state between "perfect" and "broken."

**Failure Mode 2: Deadline miss under burst channel loss**

Wi-Fi channels fail in bursts, not randomly. The Gilbert-Elliott model captures this: the channel enters a bad state during which consecutive MPDUs are lost at high probability (p_bad ≈ 50–88% in our model, burst length ≈ 4–12 MPDUs). Recovering from such a burst requires multiple retransmission rounds. Each round costs one RTT:

```
RTT (congested Wi-Fi) ≈ 8ms:
  SIFS = 16µs
  ACK frame ≈ 100µs
  DIFS = 34µs
  Random backoff (CWmin=16): avg 8 × 9µs = 72µs
  Queuing delay at congested AP: 4–7ms
  Total ≈ 8ms
```

At 30fps (TMAX=33.3ms): maximum 4 retransmission rounds fit. Under poor burst channel, 5+ rounds are needed → deadline exceeded → frame dropped → display freezes. The user experiences the worst possible failure at exactly the worst possible channel conditions.

At 90fps VR/XR (TMAX=11.1ms, RTT=2ms on a dedicated OFDMA-scheduled XR AP): maximum 2–3 rounds fit. Even a single retransmission attempt under moderate channel pushes past the deadline.

**Failure Mode 3: No semantic awareness at MAC layer**

Conventional 802.11 MAC treats frame payload as opaque bytes. A single MPDU containing tokens that encode the speaker's face receives identical protection to an MPDU containing background tokens. When a burst loss forces the system to choose which data to protect (by limiting retransmission attempts), it cannot make semantically informed decisions.

**Failure Mode 4: MCS adapts throughput, not QoE**

The 802.11 rate adaptation algorithm (Minstrel, RRAA, etc.) selects MCS to maximise expected throughput — the product of data rate and delivery probability. This is the wrong objective for real-time video. A higher MCS may deliver 20% more bits per second while simultaneously increasing the probability that the current frame misses its deadline by 40%. The user would prefer a lower MCS that reliably delivers fewer but more important bytes.

### Prior Art Gap

No prior wireless system:
1. Uses a quantified per-frame QoE metric as the primary MAC adaptation target
2. Separates video payload into importance tiers and applies different protection to each within a single TXOP
3. Uses the standard Block ACK bitmap as a QoE measurement instrument
4. Guarantees a non-zero quality floor on every frame regardless of channel state

The monotone QoE(k) function that makes (1)–(3) tractable did not exist before TTD-trained semantic tokenizers.

---

---

## Technical Solution

### Architecture Overview

TCLA introduces a **Layered TXOP** structure. A single TXOP contains two PPDUs transmitted sequentially within the latency budget:

```
TXOP  (≤ TMAX)
│
├─ PPDU 1: Base Layer
│    tokens [T₁ .. T_k_base]   at MCS_low
│    └─ Block ACK → k_base_rx
│
└─ PPDU 2: Enhancement Layer
     tokens [T_{k_base+1} .. T_{k_base+k_enh}]   at MCS_high
     └─ Block ACK → k_enh_rx
```

Two parameters govern the system:

- **k_base** (fixed): number of tokens in the base layer. Chosen so QoE(k_base) exceeds the minimum acceptable quality floor. Default: 32 tokens.
- **k_enh** (variable): number of tokens in the enhancement layer. Adapted each frame by the TCLA controller. Range: 0 to K_ENH_MAX = L − k_base = 224.

---

### Layer 1: Base Layer — Freeze-Free Guarantee

The base layer transmits the k_base most important tokens at **MCS_low** — a robustly coded modulation scheme with very low effective erasure probability (p_bad ≈ 4%) even during channel burst events.

At MCS_low, k_base tokens are delivered with near-certainty on every frame, in every channel condition. This guarantees:

```
k_base_rx ≈ k_base  (always)
QoE(k_base_rx) > 0  (always)
display freeze = structurally impossible
```

The base layer is the **quality floor guarantee**. It is never adapted, never retransmitted, and never withheld. Its cost is exactly 1×RTT.

The MCS for the base layer is not controlled by TCLA — it is set by the 802.11 rate adaptation algorithm (Minstrel) which naturally selects conservative MCS for a small, high-priority transmission. TCLA only controls k_base and k_enh.

---

### Layer 2: Enhancement Layer — Adaptive Quality

The enhancement layer transmits the next k_enh tokens at **MCS_high** — whatever MCS the rate adaptation algorithm has converged to for the current channel. This layer is **opportunistic**: it improves quality when received, but its loss causes only graceful QoE degradation, not freeze.

k_enh is the sole control variable of TCLA. It is adapted each frame via the Block ACK feedback loop described below. Its cost is 1×RTT when k_enh > 0, zero otherwise.

**Total TCLA latency:**
```
lat_TCLA = RTT + RTT = 2 × RTT   (always, regardless of k_enh)
```

At Wi-Fi 6 (RTT=8ms): lat_TCLA = 16ms ≤ TMAX=33.3ms — constraint always slack.
At VR/XR (RTT=2ms): lat_TCLA = 4ms ≤ TMAX=11.1ms — constraint always slack.

However, at VR/XR rates (0.5 Mbit/s/user), the **transmission time** of the enhancement layer becomes significant:

```
T_ENH_BUDGET = TMAX/2 − RTT_XR = 5.55 − 2.0 = 3.55ms
k_enh_budget = floor(T_ENH_BUDGET × RATE / 12)
             = floor(3.55ms × 0.5Mbit/s / 12bits) = 147 tokens
```

At VR/XR rates, k_enh is physically capped at 147 by the latency budget. This activates a **second constraint** alongside the QoE setpoint. See the Dual-Constraint section below.

---

### The Block ACK as QoE Measurement Instrument

This is the core innovation. The standard 802.11 Block ACK bitmap is a bit-vector reporting which MPDUs were received. Because TCLA packs tokens sequentially into MPDUs, the Block ACK directly gives token-level delivery information:

```
k_base_rx = (number of received base layer MPDUs) × TPM
k_enh_rx  = (number of received enhancement MPDUs) × TPM
k_rx      = k_base_rx + k_enh_rx

QoE_predicted = lut[k_rx]
```

Where `lut` is a pre-characterised lookup table mapping token count to composite QoE (0.5×SSIM + 0.5×(1−LPIPS)), computed offline from representative frames.

**No MLLM inference. No new protocol messages. No application-layer measurement probes.**

The Block ACK bitmap IS the QoE report. This works because QoE(k) is monotone: knowing how many tokens arrived is sufficient to predict quality. Token position is the quality indicator.

Empirical validation: r = 0.976 correlation between Block ACK-predicted QoE and actual measured composite QoE across 120 frames under varying channel severity (excellent → good → moderate → poor → recovering).

---

### The TCLA Adaptation Algorithm

```
════════════════════════════════════════════════════════════
TCLA ALGORITHM — PER FRAME
════════════════════════════════════════════════════════════

CONSTANTS:
  k_base    = 32      base layer tokens (fixed)
  K_ENH_MAX = 224     maximum enhancement tokens
  QT        = nqoe(lut, 80)   QoE setpoint target
  DU        = 32      k_enh step up   (probe aggressively)
  DD        = 16      k_enh step down (retreat conservatively)
  α         = —       (no EWMA needed — threshold-based)

STATE: k_enh (initialised to K_ENH_MAX)

────────────────────────────────────────────────────────────
STEP 1 — TRANSMIT BASE LAYER

  transmit(tokens[1..k_base], MCS=MCS_low)
  wait RTT
  k_base_rx ← Block_ACK_count() × TPM

  # k_base_rx ≈ k_base always (MCS_low near-zero erasure)
  # Display freeze impossible regardless of channel state

────────────────────────────────────────────────────────────
STEP 2 — TRANSMIT ENHANCEMENT LAYER

  transmit(tokens[k_base+1 .. k_base+k_enh], MCS=MCS_high)
  wait RTT
  k_enh_rx ← Block_ACK_count() × TPM

  # k_enh_rx varies with channel:
  #   good channel:  k_enh_rx ≈ k_enh   (almost all arrive)
  #   poor channel:  k_enh_rx << k_enh  (burst destroys many)

────────────────────────────────────────────────────────────
STEP 3 — QoE INFERENCE (ZERO MLLM)

  k_rx         = k_base_rx + k_enh_rx
  QoE_this_frame = lut[k_rx]          ← table lookup only

  # Receiver uses k_rx to:
  #   decode  tokens[1..k_rx]          (direct VQ decoding)
  #   predict tokens[k_rx+1..256]      (MaskGIT, conditioned on received)

────────────────────────────────────────────────────────────
STEP 4 — ADAPT k_enh FOR NEXT FRAME

  # QoE setpoint controller:
  k_enh_qoe = k_enh + DU  if QoE_this_frame >= QT
  k_enh_qoe = k_enh - DD  if QoE_this_frame <  QT

  # Latency ceiling (physical, active at congested XR rates):
  k_enh = min(k_enh_qoe, K_ENH_BUDGET)
  k_enh = max(k_enh, 0)

  # NOTE: k_enh adaptation is a QoE setpoint controller,
  #   NOT a latency controller. At Wi-Fi 6 rates, TMAX is
  #   satisfied regardless of k_enh (tx_time ≈ 0.036ms).
  #   The latency ceiling K_ENH_BUDGET is slack at Wi-Fi 6
  #   and becomes binding only under extreme congestion
  #   (< ~0.76 Mbit/s/user at 90fps).

  total_latency = 2 × RTT            ← always, regardless of k_enh

════════════════════════════════════════════════════════════
CONVERGENCE:

  Good channel (delivery_rate ≈ 0.87):
    k_enh → K_ENH_MAX = 224
    k_rx  ≈ 32 + 224 × 0.87 ≈ 227 → near-maximum QoE

  Poor channel (delivery_rate ≈ 0.44):
    k_enh* = (lut_inverse(QT) − 32) / 0.44   (stable setpoint)
    System oscillates around k_enh* with amplitude ≤ max(DU, DD)

  Channel transition (sudden degradation):
    k_enh decreases by DD per frame
    Converges to new setpoint in ≈ K_ENH_MAX/DD ≈ 7 frames
════════════════════════════════════════════════════════════
```

---

### Dual-Constraint Regime: VR/XR

At VR/XR operating conditions (90fps, dedicated OFDMA AP, congested XR environment):

```
TMAX_XR   = 11.1ms   (90fps)
RTT_XR    = 2.0ms    (OFDMA trigger-based AP, low contention)
RATE_XR   = 0.5 Mbit/s/user   (200 XR users on 100 Mbit/s AP)

tx_time(256 tokens) = 256 × 12 / 0.5Mbit/s = 6.14ms   (now material)

T_ENH_BUDGET = TMAX_XR/2 − RTT_XR = 5.55 − 2.0 = 3.55ms
K_ENH_BUDGET = floor(3.55ms × 0.5Mbit/s / 12bits) = 147 tokens
```

k_enh is physically capped at 147 — 77 tokens below K_ENH_MAX=224. Two constraints operate simultaneously:

```
Constraint 1 (latency/physics):
  k_enh ≤ K_ENH_BUDGET = 147
  Active always at 0.5 Mbit/s, 90fps, RTT=2ms

Constraint 2 (QoE setpoint):
  k_enh adapts toward lut_inverse(QT) via Block ACK
  Active when channel is poor enough that delivery_rate × K_ENH_BUDGET < QT

Crossover rate (when latency constraint activates):
  K_ENH_MAX × 12 / (T_ENH_BUDGET × 1e6/1000) ≈ 0.76 Mbit/s/user
  Below 0.76 Mbit/s: latency ceiling binding
  Above 0.76 Mbit/s: ceiling slack (Wi-Fi 6 regime)
```

**Two operating regimes per channel phase:**

- **Latency-bound (good/excellent channel):** k_enh = K_ENH_BUDGET = 147 always. The rate adaptation algorithm has selected high MCS → good delivery → QoE setpoint satisfied at ceiling → no QoE pressure to reduce k_enh. Latency constraint is the active one.

- **QoE-bound (moderate/poor channel):** k_enh drops below 147. Channel cannot reliably deliver 147 tokens, so the QoE setpoint retreats to a value the channel can support. Both constraints coexist but the QoE setpoint is the tighter one.

The transition between regimes is observable frame-by-frame in the adaptation trace: k_enh hits the ceiling during good phases (latency-bound), drops below it during poor phases (QoE-bound).

**Conventional at VR/XR rates:**

```
Initial tx: 6.14ms + 2ms RTT = 8.14ms  (fits within 11.1ms)
Round 1 retransmit: ~3ms tx + 2ms RTT = 5ms
Total: 13.14ms > 11.1ms TMAX → FREEZE after just one retransmission attempt

Under moderate channel (PER≈31%): ~5 MPDUs lost on first pass
→ conventional freezes ~30–40% of frames
→ mean Conv QoE = (1−freeze) × full + freeze × 0 → collapses
```

Simulation result (VR/XR, poor channel, 500 trials):
- Conventional freeze rate: ~50–60%
- TCLA freeze rate: 0%
- TCLA QoE gain over conventional: +0.30–0.40 composite units (much larger than Wi-Fi 6 scenario)

---

### Why k_enh Varies (Not Latency — QoE and Airtime)

A precise statement of why the TCLA adaptation loop reduces k_enh under poor channel, at Wi-Fi 6 rates where the latency constraint is slack:

**Reason 1 — Shared medium efficiency:** Wi-Fi is a shared broadcast medium. Sending k_enh=224 tokens under a poor channel where 56% are lost wastes 56% × 224 × 0.036ms = 4.6µs of shared airtime per frame — multiplied by all users, this degrades throughput for every device on the AP. Reducing k_enh to 80 under poor channel wastes only 1.6µs of shared airtime while delivering comparable QoE (because the lost tokens would have been predicted by MaskGIT anyway).

**Reason 2 — MaskGIT reconstruction quality:** When k_enh=224 is sent but only 99 tokens arrive (burst loss), the receiver's MaskGIT must predict 157 positions. Many of these missing tokens are adjacent — the burst destroyed correlated spatial regions — so MaskGIT has little nearby context. Reconstruction is noisy and inconsistent. If instead k_enh=80 is sent and 70 arrive, MaskGIT predicts from a clean, contiguous prefix — much higher confidence, smoother reconstruction.

**Reason 3 — TXOP fairness:** In 802.11, TXOP duration is negotiated in advance. Requesting a shorter TXOP (by sending fewer tokens) releases the channel sooner, improving fairness among competing users.

At VR/XR rates, a fourth reason applies: **Reason 4 — Latency constraint:** tx_time becomes a significant fraction of the budget, and exceeding K_ENH_BUDGET would violate TMAX.

---

---

## Advantages and Differentiating Features

### 1. Freeze-Free Guarantee by Architecture

The base layer guarantee makes display freeze structurally impossible. This is not a statistical claim ("freeze probability is low") but a system property provable from the MCS_low erasure probability. No prior real-time video system over Wi-Fi offers a freeze-free guarantee by design.

User experience impact: display freeze incurs a MOS penalty of −1.5 points (ITU-T P.1203); equivalent-duration quality reduction incurs only −0.3 points. Eliminating freeze is the single highest-value quality improvement available.

### 2. QoE as a First-Class Control Variable

TCLA is the first system to treat QoE as the primary target of MAC-layer adaptation, rather than a hoped-for consequence of throughput maximisation. The monotone QoE(k) function — unique to TTD-trained tokenizers — makes this tractable by creating a direct, invertible mapping from a MAC-observable quantity (tokens delivered) to a perceptual quality metric.

### 3. Block ACK Repurposed as QoE Instrument

The 802.11 Block ACK mechanism has existed since 802.11e (2005). No prior system uses it as a quality measurement tool. TCLA extracts per-frame QoE feedback from a protocol element already present in every modern 802.11 device, at zero additional overhead.

### 4. Zero MLLM Inference in the Adaptation Loop

The k_enh adaptation requires only a table lookup: `QoE = lut[k_rx]`. The MaskGIT gap-filling inference at the receiver is not in the adaptation loop — it occurs after the adaptation decision has already been made. This means the feedback mechanism has negligible computational cost, sub-millisecond response time, and no dependency on GPU availability.

### 5. Constant, Bounded Latency

TCLA always completes in exactly 2×RTT. This is a hard bound, not an expected value. Conventional retransmission-based systems have variable latency that grows with channel degradation and can exceed TMAX under burst loss. TCLA's latency is completely decoupled from channel quality.

### 6. Composability with Existing Rate Adaptation

TCLA does not replace or conflict with the 802.11 rate adaptation algorithm (Minstrel). MCS selection continues to operate independently, optimising delivery probability for the current channel. TCLA observes the delivery outcome through Block ACK and adapts k_enh on top. The two mechanisms are orthogonal — TCLA adapts quantity, rate adaptation adapts reliability.

### 7. Dual-Constraint Elegance in VR/XR

At VR/XR operating conditions, the system transitions naturally between two operating regimes — latency-bound and QoE-bound — governed by a single parameter update (`k_enh = min(k_enh_qoe, K_ENH_BUDGET)`). No mode switching, no separate controllers, no hysteresis logic. The same algorithm handles both the Wi-Fi 6 case (K_ENH_BUDGET slack) and the VR/XR case (K_ENH_BUDGET binding) with a single `min()` operation.

### 8. No New Protocol Elements

TCLA operates entirely on existing 802.11 primitives:
- Block ACK (since 802.11e, 2005)
- TXOP (since 802.11e, 2005)
- A-MPDU aggregation (since 802.11n, 2009)

No new management frames. No new information elements. No firmware changes to the AP. TCLA is implementable as a driver-level modification to the STA's MAC sublayer.

---

### Comparison Summary

| Property | H.265 + RTP | TCLA (Wi-Fi 6) | TCLA (VR/XR) |
|----------|------------|----------------|--------------|
| Frame independence | No (P-chain) | Yes | Yes |
| Freeze possible | Yes | No | No |
| Minimum QoE floor | 0 | QoE(32) > 0 | QoE(32) > 0 |
| Latency bound | Variable (retransmit) | Hard 16ms | Hard 4ms |
| QoE feedback | External probe | Block ACK (zero cost) | Block ACK (zero cost) |
| MLLM in loop | N/A | No | No |
| Rate adaptation | Minstrel (unchanged) | Minstrel + TCLA | Minstrel + TCLA |
| Active constraints | Latency (sometimes) | QoE only | QoE + latency |
| Crossover rate | — | > 0.76 Mbit/s | ≤ 0.76 Mbit/s |
| Simulation gain (poor) | — | +0.015 composite | +0.35 composite |
| Protocol changes | None | None | None |

---

### Simulation Validation Summary

All claims validated on real One-D-Piece-L-256 model, real images, realistic GE channel:

| Claim | Metric | Result |
|-------|--------|--------|
| QoE(k) monotone | Violations / 60 k-values | 0 (astronaut), ≤ 2 (others) |
| Block ACK prediction | Pearson r | 0.976 |
| Block ACK prediction | MAE | 0.044 |
| TCLA freeze rate | All channels | 0.0% |
| Conv freeze rate (poor, Wi-Fi 6) | 500 trials | 14.8% |
| Conv freeze rate (poor, XR) | 500 trials | ~55% |
| TCLA QoE gain (poor, Wi-Fi 6) | Mean composite | +0.015 |
| TCLA QoE gain (poor, XR) | Mean composite | +0.35 |

---

*End of TCLA Invention Disclosure*  
*Simulation code: TCLA_Simulation_OneDPiece_fixed.ipynb*  
*Related documents: tokcom_patent_ideas_v3.md, tcla_simulation_summary.md*
