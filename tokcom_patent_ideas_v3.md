# TokCom Patent Ideas — Three Invention Disclosures
## Token Communications for Wi-Fi / 6G Networks

**Document Status:** Internal Idea Document — Confidential  
**Date:** March 2026  
**Portfolio Scope:** Three patentable inventions arising from the Token Communications (TokCom) framework

---

## Common Technical Foundation

All three inventions share the same enabling substrate. A semantic tokenizer — specifically **One-D-Piece** (Turing Inc., 2025), trained with the **Tail Token Drop (TTD)** objective — converts a raw image or video frame into an ordered 1D sequence of discrete integer indices `[T₁, T₂, ..., T_L]`. Each index points into a learned codebook of size 4096 (12 bits per token).

Because of TTD training, the sequence has a **strict, monotone semantic importance ordering**: earlier tokens encode global structure and dominant semantics; later tokens encode fine detail. The detokenizer reconstructs a valid image from any prefix `[T₁, ..., Tₖ]` for any k. Quality increases monotonically with k. This property — QoE(k) is monotone — is the enabling condition for all three inventions.

**Empirical validation:** The simulation (TCLA_Simulation_OneDPiece_v5.ipynb) demonstrates this directly using the real One-D-Piece-L-256 model on real images. QoE is measured as a composite perceptual metric `0.5×SSIM + 0.5×(1−LPIPS)`, bounded [0,1]. The monotone curve holds across all tested images.

At the receiver, missing tokens (tail positions beyond k_rx) can be predicted by a **Masked Token Predictor (MaskGIT)** using surrounding context. Prediction quality is highest for tail tokens (strongly conditioned on received head tokens) and lowest for head tokens. This asymmetry — tail tokens are easily predicted, head tokens are not — underpins the layered protection strategy in Patent 1.

---

---

# Patent 1: TCLA — QoE-Driven Wireless Link Adaptation with Layered TXOP

## Background and Prior Art

Modern Wi-Fi link adaptation selects a Modulation and Coding Scheme (MCS) based on channel quality, maximising expected throughput. Representative algorithms: Minstrel (Linux), RRAA, CARA. The objective is throughput — not QoE.

This causes a fundamental failure for real-time applications (video conferencing, XR, holographic display):

**Failure Mode 1 — All-or-nothing quality cliff:** Conventional codecs (H.265, H.264) require complete frame delivery. A single lost packet corrupts a P-frame. Every subsequent P-frame referencing it is also corrupted — up to 2 seconds of visual garbage until the next I-frame (IDR). This is the P-frame error propagation problem.

**Failure Mode 2 — Deadline miss = freeze:** Under burst channel loss, retransmitting lost MPDUs consumes the frame period budget. Each retransmission round costs one RTT (≈8ms in a congested Wi-Fi medium: SIFS + ACK + DIFS + backoff + queuing). With TMAX=33.3ms (30fps), only 4 rounds fit. Under poor channel (burst length ≈12 MPDUs), 5+ rounds are needed → deadline exceeded → frame dropped → display freezes.

**Failure Mode 3 — No semantic awareness:** The MAC layer has no visibility into which parts of the frame are perceptually important. A speaker's face and a background wall receive equal protection. The system cannot make intelligent trade-offs.

**Prior art gap:** No prior wireless system uses a quantified, per-frame application-layer quality metric as the primary MAC-layer adaptation target. QoE is a hoped-for consequence of physical optimisation, never a directly controlled variable. The monotone QoE(k) function — which makes direct control possible — did not exist before TTD-trained tokenizers.

---

## Technical Solution

TCLA introduces a **Layered TXOP** structure where the frame's token sequence is split into two layers transmitted in a single TXOP:

```
TXOP (≤ TMAX = 33.3ms)
│
├── PPDU 1: Base Layer    k_base=32 tokens   MCS_low  (p_bad≈4%)   → Cost: 1×RTT = 8ms
│   └── Block ACK
│
└── PPDU 2: Enhancement   k_enh tokens       MCS_high (p_bad≈50%)  → Cost: 1×RTT = 8ms
    └── Block ACK
    
Total TCLA latency: 16ms — always ≤ TMAX. No retransmissions.
```

### Layer 1: Base Layer — Freeze-Free Guarantee

**k_base=32 tokens** are transmitted at **MCS_low** — a heavily coded, robust modulation scheme with near-zero effective erasure probability (p_bad≈4%) even during channel burst events. This guarantees:
- `k_base_rx ≈ 32` tokens always arrive
- `QoE(32) > 0` always — the receiver always has enough tokens to produce a recognisable reconstruction
- **Display freeze is structurally impossible** as long as cross-modal context (class label, audio cue) is available for the k=0 generative fallback

The base layer size k_base is chosen so that `QoE(k_base)` exceeds a minimum acceptable quality floor (e.g., 0.25 composite, corresponding to a recognisable but blurry image).

### Layer 2: Enhancement Layer — Adaptive Quality

**k_enh tokens** (variable, 0 ≤ k_enh ≤ K_ENH_MAX = L − k_base) are transmitted at **MCS_high** — standard coding, subject to full channel erasure. The enhancement layer is **opportunistic**: it improves quality when received, but its loss causes only graceful QoE reduction, never freeze.

**k_enh is the primary control variable.** It is adapted each frame based on the Block ACK outcome from the previous frame — no MLLM inference needed.

### The Block ACK as QoE Instrument

This is the core innovation of TCLA. The standard 802.11 Block ACK bitmap reports which MPDUs were received. Because tokens are packed sequentially into MPDUs, this directly gives:

```
k_base_rx = sum(received base layer MPDUs) × TPM
k_enh_rx  = sum(received enhancement MPDUs) × TPM
k_rx      = k_base_rx + k_enh_rx
QoE_predicted = lut[k_rx]   ← lookup from pre-characterised QoE(k) table
```

No MLLM inference. No round-trip quality measurement. The Block ACK bitmap **is** the QoE report. Token position serves as the quality indicator because QoE(k) is monotone.

**Empirical validation:** The simulation shows r ≥ 0.95 correlation between Block ACK-predicted composite QoE and actual measured composite QoE across 120 video frames under varying channel conditions.

### The k_enh Adaptation Loop

```
After each frame Block ACK:
  predicted_QoE = lut[k_base_rx + k_enh_rx]
  
  if predicted_QoE ≥ QT:   k_enh += DU   (channel good → probe for more quality)
  if predicted_QoE  < QT:   k_enh -= DD   (channel bad  → reduce enhancement tokens)
  
  where DU > DD (asymmetric: increase aggressively, decrease conservatively)
```

This is analogous to TCP congestion control — but instead of adjusting a window size to maximise throughput, TCLA adjusts enhancement token count to track a QoE target.

### Comparison with Conventional (H.265 + RTP)

| Property | Conventional (H.265+RTP) | TCLA Layered TXOP |
|----------|--------------------------|-------------------|
| Frame independence | No (P-frame chain) | Yes (every frame) |
| Error propagation | Yes (up to IDR ≈ 2s) | Never |
| Quality on loss | Binary cliff → freeze | Graceful degradation |
| Minimum quality floor | Zero (freeze/black) | Non-zero (base layer) |
| Retransmission needed | Yes (FIR requests) | Never |
| Latency bound | Violated on burst loss | Hard 16ms guarantee |
| MAC semantic awareness | None | Token position = importance |
| QoE feedback mechanism | External measurement | Block ACK bitmap |

**Simulation result (poor channel, 500 trials):**
- Conventional: freeze rate ≈15–30% → mean QoE collapses to (1−freeze)×full + freeze×0
- TCLA: freeze rate = 0.0% → mean QoE consistently higher at moderate/poor channel

---

## Distinguishing Features

1. **Layered TXOP with importance-matched MCS:** Base layer at MCS_low (guaranteed delivery) + enhancement at MCS_high (opportunistic). No prior system applies different MCS to importance tiers of the same frame's token sequence within a single TXOP.

2. **Block ACK as QoE instrument:** The standard Block ACK mechanism is repurposed as a semantic quality measurement tool. Token position in the bitmap = quality indicator. No new protocol messages, no MLLM inference, no external quality probes.

3. **Variable enhancement token count as the primary quality lever:** k_enh replaces MCS as the adaptation control variable. Transmitting fewer tokens is semantically meaningful (graceful degradation); transmitting fewer bits in conventional coding is not (block corruption).

4. **Freeze-free guarantee by architecture:** The base layer guarantee makes display freeze structurally impossible, regardless of channel condition. This is a system property provable from the MCS_low erasure probability, not a statistical claim.

---

## Advantages and Effects

**Eliminates video freeze** — the most damaging user experience event in real-time video (MOS penalty ≈ −1.5 points for freeze vs. −0.3 for equivalent-duration quality reduction).

**Constant, bounded latency** — TCLA always completes in 16ms. Conventional latency grows with retransmissions, violating the 33.3ms deadline under poor channel.

**Zero-overhead QoE feedback** — Block ACK bitmap repurposed as quality signal. No new protocol fields, no MLLM at feedback time, no application-layer measurement probes.

**Energy efficiency** — No retransmissions means no wasted radio airtime on tail token recovery. Enhancement layer is sized to exactly what the channel can reliably carry.

**Graceful user experience** — Failure mode is "video was slightly blurry" (MOS −0.3), not "video froze" (MOS −1.5).

---

---

# Patent 2: Semantic Token Interleaving for Burst-Loss Resilience

## Background and Prior Art

Wi-Fi channels fail in **bursts**, not randomly at the bit level. The Gilbert-Elliott model captures this: the channel transitions between Good and Bad states, with the Bad state (representing interference, fading, or collision) persisting for multiple consecutive MPDU transmission attempts.

In conventional sequential MPDU packing, tokens are placed in order: MPDU 1 carries tokens T1..T16, MPDU 2 carries T17..T32, etc. A burst destroying MPDUs 3, 4, 5 eliminates tokens T33..T80 — a contiguous importance band.

**For a 1D TTD tokenizer:** This destroys a contiguous quality tier (e.g., the entire medium-detail layer), creating a sharp quality cliff rather than smooth degradation. The MaskGIT predictor performs poorly because it must reconstruct a large block of correlated missing tokens with no nearby received context.

**For a 2D spatial tokenizer:** Contiguous MPDUs correspond to contiguous image regions. A burst destroys one spatial area entirely (e.g., the top-right quadrant), again giving MaskGIT nothing nearby to condition on.

**Prior art gap:** No prior work addresses token-level interleaving across MPDUs as a mechanism for maximising MaskGIT reconstruction quality under burst losses. Conventional PHY-layer interleaving operates on encoded bits to combat frequency-selective fading — a different problem at a different layer with no semantic awareness.

---

## Technical Solution

Semantic Token Interleaving assigns tokens to MPDUs such that any burst of B consecutive lost MPDUs produces **maximally uncorrelated token losses** — spreading damage across both the image spatially and across the importance spectrum.

### Mode A: Stratified Importance Interleaving (for 1D TTD tokenizers)

Each MPDU receives a **stratified sample** across the full importance spectrum. For L=256 tokens and 16 MPDUs of 16 tokens each:

```
MPDU  1: tokens { 1, 17, 33, 49, 65, 81, 97, 113, 129, 145, 161, 177, 193, 209, 225, 241 }
MPDU  2: tokens { 2, 18, 34, 50, 66, 82, 98, 114, 130, 146, 162, 178, 194, 210, 226, 242 }
...
MPDU 16: tokens { 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256 }
```

Each MPDU contains one token from every importance stratum. Losing any single MPDU removes one token from each stratum — a small, **uniform quality degradation** that MaskGIT handles well (each missing token has rich context from its received neighbours in the same stratum).

Losing a burst of B MPDUs removes B tokens from each stratum — still uniform. MaskGIT's reconstruction quality is substantially higher than for a contiguous loss of the same total number of tokens.

### Mode B: Spatial Sub-Lattice Interleaving (for 2D tokenizers)

For tokenizers with a spatial layout (e.g., VQGAN 16×16 grid), tokens are assigned to MPDUs via a sub-lattice mapping: token at grid position (r, c) → MPDU index `(r mod stride) × stride + (c mod stride)`. Tokens in the same MPDU are at least `stride` positions apart in both spatial dimensions. A burst destroying consecutive MPDUs hits spatially distributed tokens, not a contiguous region.

### Adaptive Interleaving Depth

The interleaving stride is matched to the estimated burst length, derived from Block ACK history. Longer estimated burst → deeper interleaving. The SCS (Patent 3) tracks recent Block ACK bitmaps and provides a burst length estimate.

### Interaction with TCLA

When TCLA has selected k_enh for the enhancement layer, the Semantic Interleaving Engine operates on those k_enh tokens to determine their MPDU assignment. The interleaving map is registered with the SCS so that the Block ACK bitmap correctly identifies which tokens were received (required by TCLA's QoE feedback loop). The two mechanisms compose cleanly.

---

## Distinguishing Features

**Semantic-aware interleaving objective:** The assignment is driven by token semantic properties (importance rank or spatial position), not by generic error-correction principles. This is only possible because the token representation exposes semantic structure in a form the MAC layer can act on.

**Optimisation for MLLM reconstruction quality:** Conventional interleaving maximises BER performance after decoding. Semantic interleaving maximises MaskGIT prediction accuracy on the received token context — a fundamentally different and novel objective.

**Stratified importance sampling:** Each MPDU mirrors the full importance spectrum of the original sequence. No prior work applies stratified sampling from an importance ordering to MPDU composition.

---

## Advantages and Effects

**Under burst loss (the dominant Wi-Fi failure mode):** Stratified interleaving converts a catastrophic event (entire quality tier missing) into a manageable one (one token from each tier missing). MaskGIT reconstruction quality is substantially higher.

**For MLLM efficiency:** Scattered losses allow MaskGIT to use nearby received tokens as context for each prediction. Confidence per masked position is higher; reconstruction converges faster; fewer iterative MaskGIT rounds needed.

**Composability:** Orthogonal to TCLA (which controls how many tokens to send) and to SCS (which handles the cross-layer interface). Each addresses a distinct failure mode; combined benefits are additive.

---

---

# Patent 3: Semantic Convergence Sublayer (SCS) — Cross-Layer Token Signaling

## Background and Prior Art

TCLA and Semantic Interleaving both require the MAC layer to have knowledge it currently has no way of obtaining:

- Which bytes in the MPDU payload correspond to which tokens
- The importance rank of each token
- What codebook the tokens are drawn from
- Whether the receiver's detokenizer and MaskGIT are compatible with the transmitter

The IEEE 802.11 MAC sublayer is designed to be payload-agnostic — it treats MSDU payloads as opaque byte sequences. There is no mechanism for the application layer to annotate payload bytes with semantic metadata, and no mechanism for the receiver to communicate payload-level reception outcomes back to the application in semantic terms.

**Prior art gap:** No standard or research system defines a cross-layer interface that: (a) passes per-token importance metadata from application to MAC, (b) maintains and exposes the bit-to-token mapping for Block ACK translation, (c) synchronises tokenizer codebook and MaskGIT model identity between transmitter and receiver, and (d) binds transmitted token count and interleaving map to each frame. The SCS is the enabling infrastructure without which TCLA and Semantic Interleaving cannot be correctly implemented on real hardware.

---

## Technical Solution

The **Semantic Convergence Sublayer (SCS)** is a thin sublayer inserted between the application layer and the conventional 802.11 MAC sublayer. It defines four functional components:

### Component 1: Token Metadata Registry (Application → MAC, per-frame)

```
TokenFrameDescriptor {
    frame_id:          uint32    // unique frame identifier
    codebook_id:       uint16    // tokenizer codebook version
    L:                 uint16    // total tokens in full sequence (e.g., 256)
    k_base:            uint16    // base layer token count
    k_enh:             uint16    // enhancement layer token count (set by TCLA)
    bits_per_token:    uint8     // fixed width (e.g., 12 for codebook 4096)
    importance_mode:   enum      // TTD_POSITIONAL | EXPLICIT_SCORES | SPATIAL_2D
    importance_scores: float32[] // omitted for TTD (position = score — zero overhead)
    interleaving_map:  uint8     // which interleaving pattern to use
    context_token:     bytes     // cross-modal context for k=0 generative fallback
}
```

For TTD-trained tokenizers, the `importance_scores` field is omitted entirely — token position is the importance score by definition. This is the **zero-overhead case** — no side information is needed.

### Component 2: Bit-to-Token Mapping Engine (MAC internal)

The SCS maintains a per-MPDU token range table used in two directions:

**Forward:** During TCLA k_enh selection, to know how many MPDUs are needed for k tokens.

**Backward:** During Block ACK processing, to translate each ACK/NACK bit into received vs. lost token positions, enabling the QoE inference: `k_rx = received_MPDUs × TPM → QoE(k_rx) from lut`.

Without this mapping, the Block ACK cannot be used as a QoE instrument.

### Component 3: Codebook and Model Synchronisation (Transmitter ↔ Receiver, session setup)

Before a TokCom session begins, the SCS exchanges a capability advertisement during the 802.11 association procedure (piggybacked as a vendor-specific information element):

```
TokComCapabilityAdvertisement {
    codebook_id:        uint16    // must match on both sides
    codebook_version:   uint8     // for codebook updates
    tokenizer_model_id: uint32    // identifies the encoder model
    maskgit_model_id:   uint32    // identifies the masked token predictor
    max_L:              uint16    // maximum sequence length supported
    supported_features: bitmask   // TCLA | INTERLEAVING | LAYERED_TXOP
}
```

If codebook IDs do not match, TokCom mode is not activated — the session falls back to conventional transmission. This makes the SCS **fully backward-compatible**: a TokCom AP serves both TokCom and legacy clients simultaneously.

### Component 4: Per-Frame Binding Log (MAC → Application, per-frame confirmation)

After each TXOP and Block ACK, the SCS returns to the application:

```
TokComFrameResult {
    frame_id:       uint32    // matches transmitted frame
    k_base_rx:      uint16    // base layer tokens confirmed by Block ACK
    k_enh_rx:       uint16    // enhancement layer tokens confirmed
    k_rx:           uint16    // total received tokens (= k_base_rx + k_enh_rx)
    qoe_predicted:  float32   // QoE(k_rx) from lut — zero MLLM
    burst_estimate: uint8     // estimated burst length from this Block ACK
    interleaving_id: uint8    // which map was used — receiver needs this to unmask
}
```

The receiver uses `k_rx` and `interleaving_id` to set up the MaskGIT mask correctly — marking positions k_rx+1..L for prediction. Without this binding, the receiver cannot know how many tokens were actually sent (TCLA may have sent fewer than L).

### The SCS Stack

```
┌──────────────────────────────────────────────────────┐
│                APPLICATION LAYER                     │
│   Tokenizer → TokenFrameDescriptor + token bytes    │
└─────────────────────┬────────────────────────────────┘
                      │ TokenFrameDescriptor (downward)
                      │ TokComFrameResult    (upward)
┌─────────────────────▼────────────────────────────────┐
│          SEMANTIC CONVERGENCE SUBLAYER (SCS)         │
│                                                      │
│  Token Metadata Registry                            │
│  Bit-to-Token Mapping Engine  ←── Block ACK bitmap  │
│  TCLA Controller (k_enh selection + adaptation)     │
│  Interleaving Engine                                │
│  Codebook/Model Sync (session setup)                │
│  Per-Frame Binding Log                              │
└─────────────────────┬────────────────────────────────┘
                      │ Standard MSDU interface
┌─────────────────────▼────────────────────────────────┐
│           CONVENTIONAL 802.11 MAC SUBLAYER           │
│   A-MPDU aggregation → CSMA/CA → Block ACK          │
└──────────────────────────────────────────────────────┘
```

---

## Distinguishing Features

**Importance signaling:** First standardisable interface for passing per-token semantic importance from application to MAC. The TTD zero-overhead case (position = importance, no scores needed) is particularly elegant — the interface adds negligible complexity for the most important tokenizer class.

**Bit-to-token mapping:** Formally defines the correspondence between MAC-level delivery outcomes (Block ACK bits) and application-level semantic outcomes (token positions received). This enables TCLA's QoE feedback loop to operate on existing MAC machinery without any protocol modification.

**Codebook synchronisation:** First wireless protocol mechanism for negotiating shared token representation between transmitter and receiver. Analogous to codec negotiation in SIP/SDP but operating at the discrete token codebook level, with explicit model identity. Backward compatibility built in.

**Per-frame binding:** Ensures that k_enh, the interleaving map, and the importance metadata are atomically bound to each transmitted frame. Prevents correctness failures where the receiver's MaskGIT masks the wrong positions because it doesn't know how many tokens were sent.

---

## Advantages and Effects

**Enables TCLA and Interleaving on real hardware:** Without the SCS, both Patent 1 and Patent 2 are conceptually valid but unimplementable on real 802.11 hardware. The SCS is the bridge from concept to deployable system.

**Zero new frame types:** The capability advertisement reuses existing management frame vendor-specific elements. No new on-air protocol. All new logic is internal to the device driver stack.

**Backward compatibility:** SCS activates only when both sides advertise TokCom capability. An SCS-capable AP can simultaneously serve TokCom and legacy Wi-Fi clients.

**Enables heterogeneous MLLM deployment:** Different receivers may run different MaskGIT implementations. The `maskgit_model_id` allows the transmitter to know receiver capabilities and potentially adapt k_enh accordingly — a new form of receiver-capability-aware transmission with no precedent in wireless standards.

---

---

## Cross-Patent Relationships

```
           Patent 3: SCS
           (infrastructure — enables the other two)
                │
        ┌───────┴───────┐
        ↓               ↓
  Patent 1: TCLA    Patent 2: Interleaving
  (how many tokens  (which tokens go in
   to send per       which MPDUs, for
   layer; QoE        burst-loss diversity)
   feedback loop)
```

The SCS provides the data — importance scores, bit-to-token map, codebook sync, frame binding — that both TCLA and Interleaving need to operate correctly. TCLA and Interleaving are composable: TCLA selects k_enh, Interleaving determines the MPDU assignment for those tokens. Block ACK feeds back to both: to TCLA for QoE-driven k_enh adaptation, to the Interleaving engine for burst length estimation and adaptive depth.

An ablation shows each mechanism contributing independently: Interleaving alone reduces burst loss damage; TCLA alone prevents latency deadline misses; SCS alone enables correct implementation. Together they form a coherent semantic MAC architecture where quality, resilience, and latency are jointly optimised for the first time.

---

## Prior Art Clearance Summary

| Invention | Closest Prior Art | Decisive Distinction |
|-----------|-------------------|----------------------|
| TCLA Layered TXOP | RTP/H.265, ABR streaming, HARQ | Layered MCS per importance tier; Block ACK as QoE instrument; variable k_enh as quality lever; freeze-free guarantee by architecture |
| Semantic Interleaving | PHY-layer bit interleaving | Token-level, semantic-aware, MLLM reconstruction quality as optimisation objective |
| SCS | Cross-layer video scheduling (LTE/5G) | Per-token importance, bit-to-token mapping, codebook synchronisation, per-frame binding — none exist in any standard |

---

*End of TokCom Patent Idea Document v3*  
*Three inventions. One coherent semantic MAC stack. Simulation-validated on real One-D-Piece TTD tokenizer.*
