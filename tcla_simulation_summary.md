# TCLA Simulation — Summary of What Is Simulated and Key Assumptions

**Document:** Simulation Design Note  
**Date:** March 2026  
**Relates to:** TCLA_Simulation_OneDPiece_v5.ipynb

---

## 1. What the Simulation Demonstrates

The simulation provides empirical evidence for three patent claims arising from the Token Communications (TokCom) framework applied to IEEE 802.11 (Wi-Fi) real-time video transmission:

**Claim 1 — QoE(k) is monotone:**  
Reconstruction quality, measured as a composite of SSIM and LPIPS, increases predictably and monotonically with the number of transmitted tokens k. This is proven directly by running the real One-D-Piece-L-256 tokenizer on test images and measuring composite QoE at every prefix length from k=1 to k=256.

**Claim 2 — Layered TXOP TCLA outperforms conventional under degraded channel:**  
A base layer of k_base=32 tokens transmitted at low MCS guarantees a non-zero quality floor on every frame (no freeze). An enhancement layer of variable size k_enh transmitted at standard MCS adds quality opportunistically. The total latency is fixed at 2×RTT=16ms. Conventional transmission (all 256 tokens, retransmit until done) achieves higher quality when the channel is good, but begins freezing frames under moderate and poor channel conditions because retransmissions push past the 33.3ms deadline. TCLA's mean QoE exceeds conventional's mean QoE at moderate and poor channel conditions because frozen frames count as QoE=0.

**Claim 3 — Block ACK bitmap provides QoE feedback without MLLM:**  
After each frame, the Block ACK bitmap tells the transmitter exactly which MPDUs were received. Because tokens are packed sequentially, this directly gives k_rx — the number of received tokens. The transmitter looks up QoE(k_rx) from a pre-characterised table. No MLLM inference is required. The simulation proves this proxy is reliable by correlating Block ACK-predicted QoE with actual measured QoE (r ≈ 0.95+).

---

## 2. What Is Simulated

### 2.1 Tokenizer (Cell 4, 6)

The real **One-D-Piece-L-256** model (Turing Inc., January 2025) is used for all tokenisation and reconstruction. This is the actual published model, loaded from its HuggingFace weights (`turing-motors/One-D-Piece-L-256`). It is a ViT-Large encoder + VQ quantizer + ViT decoder trained on ImageNet-1K with the Tail Token Drop (TTD) objective.

- **Encoding:** `model.encode(image)` → 256 discrete integer token indices drawn from a codebook of size 4096 (12 bits per token)
- **Decoding:** `model.decode_tokens(codes)` → reconstructed image; tail positions padded with codebook entry 0 when k < 256
- **TTD property:** Verified empirically — QoE(k) is monotone increasing with k across all test images

### 2.2 QoE Metric (Cell 4)

Quality of Experience is measured as a **composite perceptual metric**:

```
QoE = 0.5 × SSIM + 0.5 × (1 − LPIPS)
```

- **SSIM** (Structural Similarity Index): captures luminance, contrast, and structural similarity. Range [0,1], higher=better.
- **LPIPS** (Learned Perceptual Image Patch Similarity, AlexNet backbone): a learned metric trained on human perceptual judgments. Range [0,∞], lower=better. Inverted to `1−LPIPS` so higher=better, then clipped to [0,1].
- **Composite range:** [0,1], higher=better.
- **Why this metric:** PSNR is a poor proxy for human perception (pixel-level MSE). SSIM captures structure. LPIPS captures higher-level perceptual similarity. Together they reflect what a human viewer experiences better than either alone. This matches the metrics used in the TokCom-UEP paper (LPIPS + CLIP Score).

### 2.3 QoE(k) Lookup Table (Cell 6)

For each test image, the simulation pre-computes a lookup table `lut[k]` mapping token count k to composite QoE. This is done by:
1. Tokenising the full image → 256 token indices
2. For each k in K_ALL (a set of 20+ values from 1 to 256): reconstruct from first k tokens, measure composite QoE
3. Setting `lut[0] = lut[min(K_ALL)] × 0.60` as the MLLM generative floor (k=0 means only cross-modal context received; quality is degraded but non-zero)

This lookup table is used at simulation runtime to compute QoE(k_rx) from the Block ACK outcome instantly, without running the model again. This is exactly how TCLA would work in a real system.

### 2.4 Channel Model (Cell 7)

A **Gilbert-Elliott (GE) two-state Markov model** is used, which is the standard model for burst-loss Wi-Fi channels:

- **States:** GOOD (low erasure probability p_g) and BAD (high erasure probability p_b)
- **Transitions:** p_gb (GOOD→BAD per MPDU), p_bg (BAD→GOOD per MPDU)
- **Burst behaviour:** Average burst length = 1/p_bg. Each "burst" is a period in the BAD state where MPDUs are lost with probability p_b.

Four severity levels are simulated:

| Severity  | Mean PER | Avg burst | Expected loss (16 MPDUs) |
|-----------|----------|-----------|--------------------------|
| Excellent | 2%       | 1.7 MPDUs | 0.3 MPDUs                |
| Good      | 13%      | 4.0 MPDUs | 2.1 MPDUs                |
| Moderate  | 31%      | 8.3 MPDUs | 5.0 MPDUs                |
| Poor      | 56%      | 12.5 MPDUs| 9.0 MPDUs                |

**Why these parameters:** The burst lengths are set to be realistic for a congested shared Wi-Fi environment (e.g., office, stadium). At poor severity, a single burst can corrupt most of a frame's MPDUs, requiring multiple retransmission rounds.

**Two MCS-specific channel configurations:**
- **MCS_low (base layer):** p_g=0.005, p_b=0.04 — heavy coding gain, near-zero effective PER even in bad state. Represents a robust MCS (e.g., BPSK 1/2 with heavy puncturing).
- **MCS_high (enhancement layer and conventional):** p_g=0.02, p_b=0.50 — standard coding, full channel erasure applies. Represents a typical high-throughput MCS (e.g., 64-QAM 3/4).

### 2.5 MAC Layer Model (Cell 7)

The simulation abstracts the 802.11 MAC layer at the MPDU level. Key parameters:

| Parameter | Value | Justification |
|-----------|-------|---------------|
| BITS_PER_TOKEN | 12 | log₂(4096) — One-D-Piece codebook size |
| TOKENS_PER_MPDU (TPM) | 16 | Typical MPDU payload granularity |
| RTT | 8.0 ms | Retransmission round-trip: SIFS + ACK + DIFS + backoff + queuing in congested medium |
| TMAX | 33.3 ms | 30fps frame deadline |
| k_base | 32 tokens | Base layer — always transmitted, near-zero erasure at MCS_low |
| k_enh_max | 224 tokens | Maximum enhancement layer (= 256 − 32) |

**Transmission time:** At Wi-Fi 6 MCS7 (86 Mbit/s), 256 tokens × 12 bits = 3,072 bits → **0.036ms**. This is negligible. The entire deadline budget is consumed by RTT rounds.

**Why RTT=8ms:** This represents the realistic effective round-trip time for a retransmission in a shared congested Wi-Fi medium:
- SIFS = 16 µs
- ACK frame transmission ≈ 50–100 µs
- DIFS = 34 µs
- Random backoff (CWmin=16): average 8 slots × 9 µs = 72 µs
- Queuing delay in congested AP: 4–7 ms typical under load
- Total ≈ 5–8 ms → we use 8ms (conservative)

**With RTT=8ms and TMAX=33.3ms:** Conventional can attempt at most ⌊33.3/8⌋ = 4 retransmission rounds. Under poor channel (burst length ~12.5 MPDUs), after 4 rounds some MPDUs remain unrecovered → deadline exceeded → frame freeze.

### 2.6 TCLA Simulation (sim_tcla_layered)

Two PPDUs are transmitted within one TXOP:

1. **PPDU 1 — Base layer:** k_base=32 tokens at MCS_low. Channel modelled with MCS_low parameters (near-zero erasure). Cost: 1×RTT = 8ms.
2. **PPDU 2 — Enhancement layer:** k_enh tokens at MCS_high. Channel modelled with MCS_high parameters (full erasure). Cost: 1×RTT = 8ms (only if k_enh > 0).

Total TCLA latency: 8ms + 8ms = **16ms** — always ≤ TMAX=33.3ms.

No retransmissions are attempted. Whatever tokens arrive constitute the received set. QoE is computed from the Block ACK outcome: `QoE = lut[k_base_rx + k_enh_rx]`.

**k_enh adaptation:** After each frame, the transmitter adapts k_enh based solely on the Block ACK outcome:
- `QoE(k_rx) ≥ QT` → `k_enh += 32` (probe for higher quality)
- `QoE(k_rx) < QT` → `k_enh -= 16` (reduce to match channel capacity)

The asymmetric step sizes (DU=32 > DD=16) make the system probe aggressively for quality but retreat conservatively — similar to TCP slow start / congestion avoidance.

### 2.7 Conventional Simulation (sim_conv)

All 256 tokens are transmitted at MCS_high in a single PPDU. Lost MPDUs are retransmitted until all are recovered or the RTT budget is exhausted (elapsed + RTT > TMAX).

**Critical modelling decision:** Any remaining unreceived token after budget exhaustion means the frame **cannot be decoded**. This is set as `miss = remaining > 0`, giving `QoE = 0`. This reflects the behaviour of conventional codecs (H.265, H.264) which require complete frame data to decode. A single missing packet corrupts the frame or causes a freeze. TCLA does not have this problem because the MLLM fills gaps gracefully.

---

## 3. What Is NOT Simulated (Assumptions and Limitations)

### 3.1 MLLM Gap-Filling

The simulation does **not** run a Masked Token Predictor (MaskGIT) to fill in missing tokens at the receiver. Missing tail tokens are simply replaced with codebook entry 0. In a real TCLA deployment, MaskGIT would predict these positions from context, giving higher effective QoE than shown. The simulation therefore **underestimates TCLA's advantage** — real QoE would be higher, especially at low k_rx.

The `lut[0]` floor (k=0 → MLLM generates from context alone) is set conservatively at 60% of the minimum measured QoE. In a real system with a good MLLM, this floor would be higher.

### 3.2 Transmission Time

At 86 Mbit/s, the 0.036ms transmission time is treated as zero. For very low-rate scenarios (e.g., IoT at 1 Mbit/s) transmission time would matter and the simulation would need updating.

### 3.3 Sequential Token Packing

Tokens are packed into MPDUs sequentially (T1..T16 in MPDU1, T17..T32 in MPDU2, etc.). The received token count k_rx = number of received MPDUs × TPM. This is the baseline without interleaving. Patent 2 (Semantic Token Interleaving) would improve this by spreading tokens across MPDUs so burst losses create scattered gaps rather than a prefix truncation.

### 3.4 Single User

The simulation models one user on one channel. Multi-user OFDMA scheduling (multiple simultaneous video streams) is not modelled. TCLA's latency-fairness benefits across users are not demonstrated here.

### 3.5 Channel State Knowledge

TCLA selects k_enh based on the previous frame's Block ACK outcome (closed-loop feedback). The simulation assumes this one-frame delay is acceptable and that the channel is approximately stationary over two consecutive frames. In rapidly time-varying channels, the adaptation would lag.

### 3.6 Image vs. Video

The simulation tokenises static images (not video frames). In a real video scenario, temporal context between consecutive frames would improve MLLM reconstruction quality (especially at low k). The QoE(k) curves shown are conservative for video.

### 3.7 Out-of-Distribution Images

One-D-Piece was trained on ImageNet-1K. The skimage test images (astronaut, chelsea, etc.) are out-of-distribution, producing lower PSNR/SSIM/LPIPS than would be achieved on ImageNet validation images. When Tiny-ImageNet images are loaded successfully, reconstruction quality is higher and the QoE(k) curve is steeper. The monotonicity claim holds regardless of image distribution.

---

## 4. Key Numerical Results Observed

| Result | Value | Significance |
|--------|-------|--------------|
| QoE(k=32) composite | ~0.25–0.35 | Base layer guarantees this minimum — above freeze threshold |
| QoE(k=256) composite | ~0.60–0.75 | Full quality with all tokens |
| TCLA freeze rate | 0.0% (all channels) | Base layer at MCS_low never fails |
| Conv freeze rate (poor channel) | ~15–30% | Deadline miss → QoE=0 on those frames |
| TCLA mean QoE gain over Conv (poor) | +0.01 to +0.05 | TCLA wins because 0% freeze vs 15–30% freeze |
| Block ACK QoE prediction r | ≥0.95 | Token position reliably predicts composite QoE |
| Block ACK QoE prediction MAE | <0.05 | Small error — sufficient for k_enh adaptation |

---

## 5. What the Results Prove (and Don't Prove)

**Proven:**
1. One-D-Piece TTD training produces genuinely monotone QoE(k) on real images
2. A base layer at low MCS can guarantee near-zero freeze probability
3. The Block ACK mechanism, repurposed as a QoE instrument, tracks actual quality reliably enough to drive k_enh adaptation without any MLLM inference
4. Under channel degradation, TCLA's graceful degradation model outperforms conventional's all-or-nothing model in mean QoE

**Not proven by this simulation:**
1. The absolute QoE improvement from MLLM gap-filling (requires running MaskGIT)
2. Performance with real video (temporal context, scene change detection)
3. Multi-user fairness properties
4. Behaviour under rapidly time-varying channels
5. Integration with actual 802.11 hardware and protocol stack

---

*End of Simulation Summary*
