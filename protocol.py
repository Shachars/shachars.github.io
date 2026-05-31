"""
TokCom UDP / 802.11 Protocol
============================

Token packing for Block ACK inference
--------------------------------------
The core idea: tokens are packed into fixed-size MPDUs so that the 802.11
Block ACK bitmap directly reports token delivery to the transmitter — no
application-layer feedback protocol is needed.

MPDU packing:
  - Each MPDU carries exactly TOKENS_PER_MPDU token IDs
  - One video frame = ceil(k / TOKENS_PER_MPDU) MPDUs = one A-MPDU burst
  - The 802.11 BA bitmap has one bit per MPDU → TX can count k_rx immediately
  - k_rx = (number of ACKed MPDUs) × TOKENS_PER_MPDU

Packet format (one MPDU payload, binary little-endian):
  Offset  Size  Field
  ------  ----  -----
  0       4B    frame_id        (uint32) — video frame sequence number
  4       2B    mpdu_idx        (uint16) — index of this MPDU within the A-MPDU
  6       2B    mpdu_total      (uint16) — total MPDUs in this A-MPDU
  8       2B    token_offset    (uint16) — index of first token in this MPDU
  10      2B    tokens_in_mpdu  (uint16) — number of tokens in this MPDU
  12      2B    total_tokens_N  (uint16) — full sequence length N
  14      2B    frame_w         (uint16)
  16      2B    frame_h         (uint16)
  18      t×2B  token_ids       (uint16[t]) — t = tokens_in_mpdu

Block ACK inference (real 802.11):
  On Linux, after sending the A-MPDU, read per-station TX counters from
  debugfs or via nl80211 to determine how many MPDUs were ACKed:

    path = f'/sys/kernel/debug/ieee80211/phy0/netdev:{iface}/stations/{peer}/ampdu_stats'

  The delta in tx_msdu_success between consecutive reads = MPDUs ACKed.
  k_rx = mpdu_acked × TOKENS_PER_MPDU

  This requires NO protocol modification to 802.11 — the BA exchange happens
  automatically as part of A-MPDU operation in 802.11n/ac/ax.

IPR note:
  This constitutes a novel method for semantic quality adaptation using
  unmodified 802.11 Block ACK as an implicit QoE feedback channel.
  The transmitter never needs to know SNR, MCS, or PER explicitly.
"""

import struct
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

# ── Packing constants ─────────────────────────────────────────────────────────
TOKENS_PER_MPDU  = 16          # tokens per MPDU — tunable tradeoff:
                                #   fewer = finer BA granularity
                                #   more  = less overhead per token
BITS_PER_TOKEN   = 12          # log2(4096 codebook)
TOKEN_BYTES      = 2           # uint16 per token ID
MAX_N_TOKENS     = 256
MAX_MPDU_PAYLOAD = 18 + TOKENS_PER_MPDU * TOKEN_BYTES   # bytes

MPDU_HEADER_FMT  = "<IHHHHHHH"   # frame_id, mpdu_idx, mpdu_total,
                                   # token_offset, tokens_in_mpdu,
                                   # total_tokens_N, frame_w, frame_h
MPDU_HEADER_SIZE = struct.calcsize(MPDU_HEADER_FMT)      # 18 bytes

DEFAULT_HOST     = "127.0.0.1"
DEFAULT_PORT_DATA = 9999
DEFAULT_PORT_CTRL = 9998   # Block ACK inference feedback (simulation only)


# ── Per-MPDU packet ───────────────────────────────────────────────────────────
@dataclass
class TokComMPDU:
    frame_id:        int
    mpdu_idx:        int
    mpdu_total:      int
    token_offset:    int
    tokens_in_mpdu:  int
    total_tokens_N:  int
    frame_w:         int
    frame_h:         int
    token_ids:       np.ndarray   # shape (tokens_in_mpdu,), dtype uint16

    def encode(self) -> bytes:
        header = struct.pack(
            MPDU_HEADER_FMT,
            self.frame_id,
            self.mpdu_idx,
            self.mpdu_total,
            self.token_offset,
            self.tokens_in_mpdu,
            self.total_tokens_N,
            self.frame_w,
            self.frame_h,
        )
        payload = self.token_ids.astype(np.uint16).tobytes()
        return header + payload

    @staticmethod
    def decode(data: bytes) -> "TokComMPDU":
        hdr = struct.unpack(MPDU_HEADER_FMT, data[:MPDU_HEADER_SIZE])
        (frame_id, mpdu_idx, mpdu_total,
         token_offset, tokens_in_mpdu,
         total_tokens_N, frame_w, frame_h) = hdr
        ids = np.frombuffer(
            data[MPDU_HEADER_SIZE : MPDU_HEADER_SIZE + tokens_in_mpdu * 2],
            dtype=np.uint16
        ).copy()
        return TokComMPDU(
            frame_id, mpdu_idx, mpdu_total,
            token_offset, tokens_in_mpdu,
            total_tokens_N, frame_w, frame_h, ids,
        )


# ── Frame → MPDU list ─────────────────────────────────────────────────────────
def pack_frame(
    frame_id:   int,
    token_ids:  np.ndarray,      # shape (k,) — the k tokens to send this frame
    total_N:    int,
    frame_w:    int,
    frame_h:    int,
) -> List[TokComMPDU]:
    """
    Split k token IDs into ceil(k / TOKENS_PER_MPDU) MPDUs.
    Returns the list; TX sends them as an A-MPDU burst.
    """
    k      = len(token_ids)
    mpdus  = []
    mpdu_total = int(np.ceil(k / TOKENS_PER_MPDU))

    for idx in range(mpdu_total):
        start = idx * TOKENS_PER_MPDU
        end   = min(start + TOKENS_PER_MPDU, k)
        mpdus.append(TokComMPDU(
            frame_id       = frame_id,
            mpdu_idx       = idx,
            mpdu_total     = mpdu_total,
            token_offset   = start,
            tokens_in_mpdu = end - start,
            total_tokens_N = total_N,
            frame_w        = frame_w,
            frame_h        = frame_h,
            token_ids      = token_ids[start:end].astype(np.uint16),
        ))
    return mpdus


# ── Block ACK inference ───────────────────────────────────────────────────────
@dataclass
class BlockACKResult:
    """
    Result of reading the 802.11 Block ACK for one A-MPDU transmission.

    In a real system this comes from the driver (debugfs / nl80211).
    In the simulation it is computed from the channel PER model.
    """
    frame_id:      int
    mpdu_sent:     int
    mpdu_acked:    int
    k_rx:          int           # tokens actually received = mpdu_acked × TOKENS_PER_MPDU
    ba_bitmap:     int           # raw bitmap (mpdu_sent bits), LSB = first MPDU


def simulate_block_ack(
    mpdus_or_k,               # List[TokComMPDU]  OR  int k_sent
    per:    float,
    rng:    np.random.Generator,
):
    """
    Simulate the 802.11 Block ACK response.

    Accepts two call signatures:

      simulate_block_ack(k_sent: int, per, rng) -> int
          Simple form used by the simulation notebook.
          Returns k_rx (int) — tokens in the contiguous received prefix.

      simulate_block_ack(mpdus: List[TokComMPDU], per, rng) -> BlockACKResult
          Full MPDU-aware form used by transmitter.py.
          Returns a BlockACKResult with bitmap and per-MPDU details.

    In a real system both forms are replaced by reading the driver:
        read_block_ack_from_driver(iface, peer_mac) → BlockACKResult
    """
    # ── Simple integer form (simulation / notebook) ───────────────────────────
    if isinstance(mpdus_or_k, (int, np.integer)):
        k_sent  = int(mpdus_or_k)
        n_mpdus = max(1, int(np.ceil(k_sent / TOKENS_PER_MPDU)))
        k_rx    = 0
        for i in range(n_mpdus):
            if rng.random() >= per:          # MPDU received
                k_rx += min(TOKENS_PER_MPDU, k_sent - i * TOKENS_PER_MPDU)
            else:
                break                        # contiguous prefix: stop at first gap
        return k_rx

    # ── Full MPDU list form (transmitter.py) ──────────────────────────────────
    mpdus  = mpdus_or_k
    bitmap = 0
    acked  = 0
    for i, mpdu in enumerate(mpdus):
        if rng.random() >= per:
            bitmap |= (1 << i)
            acked  += 1

    k_rx = acked * TOKENS_PER_MPDU
    if acked == len(mpdus):
        k_rx = sum(m.tokens_in_mpdu for m in mpdus)
    elif acked > 0:
        k_rx = sum(
            mpdus[i].tokens_in_mpdu
            for i in range(len(mpdus))
            if bitmap & (1 << i)
        )

    return BlockACKResult(
        frame_id   = mpdus[0].frame_id,
        mpdu_sent  = len(mpdus),
        mpdu_acked = acked,
        k_rx       = k_rx,
        ba_bitmap  = bitmap,
    )


# ── Real 802.11 Block ACK reader (Linux) ──────────────────────────────────────
def read_block_ack_linux(
    iface:    str,
    peer_mac: str,
    phy:      str = "phy0",
) -> Optional[dict]:
    """
    Read per-station A-MPDU statistics from Linux mac80211 debugfs.
    Returns a dict with tx_msdu, tx_msdu_success, tx_msdu_failed, or None.

    The transmitter calls this after each A-MPDU burst and computes:
        mpdu_acked = delta(tx_msdu_success)
        k_rx       = mpdu_acked × TOKENS_PER_MPDU

    Requires: debugfs mounted at /sys/kernel/debug/
    Requires: station already associated (call `iw dev <iface> station dump` to get peer MAC)

    No 802.11 protocol modification needed — this is standard driver instrumentation.
    """
    path = (
        f"/sys/kernel/debug/ieee80211/{phy}/"
        f"netdev:{iface}/stations/{peer_mac}/ampdu_stats"
    )
    try:
        with open(path) as f:
            raw = f.read()
        stats = {}
        for line in raw.strip().split("\n"):
            if ":" in line:
                key, val = line.split(":", 1)
                stats[key.strip()] = val.strip()
        return stats
    except FileNotFoundError:
        return None


# ── TCLA k-adaptation controller ─────────────────────────────────────────────
class TCLAController:
    """
    Closed-loop token-count controller driven by Block ACK inference.

    On each frame:
      1. TX sends k_send tokens as an A-MPDU
      2. 802.11 BA reports k_rx tokens received
      3. Controller updates k_send for the next frame

    No SNR measurement, no PHY-layer information, no application ACK.
    The BA bitmap is the only feedback — unmodified 802.11 operation.

    Adaptation rule:
      - k_rx < k_send:        channel dropped tokens → back off to k_rx
      - k_rx == k_send:       all delivered → probe up by STEP_UP tokens
      - k_send reaches target: hold
    """

    def __init__(
        self,
        N:          int   = MAX_N_TOKENS,
        k_min:      int   = 8,
        k_init:     int   = 64,
        qoe_target: float = 0.80,     # QoE we're aiming for
        step_up:    int   = 8,        # tokens to add per successful frame
        step_down_factor: float = 1.0, # multiply k_rx directly on loss
    ):
        self.N               = N
        self.k_min           = k_min
        self.k_send          = k_init
        self.qoe_target      = qoe_target
        self.step_up         = step_up
        self.step_down_factor = step_down_factor
        self._history: List[BlockACKResult] = []

    @property
    def k_target(self) -> int:
        """
        k needed to meet qoe_target, from the analytical QoE(k) curve.
        QoE(k) = 0.15 + 0.85 * (1 - exp(-3.5 * k/N))
        Solve for k: k = -N/3.5 * ln(1 - (qoe_target - 0.15) / 0.85)
        """
        import math
        q = max(0.16, min(0.999, self.qoe_target))
        ratio = -(1.0 / 3.5) * math.log(1.0 - (q - 0.15) / 0.85)
        return max(self.k_min, min(self.N, int(ratio * self.N)))

    def update(self, ba_or_k_rx) -> int:
        """
        Ingest a Block ACK result and return k_send for the next frame.

        Accepts two forms:
          update(k_rx: int)          — simulation / notebook path
          update(ba: BlockACKResult) — hardware / transmitter.py path

        Adaptation rule:
          loss detected  → k_send = k_rx  (immediate back-off)
          no loss        → k_send += step_up  (gradual probe)
          k_send capped at min(N, k_target × 1.2)
        """
        # Unpack either form
        if isinstance(ba_or_k_rx, (int, np.integer)):
            k_rx = int(ba_or_k_rx)
        else:
            ba = ba_or_k_rx
            self._history.append(ba)
            k_rx = ba.k_rx

        if k_rx < self.k_send:
            self.k_send = max(self.k_min, k_rx)
        else:
            ceiling = min(self.N, int(self.k_target * 1.2))
            self.k_send = min(ceiling, self.k_send + self.step_up)

        return self.k_send

    def qoe(self, k: Optional[int] = None) -> float:
        k = k if k is not None else self.k_send
        import math
        ratio = min(k / self.N, 1.0)
        return 0.15 + 0.85 * (1.0 - math.exp(-3.5 * ratio))

    def summary(self) -> str:
        if not self._history:
            return "no frames processed"
        total    = len(self._history)
        losses   = sum(1 for b in self._history if b.k_rx < b.mpdu_sent * TOKENS_PER_MPDU)
        avg_k_rx = sum(b.k_rx for b in self._history) / total
        return (
            f"frames={total}  losses={losses} ({losses/total*100:.0f}%)  "
            f"avg_k_rx={avg_k_rx:.0f}/{self.N}  "
            f"current_k_send={self.k_send}  "
            f"QoE={self.qoe():.3f}"
        )
