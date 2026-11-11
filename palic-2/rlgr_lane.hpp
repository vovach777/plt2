// rlgr_lane.hpp - one bit lane per tile for many RLGR coders (variant "b", Vladimir's repair of full windows).
// Every code is written at the one bit head of the lane the moment it is known (Rice codes, raw bits). The only
// thing written later than it is read is the run header (flag 0 for a full window, or 1 + rl in k bits): it gets a
// hole of 1 + k bits at the head when the run starts (and again right after a full window); the encoder fills the
// hole at the end of the run, the decoder reads it from the hole by position when the run starts. A full window
// uses only the flag (a 0 bit, nothing to write); if its hole still ends at the head (nobody wrote after it during
// the run), the k unused bits are given back: head = hole + 1. The decoder moves its head the same way, so it sees
// the same condition. No pages, no reservation of worst cases, no ring: each coder keeps one open hole; the decoder
// is one sequential bit reader for the whole tile. Bits are MSB first inside 64-bit words (host order).
// The RLGR adaptation is llrice's (rlgr_fast.hpp constants).
#pragma once
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include "rlgr_fast.hpp"

namespace rlgrl {
using rlgrf::MAX_K; using rlgrf::UP_K; using rlgrf::DOWN_K; using rlgrf::UP_K_DIRECT_RL; using rlgrf::DOWN_K_DIRECT_RL;
using rlgrf::adapt_kr;

struct Lane {                                       // encoder side: zeroed words, a bit head
    std::vector<uint64_t> w; uint64_t head = 0;
    inline void grow() { const size_t need = size_t(head >> 6) + 4; if (__builtin_expect(need > w.size(), 0)) w.resize(std::max(need, w.size() * 2)); }
    inline void at(uint64_t pos, int n, uint64_t v) {   // OR n bits (1..64) at bit pos; the bits there are 0
        const size_t i = size_t(pos >> 6); const int off = int(pos & 63);
        if (off + n <= 64) w[i] |= v << (64 - off - n);
        else { w[i] |= v >> (off + n - 64); w[i + 1] |= v << (128 - off - n); }
    }
    inline void put64(int n, uint64_t v) { grow(); at(head, n, v); head += uint64_t(n); }
    inline void put(int n, uint32_t v) { put64(n, v); }
    inline void reserve(int) {}                     // nothing is laid ahead
    inline size_t words() const { return size_t((head + 63) >> 6); }
};
struct LaneR {                                      // decoder side: one sequential reader for the tile
    const uint64_t* w = nullptr; size_t nw = 0; uint64_t head = 0;   // nw includes 2 zero words of padding
    inline uint64_t peek_at(uint64_t pos) const {
        const size_t i = size_t(pos >> 6); const int off = int(pos & 63);
        if (__builtin_expect(i + 1 >= nw, 0)) throw std::runtime_error("rlgr lane: read past the end");
        return off ? (w[i] << off) | (w[i + 1] >> (64 - off)) : w[i];
    }
    inline uint64_t window() const { return peek_at(head); }
    inline void skip(int n) { head += uint64_t(n); }
    inline uint32_t get(int n) { if (!n) return 0; const uint32_t v = uint32_t(window() >> (64 - n)); head += uint64_t(n); return v; }
    inline void reserve(int) {}
};

inline void rice_put(Lane& s, uint32_t x, uint32_t m) {
    const uint32_t q = x >> m;
    if (q >= 32) { s.put64(64, (0xffffffffull << 32) | x); return; }
    s.put64(int(q + 1 + m), (((2ull << q) - 2) << m) | (x & ((1ull << m) - 1)));
}
inline uint32_t rice_get(LaneR& s, uint32_t m) {
    const uint64_t win = s.window(); const int q = win == ~0ull ? 64 : __builtin_clzll(~win);
    if (q >= 32) { s.skip(32); return s.get(32); }
    const uint32_t low = m ? uint32_t((win << (q + 1)) >> (64 - m)) : 0;
    s.skip(q + 1 + int(m)); return (uint32_t(q) << m) | low;
}

struct Encoder {
    Lane* s; uint32_t rl = 0, k = 1 << 3, kr = 1 << 3; bool inrun = false; uint64_t hole = 0, hole_end = 0;
    explicit Encoder(Lane* l) : s(l) {}
    inline void open_hole(uint32_t k_) { s->grow(); hole = s->head; s->head += 1 + k_; hole_end = s->head; }
    __attribute__((always_inline)) inline void put(uint32_t v) {
        uint32_t k_ = k >> 3; const uint32_t kr_ = kr >> 3;
        if (k_ > 0) {
            if (!inrun) { open_hole(k_); inrun = true; }
            if (v == 0) {
                if (++rl == (1u << k_)) {           // full window: flag 0 = the hole's first bit stays 0
                    if (s->head == hole_end) s->head = hole + 1;
                    rl = 0; k += UP_K; if (k > MAX_K) k = MAX_K; k_ = k >> 3; open_hole(k_);
                }
            } else {
                inrun = false; s->at(hole, int(1 + k_), (1ull << k_) | rl);
                k -= k > DOWN_K ? DOWN_K : k;
                --v; rl = 0; rice_put(*s, v, kr_); adapt_kr(kr, v, kr_);
            }
        } else {
            rl = 0; rice_put(*s, v, kr_); adapt_kr(kr, v, kr_);
            if (v == 0) { k += UP_K_DIRECT_RL; if (k > MAX_K) k = MAX_K; } else k -= k > DOWN_K_DIRECT_RL ? DOWN_K_DIRECT_RL : k;
        }
    }
    // an open run with rl > 0 gets its header; an empty open hole stays (the decoder laid it too)
    void flush() { if (inrun && rl > 0) { s->at(hole, int(1 + (k >> 3)), (1ull << (k >> 3)) | rl); rl = 0; } }
};

struct Decoder {
    LaneR* s; uint32_t rl = 0, k = 1 << 3, kr = 1 << 3; bool run_mode = false, fullrl = false; uint64_t hole = 0, hole_end = 0;
    explicit Decoder(LaneR* l) : s(l) {}
    inline void open_hole(uint32_t k_) { hole = s->head; s->head += 1 + k_; hole_end = s->head; }
    inline uint32_t read_rice() { const uint32_t kr_ = kr >> 3; const uint32_t v = rice_get(*s, kr_); adapt_kr(kr, v, kr_); return v; }
    inline uint32_t finish_partial_run() { k -= k > DOWN_K ? DOWN_K : k; run_mode = false; return read_rice() + 1; }
    inline uint32_t process_run() {
        if (fullrl) {
            if (--rl == 0) {                        // end of a full window: same give-back rule as the encoder
                if (s->head == hole_end) s->head = hole + 1;
                k += UP_K; if (k > MAX_K) k = MAX_K; open_hole(k >> 3); run_mode = false;
            }
            return 0;
        }
        if (rl == 0) return finish_partial_run();
        --rl; return 0;
    }
    __attribute__((always_inline)) inline uint32_t get() {
        if (run_mode) return process_run();
        const uint32_t k_ = k >> 3;
        if (k_ > 0) {
            if (!fullrl) open_hole(k_);             // a run starts (after a full window the hole is already open)
            const uint64_t h = s->peek_at(hole); run_mode = true; fullrl = !(h >> 63);
            if (fullrl) rl = 1u << k_;
            else { rl = uint32_t((h << 1) >> (64 - k_)); if (rl == 0) return finish_partial_run(); }
            return process_run();
        }
        const uint32_t v = read_rice();
        if (v == 0) { k += UP_K_DIRECT_RL; if (k > MAX_K) k = MAX_K; } else k -= k > DOWN_K_DIRECT_RL ? DOWN_K_DIRECT_RL : k;
        return v;
    }
};
}  // namespace rlgrl
