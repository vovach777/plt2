[
('''    if (!g_sink->pair_mask) { qh_plane_enc(z, nullptr, nb, tw, c.off); return; }''','''    TS(2) if (!g_sink->pair_mask) { qh_plane_enc(z, nullptr, nb, tw, c.off); TS(3) return; }'''),
('''    if (!g_sink->pair_mask) { qh_plane_enc(S, a, nb, tw, c.off + PLR_UVOFF); return; }''','''    TS(4) if (!g_sink->pair_mask) { qh_plane_enc(S, a, nb, tw, c.off + PLR_UVOFF); TS(5) return; }'''),
('''        TileEnc te(1, size_t(g.tw(i)) * g.th(i)); te.sk[0].pair_mask = c.fast ? PLR_FAST_MASK : 0; g_sink = &te.sk[0];
        tile_enc(ti, W, g.tw(i), g.th(i), c, o, tmp); te.finish(o);''','''        TS(7) TileEnc te(1, size_t(g.tw(i)) * g.th(i)); te.sk[0].pair_mask = c.fast ? PLR_FAST_MASK : 0; g_sink = &te.sk[0]; TS(1)
        tile_enc(ti, W, g.tw(i), g.th(i), c, o, tmp); te.finish(o); TS(6)'''),
('''        rgb_split(rgb + (size_t(g.ty(i)) * W + g.tx(i)) * 3, W, g.tw(i), g.th(i), Y, U, V);
        TileEnc te(3, size_t(g.tw(i)) * g.th(i) * 3); for (auto& k : te.sk) k.pair_mask = c.fast ? PLR_FAST_MASK : 0;''','''        TS(7) rgb_split(rgb + (size_t(g.ty(i)) * W + g.tx(i)) * 3, W, g.tw(i), g.th(i), Y, U, V); TS(0)
        TileEnc te(3, size_t(g.tw(i)) * g.th(i) * 3); for (auto& k : te.sk) k.pair_mask = c.fast ? PLR_FAST_MASK : 0; TS(1)'''),
('''        te.finish(o);
        const uint32_t l = uint32_t(o.b.size() - b); memcpy(&o.b[lens + size_t(i) * 4], &l, 4);
    }
    return std::move(o.b);
}
void decompress_rgb''','''        te.finish(o); TS(6)
        const uint32_t l = uint32_t(o.b.size() - b); memcpy(&o.b[lens + size_t(i) * 4], &l, 4);
    }
    return std::move(o.b);
}
void decompress_rgb'''),
('''for (int k = 0; k < reps; ++k) { double a = now_s(); o =''','''for (int k = 0; k < reps; ++k) { g_t0 = __rdtsc(); double a = now_s(); o ='''),
('''double b = now_s();
                    if (rgb) decompress_rgb(o, r, w2, h2); else decompress(o, r, w2, h2); double e = now_s();''','''double b = now_s(); TS(7)
                    if (rgb) decompress_rgb(o, r, w2, h2); else decompress(o, r, w2, h2); double e = now_s(); TS(14)'''),
# decoder gray
('''        TileDec td(ti, 1); td.sr[0].pair_mask = c.fast ? PLR_FAST_MASK : 0; g_src = &td.sr[0];''','''        TS(14) TileDec td(ti, 1); td.sr[0].pair_mask = c.fast ? PLR_FAST_MASK : 0; g_src = &td.sr[0]; TS(8)'''),
('''        td.sr[1].pid = 1; td.sr[2].pid = 2; g_src = &td.sr[0]; tile_dec(ti, Y.data(), tw, tw, th, c);''','''        td.sr[1].pid = 1; td.sr[2].pid = 2; g_src = &td.sr[0]; TS(8) tile_dec(ti, Y.data(), tw, tw, th, c);'''),
('''        TileDec td(ti, 3); for (auto& k : td.sr) k.pair_mask = c.fast ? PLR_FAST_MASK : 0;''','''        TS(14) TileDec td(ti, 3); for (auto& k : td.sr) k.pair_mask = c.fast ? PLR_FAST_MASK : 0;'''),
('''
#if defined(PLR_VEC) && !defined(PLN_ROWS)
    recon_wave<false>(img, S, tw, th, z);''','''
    TS(9)
#if defined(PLR_VEC) && !defined(PLN_ROWS)
    recon_wave<false>(img, S, tw, th, z); TS(10)'''),
('''#if defined(PLR_VEC) && !defined(PLN_ROWS)
    recon_wave<true>(U, tw, tw, th, a); recon_wave<true>(V, tw, tw, th, b);''','''    TS(11)
#if defined(PLR_VEC) && !defined(PLN_ROWS)
    recon_wave<true>(U, tw, tw, th, a); recon_wave<true>(V, tw, tw, th, b); TS(12)'''),
('''        rgb_join(rgb.data() + (size_t(g.ty(i)) * W + g.tx(i)) * 3, W, tw, th, Y.data(), U.data(), V.data());''','''        TS(14) rgb_join(rgb.data() + (size_t(g.ty(i)) * W + g.tx(i)) * 3, W, tw, th, Y.data(), U.data(), V.data()); TS(13)'''),
]
