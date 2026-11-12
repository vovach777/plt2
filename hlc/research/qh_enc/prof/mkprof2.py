# mkprof2.py - stage timers (rdtsc) for the patched palic2r.cpp -> prof/palic2r_prof2.cpp (profiling build only)
s=open('/tmp/fable_enc/palic-2/palic2r.cpp').read()
def rep(a,b):
    global s
    assert s.count(a)==1,(a,s.count(a)); s=s.replace(a,b)
rep('namespace palic2 {\n','namespace palic2 {\n#include <x86intrin.h>\nstatic uint64_t g_t[16]; static const char* g_tn[16]={"rgb_split","tile ctor","Y resid","Y blocks","UV resid+S","UV blocks","finish","rest"};\n#define TS(k) { const uint64_t t_=__rdtsc(); g_t[k]+=t_-g_t0; g_t0=t_; }\nstatic uint64_t g_t0;\n')
rep('''    if (!g_sink->pair_mask) { qh_plane_enc(z, nullptr, nb, tw, c.off); return; }''','''    TS(2) if (!g_sink->pair_mask) { qh_plane_enc(z, nullptr, nb, tw, c.off); TS(3) return; }''')
rep('''    if (!g_sink->pair_mask) { qh_plane_enc(S, a, nb, tw, c.off + PLR_UVOFF); return; }''','''    TS(4) if (!g_sink->pair_mask) { qh_plane_enc(S, a, nb, tw, c.off + PLR_UVOFF); TS(5) return; }''')
rep('''        TileEnc te(1, size_t(g.tw(i)) * g.th(i)); te.sk[0].pair_mask = c.fast ? PLR_FAST_MASK : 0; g_sink = &te.sk[0];
        tile_enc(ti, W, g.tw(i), g.th(i), c, o, tmp); te.finish(o);''','''        TS(7) TileEnc te(1, size_t(g.tw(i)) * g.th(i)); te.sk[0].pair_mask = c.fast ? PLR_FAST_MASK : 0; g_sink = &te.sk[0]; TS(1)
        tile_enc(ti, W, g.tw(i), g.th(i), c, o, tmp); te.finish(o); TS(6)''')
rep('''        rgb_split(rgb + (size_t(g.ty(i)) * W + g.tx(i)) * 3, W, g.tw(i), g.th(i), Y, U, V);
        TileEnc te(3, size_t(g.tw(i)) * g.th(i) * 3); for (auto& k : te.sk) k.pair_mask = c.fast ? PLR_FAST_MASK : 0;''','''        TS(7) rgb_split(rgb + (size_t(g.ty(i)) * W + g.tx(i)) * 3, W, g.tw(i), g.th(i), Y, U, V); TS(0)
        TileEnc te(3, size_t(g.tw(i)) * g.th(i) * 3); for (auto& k : te.sk) k.pair_mask = c.fast ? PLR_FAST_MASK : 0; TS(1)''')
rep('''        te.finish(o);
        const uint32_t l = uint32_t(o.b.size() - b); memcpy(&o.b[lens + size_t(i) * 4], &l, 4);
    }
    return std::move(o.b);
}
void decompress_rgb''','''        te.finish(o); TS(6)
        const uint32_t l = uint32_t(o.b.size() - b); memcpy(&o.b[lens + size_t(i) * 4], &l, 4);
    }
    return std::move(o.b);
}
void decompress_rgb''')
rep('''for (int k = 0; k < reps; ++k) { double a = now_s(); o =''','''for (int k = 0; k < reps; ++k) { g_t0 = __rdtsc(); double a = now_s(); o =''')
rep('''double b = now_s();
                    if (rgb)''','''double b = now_s(); TS(7)
                    if (rgb)''')
rep('''            printf("TOTAL %zu files''','''            { uint64_t tt=0; for (int k=0;k<8;++k) tt+=g_t[k]; for (int k=0;k<8;++k) printf("  %-12s %5.1f%%  %6.2f cyc/px\\n", g_tn[k], 100.0*g_t[k]/tt, double(g_t[k])/reps/tpix); printf("  total %.2f cyc/px\\n", double(tt)/reps/tpix); }
            printf("TOTAL %zu files''')
s=s.replace('"../palic-1/netpbm.hpp"','"/tmp/fable_enc/palic-1/netpbm.hpp"').replace('#include "qh.hpp"','#include "/tmp/fable_enc/palic-2/qh.hpp"').replace('#include "rlgr_lane.hpp"','#include "/tmp/fable_enc/palic-2/rlgr_lane.hpp"').replace('#include "rlgr_fast.hpp"','#include "/tmp/fable_enc/palic-2/rlgr_fast.hpp"')
open('/tmp/fable_enc/prof/palic2r_prof2.cpp','w').write(s)
