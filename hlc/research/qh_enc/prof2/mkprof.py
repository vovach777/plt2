# mkprof.py SRC OUT - rdtsc stage timers (encoder and decoder) for palic2r.cpp (profiling build only; best-effort anchors)
import sys
src, out = sys.argv[1], sys.argv[2]; d = src.rsplit('/', 1)[0]
s = open(src).read(); miss = []
def rep(a, b, need=True):
    global s
    if s.count(a) != 1:
        if need: miss.append(a[:60])
        return
    s = s.replace(a, b)
N = ['e rgb_split/front', 'e ctor', 'e Y resid', 'e Y blocks', 'e UV resid+S', 'e UV blocks', 'e finish', 'rest', 'd ctor', 'd Y blocks', 'd Y recon', 'd UV blocks', 'd UV recon', 'd rgb_join', 'd rest']
rep('namespace palic2 {\n', 'namespace palic2 {\n#include <x86intrin.h>\nstatic uint64_t g_t[16]; static const char* g_tn[16]={' + ','.join('"%s"' % x for x in N) + '};\n#define TS(k) { const uint64_t t_=__rdtsc(); g_t[k]+=t_-g_t0; g_t0=t_; }\nstatic uint64_t g_t0;\n')
for a, b in eval(open(sys.argv[3]).read()): rep(a, b)
rep('''            printf("TOTAL %zu files''', '''            { uint64_t te_=0, td_=0; for (int k=0;k<8;++k) te_+=g_t[k]; for (int k=8;k<15;++k) td_+=g_t[k];
              for (int k=0;k<15;++k) if (g_t[k]) printf("  %-14s %5.1f%%  %6.2f cyc/px\\n", g_tn[k], 100.0*g_t[k]/(k<8?te_:td_), double(g_t[k])/reps/tpix);
              printf("  enc total %.2f  dec total %.2f cyc/px\\n", double(te_)/reps/tpix, double(td_)/reps/tpix); }
            printf("TOTAL %zu files''')
for h in ('qh.hpp', 'rlgr_lane.hpp', 'rlgr_fast.hpp'): s = s.replace('#include "%s"' % h, '#include "%s/%s"' % (d, h))
s = s.replace('"../palic-1/netpbm.hpp"', '"%s/../palic-1/netpbm.hpp"' % d)
open(out, 'w').write(s)
if miss: print('MISSING anchors:', miss)
