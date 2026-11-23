#!/usr/bin/env python3
"""Materialise ALL images (train+hold+test) of the six screen groups + the photo group
as PPM into /tmp/r8f5/sets/<group>/all, with numeric prefixes that force the exact
canonical order WorkBuddy used (path order for desktop/scans, base-name order for the rest).
Also emits split lists: <group>/{train,hold,test}.txt and photo_hold.txt.
"""
import os, sys, tarfile, shutil, json
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

DS = '/home/devin/train/datasets'
OUT = '/tmp/r8f5/sets'

GROUPS = {
    # name: (folder, ext, order)
    'ios':      ('screens_ios1k', '.png', 'base'),
    'desktop':  ('screens_screenspot_pro', '.png', 'path'),
    'websight': ('screens_websight_v02_png800', '.png', 'base'),
    'docs':     ('docs_doclaynet_small', '.png', 'base'),
    'scans':    ('scans_dibco_2016_2019', '.bmp', 'path'),
}
WEB_PAGES = ['amazon.com', 'apple.com', 'cnn.com', 'creativecommons.org',
             'imdb.com', 'microsoft.com', 'nytimes.com', 'reddit.com']

def conv(job):
    src, dst = job
    Image.open(src).convert('RGB').save(dst, 'PPM')
    return dst

def main():
    jobs = []
    manifest = {}
    for g, (folder, ext, order) in GROUPS.items():
        tf = tarfile.open(os.path.join(DS, folder, folder + '.tar'))
        ms = [m for m in tf.getmembers() if m.isfile() and m.name.lower().endswith(ext)]
        ms.sort(key=(lambda m: os.path.basename(m.name)) if order == 'base' else (lambda m: m.name))
        cd = os.path.join(OUT, g); alld = os.path.join(cd, 'all')
        shutil.rmtree(cd, ignore_errors=True); os.makedirs(alld)
        src = os.path.join(cd, 'src'); os.makedirs(src)
        names = []
        for i, m in enumerate(ms):
            sp = os.path.join(src, '%05d%s' % (i, ext))
            with tf.extractfile(m) as f, open(sp, 'wb') as o: shutil.copyfileobj(f, o)
            nm = '%04d__%s.ppm' % (i, os.path.basename(m.name).rsplit('.', 1)[0].replace('/', '_'))
            names.append(nm)
            jobs.append((sp, os.path.join(alld, nm)))
        manifest[g] = names
        tf.close()
        print(g, len(names), order, flush=True)
    # web = 8 pages + 4 gb82
    cd = os.path.join(OUT, 'web'); alld = os.path.join(cd, 'all')
    shutil.rmtree(cd, ignore_errors=True); os.makedirs(alld)
    webs = [p + '.png' for p in WEB_PAGES] + ['codec_wiki.ppm', 'gmessages.ppm', 'graph.ppm', 'gui.ppm']
    webs.sort()
    for i, f in enumerate(webs):
        srcf = os.path.join(DS, 'screenshot_web', f) if f.endswith('.png') else os.path.join(DS, 'train_gb82-sc_p6', f)
        nm = '%04d__%s.ppm' % (i, f.rsplit('.', 1)[0])
        jobs.append((srcf, os.path.join(alld, nm)))
    manifest['web'] = ['%04d__%s.ppm' % (i, f.rsplit('.', 1)[0]) for i, f in enumerate(webs)]
    print('web', len(webs), 'base', flush=True)
    # photo: clic2020_professional_train p6 tar (already PPM)
    cd = os.path.join(OUT, 'photo'); alld = os.path.join(cd, 'all')
    shutil.rmtree(cd, ignore_errors=True); os.makedirs(alld)
    tf = tarfile.open(os.path.join(DS, 'clic2020_professional_train', 'professional_train_2020_p6.tar'))
    ms = [m for m in tf.getmembers() if m.isfile() and m.name.lower().endswith('.ppm')]
    ms.sort(key=lambda m: os.path.basename(m.name))
    for i, m in enumerate(ms):
        nm = '%04d__%s.ppm' % (i, os.path.basename(m.name).rsplit('.', 1)[0])
        with tf.extractfile(m) as f, open(os.path.join(alld, nm), 'wb') as o: shutil.copyfileobj(f, o)
    manifest['photo'] = ['%04d__%s.ppm' % (i, os.path.basename(m.name).rsplit('.', 1)[0]) for i, m in enumerate(ms)]
    tf.close()
    print('photo', len(ms), flush=True)

    with ProcessPoolExecutor(max_workers=10) as ex:
        n = sum(1 for _ in ex.map(conv, jobs))
    print('converted', n, flush=True)

    # splits
    for g, names in manifest.items():
        if g == 'photo':
            hold = [nm for i, nm in enumerate(names) if i % 10 == 9]
            open(os.path.join(OUT, 'photo_hold.txt'), 'w').write('\n'.join(hold) + '\n')
            print('photo hold', len(hold)); continue
        tr, ho, te = [], [], []
        for i, nm in enumerate(names):
            if g == 'web':
                (ho if i == 0 else tr).append(nm)
            elif i % 5 == 4: te.append(nm)
            elif i % 10 == 3: ho.append(nm)
            else: tr.append(nm)
        d = os.path.join(OUT, g)
        open(os.path.join(d, 'train.txt'), 'w').write('\n'.join(tr) + '\n')
        open(os.path.join(d, 'hold.txt'), 'w').write('\n'.join(ho) + '\n')
        open(os.path.join(d, 'test.txt'), 'w').write('\n'.join(te) + '\n')
        print('%s train=%d hold=%d test=%d' % (g, len(tr), len(ho), len(te)))

if __name__ == '__main__':
    main()