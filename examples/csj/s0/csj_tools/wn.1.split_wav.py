# based on xml.simp -> start_time and end_time -> split using sox

import os
import sys
import multiprocessing

import librosa
import soundfile as sf

# use .simp as the source for .wav file splitting
def wavfn(apath):
    wavdict = dict()  # key=id, value=full.path of .wav
    for awavfn in os.listdir(apath):
        fullwavpath = os.path.join(apath, awavfn)
        aid = awavfn.replace('.wav', '')
        wavdict[aid] = fullwavpath
    return wavdict

def xmlfn(apath):
    xmldict = dict()  # key=id, value=full.path of .xml.simp
    for axmlfn in os.listdir(apath):
        if not axmlfn.endswith('.xml.simp'):
            continue
        axmlfn2 = os.path.join(apath, axmlfn)
        aid = axmlfn.replace('.xml.simp', '')
        # print('obtain id: {}\t{}'.format(axmlfn, aid))
        xmldict[aid] = axmlfn2
    return xmldict

def ch2to1(f1, outf1):
    wav1, _ = librosa.load(f1, sr=16000, mono=False)
    if wav1.ndim == 1:
        return
    wav1mono = librosa.to_mono(wav1)
    sf.write(outf1, wav1mono, 16000)
    # print('2ch to 1ch, {} -> {}'.format(f1, outf1))
    acmd = 'mv {} {}'.format(outf1, f1)
    res = os.system(acmd)
    # rename the .1ch file back to the .wav file and
    # overwrite the old .wav file which is 2ch
    # print(res, acmd)

def proc1file(fullxmlfn, fullwavfn, outwavpath):
    with open(fullxmlfn) as xmlbr:
        for axmlline in xmlbr.readlines():
            # start.time end.time ortho plainortho phonetic
            axmlline = axmlline.strip()
            cols = axmlline.split('\t')
            stime = cols[0]
            etime = cols[1]

            if len(cols) == 2:
                continue  # skip

            basename = fullwavfn.split('/')[-1]

            name2 = '{}_{}_{}.wav'.format(basename, stime, etime)
            partwavfn = os.path.join(outwavpath, name2)

            dur = float(etime) - float(stime)
            acmd = 'sox {} {} trim {} {}'.format(fullwavfn, partwavfn, stime, dur)
            res = os.system(acmd)
            # print(res, acmd)

            # perform 2ch to 1ch if necessary!
            partwavfn1ch = partwavfn + ".1ch.wav"  # NOTE must ends with '.wav'!
            # otherwise, soundfile.write will give us error report!
            ch2to1(partwavfn, partwavfn1ch)

def procpath(atag, csjpath, xmlsimppath, outwavpath, idset):
    # atag = 'core' and 'noncore'
    axmlpath = xmlsimppath
    awavpath = os.path.join(csjpath, atag)

    xmldict = xmlfn(axmlpath)
    wavdict = wavfn(awavpath)

    wavidlist = list(wavdict.keys())

    # parallel processing
    nthreads = 16
    for i in range(0, len(wavidlist), nthreads):
        pool = multiprocessing.Pool(processes=nthreads)
        for j in range(nthreads):
            if i + j < len(wavidlist):
                wavid = wavidlist[i + j]
                if len(idset) > 0 and wavid not in idset:
                    # when idset is not empty, then only process the ids
                    # that are included in idset:
                    continue

                fullwavfn = wavdict[wavid]
                if wavid in xmldict:
                    fullxmlfn = xmldict[wavid]
                    pool.apply_async(proc1file, (fullxmlfn, fullwavfn, outwavpath))
        pool.close()
        pool.join()

    print('parallel {} threads done for {} files.'.format(
        nthreads,
        len(wavidlist)))

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print(
            "Usage: {}".format(sys.argv[0]) +
            "<in.csj.path> <in.xml.simp.path> <out.wav.path> [id.list.fn]")
        exit(1)

    csjpath = sys.argv[1]
    xmlsimppath = sys.argv[2]
    outwavpath = sys.argv[3]
    idlistfn = sys.argv[4] if len(sys.argv) == 5 else ""
    idset = set()
    if len(idlistfn) > 0:
        with open(idlistfn) as br:
            for aline in br.readlines():
                aline = aline.strip()
                idset.add(aline)
    print(idset)

    for atag in ['core', 'noncore']:
        procpath(atag, csjpath, xmlsimppath, outwavpath, idset)
