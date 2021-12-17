import os
import sys

# train test1 test2 test3

def readtst(tstfn):
    outlist = list()
    with open(tstfn) as br:
        for aline in br.readlines():
            aline = aline.strip()
            outlist.append(aline)
    return outlist

def split_train_tests_xml(xmlpath, test1fn, test2fn, test3fn):
    test1list = readtst(test1fn)
    test2list = readtst(test2fn)
    test3list = readtst(test3fn)

    outtrainlist = list()  # full path ".xml.simp" files
    outt1list = list()  # test 1, full path ".xml.simp" files
    outt2list = list()
    outt3list = list()

    for afile in os.listdir(xmlpath):
        if not afile.endswith('.xml.simp'):
            continue
        afile2 = xmlpath + '/' + afile
        aid = afile.split('.')[0]
        if aid in test1list:
            outt1list.append(afile2)
        elif aid in test2list:
            outt2list.append(afile2)
        elif aid in test3list:
            outt3list.append(afile2)
        else:
            outtrainlist.append(afile2)

    return outtrainlist, outt1list, outt2list, outt3list

def all_wavs(wavpath):
    wavlist = list()
    for afile in os.listdir(wavpath):
        if not afile.endswith('.wav'):
            continue
        afile2 = wavpath + '/' + afile
        wavlist.append(afile2)
    return wavlist

def gen_text(xmllist, outpath):
    # id \t text
    # e.g., /workspace/asr/wenet/examples/csj/s0/data/xml/S11M1689.xml.simp
    # ID = S11M1689_stime_etime
    outtxtfn = os.path.join(outpath, 'text')
    with open(outtxtfn, 'w') as bw:
        for xmlfn in xmllist:
            aid = xmlfn.split('/')[-1]
            aid2 = aid.split('.')[0]

            with open(xmlfn) as br:
                for aline in br.readlines():
                    aline = aline.strip()
                    # stime \t etime \t text1 \t text2 \t text3 \t text4 \t text5
                    cols = aline.split('\t')
                    # TODO different between "< 7" and "< 4"? strange
                    # -> use "< 4", DO NOT use "< 7" !
                    if len(cols) < 4:
                        continue

                    stime = cols[0]
                    etime = cols[1]
                    atxt = cols[3].replace(' ', '')

                    afullid = '{}_{}_{}'.format(aid2, stime, etime)
                    aoutline = '{}\t{}\n'.format(afullid, atxt)
                    bw.write(aoutline)

def parse_xml_set(xmllist):
    outset = set()
    for xml in xmllist:
        aid = xml.split('/')[-1]
        aid2 = aid.split('.')[0]
        outset.add(aid2)
    return outset

def gen_wav_scp(xmllist, wavlist, outpath):
    # xmlset = pure id set, alike 'S04F1228'
    # can be from train, test1, test2, or test3
    xmlset = parse_xml_set(xmllist)

    outwavscpfn = os.path.join(outpath, 'wav.scp')
    with open(outwavscpfn, 'w') as bw:
        for wav in wavlist:
            # wav is alike "/workspace/asr/wenet/examples/csj/s0/data
            # /wav/S04F1228.wav_00458.875_00459.209.wav"
            aid = wav.split('/')[-1]
            cols = aid.split('_')

            aid2 = cols[0].split('.')[0]
            if aid2 not in xmlset:
                continue

            stime = cols[1]
            etime = cols[2].replace('.wav', '')

            afullid = '{}_{}_{}'.format(aid2, stime, etime)

            wavabspath = os.path.abspath(wav)
            aoutline = '{}\t{}\n'.format(afullid, wavabspath)
            bw.write(aoutline)


def prep_text_wavscp(
        xmlpath, wavpath, test1fn, test2fn, test3fn,
        outtrainpath, out1path, out2path, out3path):

    trainlist, t1list, t2list, t3list = split_train_tests_xml(
        xmlpath,
        test1fn,
        test2fn,
        test3fn)
    wavlist = all_wavs(wavpath)

    gen_text(trainlist, outtrainpath)
    gen_text(t1list, out1path)
    gen_text(t2list, out2path)
    gen_text(t3list, out3path)

    gen_wav_scp(trainlist, wavlist, outtrainpath)
    gen_wav_scp(t1list, wavlist, out1path)
    gen_wav_scp(t2list, wavlist, out2path)
    gen_wav_scp(t3list, wavlist, out3path)

if __name__ == '__main__':
    if len(sys.argv) < 10:
        print(
            "Usage: {}".format(sys.argv[0]) + "<xmlpath> " +
            "<wavpath> <test1fn> <test2fn> <test3fn> " +
            "<outtrainpath> <out1path> <out2path> <out3path>")
        exit(1)

    xmlpath = sys.argv[1]
    wavpath = sys.argv[2]
    test1fn = sys.argv[3]
    test2fn = sys.argv[4]
    test3fn = sys.argv[5]

    outtrainpath = sys.argv[6]
    out1path = sys.argv[7]
    out2path = sys.argv[8]
    out3path = sys.argv[9]

    prep_text_wavscp(xmlpath, wavpath, test1fn,
                     test2fn, test3fn, outtrainpath,
                     out1path, out2path, out3path)
