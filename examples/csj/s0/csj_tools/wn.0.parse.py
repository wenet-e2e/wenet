
# parse xml files and output simplified version

import xml.dom.minidom
import os
import sys
import multiprocessing

def parsexml(afile, outpath):
    outfile = os.path.join(outpath, afile.split('/')[-1] + '.simp')

    with open(outfile, 'w') as bw:
        domtree = xml.dom.minidom.parse(afile)
        collection = domtree.documentElement
        ipus = collection.getElementsByTagName('IPU')

        for ipu in ipus:
            starttime = 0
            endtime = 0
            if ipu.hasAttribute('IPUStartTime'):
                starttime = ipu.getAttribute('IPUStartTime')
            if ipu.hasAttribute('IPUEndTime'):
                endtime = ipu.getAttribute('IPUEndTime')

            # print('{}\t{}'.format(starttime, endtime))
            #  ## original format ###
            wlist = list()
            plainwlist = list()
            pronlist = list()

            #  ## pronunciation ###
            lemmalist = list()  # lemma list
            dictlemmalist = list()  # dict lemma list
            for suw in ipu.getElementsByTagName('SUW'):  # short unit word
                txt = ''
                plaintxt = ''
                # PhoneticTranscription
                prontxt = ''

                if suw.hasAttribute('OrthographicTranscription'):
                    txt = suw.getAttribute('OrthographicTranscription')
                if suw.hasAttribute('PlainOrthographicTranscription'):
                    plaintxt = suw.getAttribute('PlainOrthographicTranscription')
                if suw.hasAttribute('PhoneticTranscription'):
                    prontxt = suw.getAttribute('PhoneticTranscription')
                wlist.append(txt)
                plainwlist.append(plaintxt)
                pronlist.append(prontxt)

                lemma = ''
                dictlemma = ''

                if suw.hasAttribute('SUWLemma'):
                    lemma = suw.getAttribute('SUWLemma')
                if suw.hasAttribute('SUWDictionaryForm'):
                    dictlemma = suw.getAttribute('SUWDictionaryForm')
                lemmalist.append(lemma)
                dictlemmalist.append(dictlemma)
            txtsent = ' '.join(wlist)
            plaintxtsent = ' '.join(plainwlist)
            prontxtsent = ' '.join(pronlist)

            lemmasent = ' '.join(lemmalist)
            dictlemmasent = ' '.join(dictlemmalist)
            outrow = '{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                starttime, endtime, txtsent, plaintxtsent,
                prontxtsent, lemmasent, dictlemmasent)
            bw.write(outrow)

def procfolder_orig(apath, outpath):
    count = 0
    for afile in os.listdir(apath):
        if not afile.endswith('.xml'):
            continue
        afile = os.path.join(apath, afile)
        parsexml(afile, outpath)
        count += 1
        print('done: {} [{}]'.format(afile, count))

def procfolder(apath, outpath):
    # count = 0
    fnlist = list()
    for afile in os.listdir(apath):
        if not afile.endswith('.xml'):
            continue
        fnlist.append(afile)
    # now parallel processing:
    nthreads = 16
    for i in range(0, len(fnlist), nthreads):
        # fnlist[i, i+16]
        pool = multiprocessing.Pool(processes=nthreads)
        for j in range(nthreads):
            if i + j < len(fnlist):
                afile = os.path.join(apath, fnlist[i + j])
                pool.apply_async(parsexml, (afile, outpath))
        pool.close()
        pool.join()
    print('parallel {} threads done for {} files in total.'.format(
        nthreads, len(fnlist)))

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: {} <in.csj.path> <out.csj.path>".format(sys.argv[0]))
        exit(1)
    # e.g., csjpath='/workspace/asr/csj/'
    csjpath = sys.argv[1]
    outcsjpath = sys.argv[2]

    apath = os.path.join(csjpath, 'XML/BaseXML/core')
    apath2 = os.path.join(csjpath, 'XML/BaseXML/noncore')

    outapath = os.path.join(outcsjpath, 'xml')
    # create the "outapath" dir:
    if not os.path.exists(outapath):
        os.mkdir(outapath)

    # range over the following two folders:
    procfolder(apath, outapath)
    procfolder(apath2, outapath)
