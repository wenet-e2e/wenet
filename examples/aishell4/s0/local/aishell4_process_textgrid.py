# -*- coding: utf-8 -*-
"""
Process the textgrid files
"""
import argparse
import codecs
from pathlib import Path
import textgrid


class Segment(object):
    def __init__(self, uttid, spkr, stime, etime, text):
        self.uttid = uttid
        self.spkr = spkr
        self.stime = round(stime, 2)
        self.etime = round(etime, 2)
        self.text = text


def get_args():
    parser = argparse.ArgumentParser(description="process the textgrid files")
    parser.add_argument("--path", type=str, required=True, help="Data path")
    args = parser.parse_args()
    return args


def main(args):
    wav_scp = codecs.open(Path(args.path) / "wav.scp", "r", "utf-8")
    textgrid_flist = codecs.open(
        Path(args.path) / "textgrid.flist", "r", "utf-8")
    # get the path of textgrid file for each utterance
    utt2textgrid = {}
    for line in textgrid_flist:
        path = Path(line.strip())
        # the name of textgrid file is different between training and test set
        if "train" in path.parts:
            uttid = "%s_%s" % (path.parts[-2], path.stem)
        else:
            uttid = path.stem
        utt2textgrid[uttid] = path
    # parse the textgrid file for each utterance
    all_segments = []
    for line in wav_scp:
        uttid = line.strip().split(" ")[0]
        if uttid not in utt2textgrid:
            print("%s doesn't have transcription" % uttid)
            continue
        segments = []
        tg = textgrid.TextGrid.fromFile(utt2textgrid[uttid])
        for i in range(tg.__len__()):
            for j in range(tg[i].__len__()):
                if tg[i][j].mark.strip():
                    segments.append(
                        Segment(
                            uttid,
                            tg[i].name,
                            tg[i][j].minTime,
                            tg[i][j].maxTime,
                            tg[i][j].mark.strip(),
                        ))

        segments = sorted(segments, key=lambda x: x.stime)
        all_segments += segments

    wav_scp.close()
    textgrid_flist.close()

    segments_file = codecs.open(Path(args.path) / "segments_all", "w", "utf-8")
    utt2spk_file = codecs.open(Path(args.path) / "utt2spk_all", "w", "utf-8")
    text_file = codecs.open(Path(args.path) / "text_all", "w", "utf-8")
    utt2dur_file = codecs.open(Path(args.path) / "utt2dur_all", "w", "utf-8")

    for i in range(len(all_segments)):
        utt_name = "%s-%s-%07d-%07d" % (
            all_segments[i].uttid,
            all_segments[i].spkr,
            all_segments[i].stime * 100,
            all_segments[i].etime * 100,
        )

        segments_file.write("%s %s %.2f %.2f\n" % (
            utt_name,
            all_segments[i].uttid,
            all_segments[i].stime,
            all_segments[i].etime,
        ))
        utt2spk_file.write(
            "%s %s-%s\n" %
            (utt_name, all_segments[i].uttid, all_segments[i].spkr))
        text_file.write("%s %s\n" % (utt_name, all_segments[i].text))
        utt2dur_file.write(
            "%s %.2f\n" %
            (utt_name, all_segments[i].etime - all_segments[i].stime))
        if len(all_segments[i].text) / (all_segments[i].etime -
                                        all_segments[i].stime) > 100:
            print(utt_name)
            print(
                len(all_segments[i].text) /
                (all_segments[i].etime - all_segments[i].stime))

    segments_file.close()
    utt2spk_file.close()
    text_file.close()
    utt2dur_file.close()


if __name__ == "__main__":
    args = get_args()
    main(args)
