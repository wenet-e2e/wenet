
README File for "sph2pipe"
--------------------------

1. Introduction

The "sph2pipe" program was created by the Linguistic Data Consortium
to provide greater flexibility and ease of use for SPHERE-formatted
digital audio data.  It is equivalent in most respects to the related
utility "sph_convert", but each of these tools provides some abilities
that the other does not.  Here is a brief summary of the similarities
and differences.

Both sph_convert and sph2pipe will:

 - work on all Microsoft Windows systems, via the "MS-DOS" command
   line prompt
 - read any SPHERE-formatted data file and convert it to Microsoft
   RIFF ("WAV") format, Sun/Java AU format, MAC AIFF format or raw
   (headerless) format
 - automatically uncompress SPHERE files that have been compressed
   using the "shorten" algorithm (often used in LDC speech corpora)
 - allow demultiplexing of two-channel waveform data, to output one
   or the other channel alone
 - allow conversion of the sample data to 16-bit linear PCM or to
   8-bit mu-law encoding, regardless of the input sample encoding

Only sph_convert can:

 - run on older (pre-OSX) Macintosh systems, via the old Mac-style GUI
 - do multiple file conversions in a single run (sph2pipe only does
   one file at a time); there are two methods for doing "batches":
      * treat all files in a chosen directory that match a
        user-specified file-name pattern, or 
      * treat all files in all subdirectories under a chosen
        base directory 
 - in either case, convert all SPHERE files and copy (or bypass) all
   non-SPHERE files 

Only sph2pipe can:

 - run on UNIX systems (should also work on MacOS X, via its unix
   shell/command-line interface, using the "Terminal" utility)
 - provide SPHERE-formatted output as well as RIFF, AU, AIFF and raw
 - handle raw sample data as input, using a SPHERE header stored in a
   separate file.
 - trim off the beginning and/or end of the input data, to output just
   a user-specified segment based on either time or sample offsets
   (sph_convert always outputs the entire file)
 - write the output data to stdout, for redirection to any named file,
   or to a pipeline process (sph_convert always writes the data to a
   new file, with a name derived automatically from the input file)
 - support input and output of A-law speech data

When installed on MS Windows or MacOS, these tools will produce RIFF
output files by default; when compiled for UNIX systems (Linux, Solaris,
etc), sph2pipe will output SPHERE format by default.  In any case, the
user has the option to specify what format is desired -- any machine can
be used to generate any kind of output.  (Well, a Mac that is running
OS-9 or older cannot produce SPHERE output, but we haven't heard any
requests for that...)

Sph2pipe will not work on older Mac systems because the notion of a
pipeline command did not exist on Macs prior to OS X.  Of course, it is
possible to create custom-edited RIFF/AIFF/AU/raw files using sph2pipe
on unix or wintel, then copy those files to an older Mac; but the
combination of sph_convert and any of several waveform editing tools for
Macs can provide all the functionality of sph2pipe, and then some.

The "shorten" speech compression technique, used in the LDC's
publication of many speech corpora, was developed by Tony Robinson,
originally at Cambridge University; "shorten" is available from
SoftSound, Inc. (http://www.softsound.com/Shorten.html).  The algorithm
and source code for uncompressing "shortened" speech data are included
here by permission of Tony Robinson and SoftSound, Inc.

People who have used the original "shorten" package (dating from the
mid-1990's) will find that sph2pipe is more much flexible, because of
the range of options available for controlling output.  UNIX users who
are familiar with the NIST SPHERE utilities "w_decode" and "w_edit"
will find that sph2pipe runs faster and is easier to use, especially
when extracting a subset of data from a compressed file: in this case
sph2pipe alone handles a job that would require both w_decode and
w_edit, and works a lot quicker (and also avoids a nasty bug in the
sphere_2.6a package that can arise when you try to run w_decode and
w_edit together in a pipeline).

Note that sph2pipe and sph_convert are NOT able to do sample-rate
conversion.  If you have a need for this, try the "SoX" package -- see
under "Licensing" below for more information about SoX.


2. Installation

Wintel users can simply download the executable file (sph2pipe.exe) that
has been precompiled for MS Windows/DOS systems, and start using it.
(You can download the source files too, if you have your own C compiler
and want to customize the program for your needs.)  UNIX and MacOS X
users are advised to compile the program from the source code.

To build from sources, download "sph2pipe_v2.4.tgz", and do this:

 -- if you have the Gnu version of tar (standard on linux):

     tar xzf sph2pipe_v2.4.tgz

 -- otherwise (with Wintel systems or non-Gnu versions of tar):

     gzip -c -d sph2pipe_v2.4.tgz | tar xf -

 -- then:

     cd sph2pipe_v2.4

     gcc -o sph2pipe *.c -lm     ## on unix
 or
     gcc -o sph2pipe.exe *.c -lm ## on wintel, using the djgpp compiler

That's it -- no configuration scripts, makefiles or special libraries
are needed (the source code consists of just 3 *.c files, and 3 *.h
files; the standard math library is needed for compilation).  Put the
resulting "sph2pipe" executable in your path and start using it.  If you
don't have gcc, try whatever C compiler you do have; you might need to
change a few details in sph_convert.h, but we hope the code is generic
enough (POSIX compliant) to work anywhere.


3. Usage

The command line syntax is:

 sph2pipe [-h hdr] [-t|-s b:e] [-c 1|2] [-p|-u|-a] [-f typ] infile [outfile]

   -h hdr -- treat the input file as raw (headerless) sample data, and
         read header information from a separate file, given as the
         "hdr" argument; the "hdr" must contain a valid SPHERE header
         that correctly describes the nature of the input sample data
         ("hdr" may contain actual sample data as well, which will be
         ignored).  If the output format is "sph", the SPHERE header
         in "hdr" will be written first, with appropriate adjustments
         where needed.  (When this option is not used, "input" must
         begin with a valid SPHERE header.)

   -t b:e -- output only the portion of waveform data that lies
             between the stated beginning and ending points, given in
             seconds, as positive real numbers; "b" defaults to
             start-of-file, "e" defaults to end-of-file -- so the
             following usages are valid:

	     "-t :10.05"  (output first 10.05 sec, skip the rest)
	     "-t 4:"      (skip first 4 sec, output the rest)
	     "-t 4:10.05  (output 6.05 sec, starting at 4 sec in)

   -s b:e -- output only the portion of waveform data that lies
	     between the stated beginning and ending points, given in
	     samples as positive integers; "b" defaults to
	     start-of-file, "e" defaults to end-of-file -- so the
	     following usages are valid:

	     "-s :32000"    (output first 32K samples, skip the rest)
	     "-s 8000:"     (skip first 8K samples, output the rest)
	     "-s 8000:32000 (output 24K samples, starting at 8K in)

   -c 1 or -c 2 -- output only the first or second channel, in case
	           input is two-channel (has no effect if input is
	           single channel); default is to output all channels

   -p -- force 16-bit PCM output, in case input is something else (has
         no effect if input is already 16-bit PCM)

   -u -- force 8-bit mu-law output, in case input is 16-bit pcm (has
	 no effect if input is already mu-law)

   -a -- force 8-bit a-law output, in case input is 16-bit pcm (has
	 no effect if input is already a-law)

	 The -p, -u and -a options are ignored if "-f aif" is used,
	 because AIFF only supports PCM samples.  When none of these
	 three is specified, the default behavior is to leave original
	 sample format "as is" (or to force PCM if using "-f aif")

   -f fmt -- selects the output header format; "fmt" can be:
	 rif (or wav) -- default for Wintel & Mac systems
	 aif (or mac) -- similar to rif, but more Mac-ish...
	 sph -- SPHERE format, default on unix systems
	 au  -- common on Sun/Java/Next
	 raw -- i.e. headerless

If only one file name is given on the command line, output is written
to stdout (i.e. for redirection via "> output.file", or for input to a
pipeline).  If a second file name is given, output is written directly
to a file with this name, and not to stdout; if the named output file
already exists and contains data, its contents will be overwritten
(replaced) by the sph2pipe output.

If the output format is RIFF, AU, AIFF or SPH, a fully specified and
correct file header is written first (*).  When writing via stdout to
a pipeline, a downstream process can behave exactly as it would for a
valid disk file in the target format (except that "seek()" does not
work on stdin, of course).

(*) Note: for SPHERE-formatted output, sph2pipe will eliminate the
"sample_checksum" field, since this cannot be given a correct value
prior to processing and writing the output data.  Also, when
converting PCM input to mu-law or a-law, sph2pipe removes the
"sample_byte_format" header field, which defines the byte order for
16-bit sample data.  Apart from these two circumstances, the output
sphere header retains all information in the original input header,
along with appropriate changes, where necessary, to the sample_count,
channel_count, sample_coding, sample_n_bytes, sample_byte_format and
sample_sig_bits fields, making the header information consistent with
the data being written.

A useful benefit provided by pipeline operation is the ability to
"compose" a single output file by concatenating any number of input
files, or pieces of one or more input files.  For instance, to combine
all the speech data in one directory into a single file for signal
analysis (using bash as the command-line shell, which is available for
wintel systems as well as for unix):

   $ for i in *.sph; do
   > sph2pipe -f raw $i >> allsph.raw
   > done

Or, to put together a set of excerpts that you want to play back
during your next PowerPoint presentation:

   sph2pipe -f raw -t 0:1 empty.sph > silence.raw
   sph2pipe -f sph -t 0:1 empty.sph > slideshow.sph
   sph2pipe -f raw -t 15.5:18.2 example1.sph >> slideshow.sph
   cat silence.raw >> slideshow.sph
   sph2pipe -f raw -t 300:305.5 example2.sph >> slideshow.sph
   cat silence.raw >> slideshow.sph
   sph2pipe -f raw -t 1832:1838 example3.sph >> slideshow.sph
   cat silence.raw >> slideshow.sph
   ...
   sph2pipe -f wav slideshow.sph > slideshow.wav

Note the use of "raw" format to concatenate waveform data (we don't
want file headers to be interspersed with the speech).  Also, in the
second example, the sphere header that is initially created for
"slideshow.sph" will be "numerically" correct only in reference to the
initial one-second chunk; as more segments are appended to this file,
the "sample_count" field in the header will be further and further
from the truth.  But this doesn't matter -- at the final stage, when
this file is converted to RIFF, sph2pipe will notice the discrepancy
between the "sample_count" value in the header and the actual size of
the file, and will automatically correct the sample_count to be
consistent with the file size.

There are important rules to follow when combining segments from
multiple files.  If you happen to violate any of these rules, the
resulting output will certainly come out sounding wrong (sometimes
painfully so):

(1) be sure that all the input files have the same sampling rate.
(2) be sure to append data using a consistent number of channels,
       always a single channel, or always two channels
(3) it's a good idea to specify "-p" on all runs -- or "-u" or "-a" on
       all runs -- to guarantee that the output file will have the same
       sample coding throughout, no matter what the original sample
       codings may have been in the source files

When combining data from files in any single LDC corpus, these issues
normally won't pose any problem: within a given corpus, all files tend
to have the same properties.


4. Version specific information

This version will only convert one sphere file in one run, and must
read that file directly from disk or cdrom (it does not accept input
via stdin, because it must be able to do "fseek()" on the input file).
Handling bunches of files is easily done on both unix and wintel
systems using generic tools like the unix "bash" shell, the unix
"find" utility, and/or the Perl or Python scripting languages; fully
capable ports of all these tools are available for wintel systems.

 Version History:

 - Version 2.0 was the first "public" release; it did not support a-law
 sample coding, AU or AIFF output formats, the "-h hdrfile" option, or
 the "-s|-t bgn:end" options.  It contained a significant bug that arose
 when converting some 16-bit PCM sphere files to ulaw output.

 - Version 2.1 provided a fix for the pcm-to-ulaw bug.

 - Version 2.2 added the options for AU and AIFF output formats.

 - Version 2.3 added the "-s|-t" options to select regions for output
 based on sample or time offsets, and also added the "-h" option for
 using "stand-off" sphere headers with raw sample data files.

 - Version 2.4 added support for a-law sample coding, and added a
 thorough test suite, allowing end users to verify their installation;
 there were some minor bug fixes involving the "-h" option; the README
 file has also been revised to bring various URL's up to date.

 - Version 2.5 added the ability to include an output file name as a
 command line argument; this was done to avoid concerns on MS-Windows
 systems about some command-line shells that impose "text-mode"
 alterations to data when running commands with redirection or pipes.


5. License

Various portions of source code from Tony Robinson's "shorten-2.0"
package are used here by permission of Tony Robinson and SoftSound,
Inc. <http://www.softsound.com> -- these portions are found in the file
"shorten_x.c"; please note the copyright information in that file.  By
agreement with Tony Robinson and SoftSound, Inc, the Linguistic Data
Consortium (LDC) grants permission to copy and use this software for the
purpose of reading "shorten"-compressed speech data provided in NIST
SPHERE file format by the LDC or others.  SoftSound provides useful
tools for audio compression and other signal processing tasks.

Other portions of source code (in particular the "writeRIFFHeader" and
"writeAIFFHeader" functions in "file_headers.c", and the "alaw2pcm"
conversion function) were adapted from the "SoX" package, a valuable
open-source tool maintained primarily by Chris Bagwell, with assistance
from many others (http://sox.sourceforge.net/).  We gratefully
acknowledge the value provided by all contributors to SoX; sph2pipe
would have been much harder to write without this resource.  We
recommend that you use SoX if you need to do sample-rate conversion on
audio data.

