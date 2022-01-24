/*************************************************************
 * Source File:	sph2pipe.c
 * Compilation:	gcc -o sph2pipe sph2pipe.c shorten_x.c file_headers.c
 * Authors:	Dave Graff, Willie Dong; LDC, University of Pennsylvania
 * Purpose:	multi-platform utility for converting SPHERE waveform files
 *		to other common digital audio file formats
 *
 * Usage:  sph2pipe [-f sph|wav|raw] [-t bsec:esec | -s bsamp:esamp] 
 *                [-h hdrfile] [-c 1|2] [-p|-u|-a] infile [outfile]
 *
 * The NIST "SPHERE" file format for waveform data consists of a plain-text
 * header that describes the file contents, followed by the raw (binary)
 * sample data; the size of the sphere header is always a multiple of 1024
 * bytes, and is always stated as an ASCII digit string in the second line
 * of text (bytes 8-15 of the file); the description of content always
 * includes the following elements, though not in any specified order:
 *  - sample rate
 *  - sample count
 *  - channel count
 *  - bytes per sample
 *  - byte order (when bytes per sample is > 1)
 *  - sample coding -- one of: mulaw|alaw|pcm (linear signed int), with
 *                 an added qualifier when the sample data are compressed
 * Other information may be contained in the header as well, but this has
 * no effect on the conversion to other file formats.
 *
 * Apple/Macintosh and Intel/Microsoft systems typically support RIFF
 * format for digital audio data, and users of these systems typically
 * do not have tools that can use sphere-formatted files as input.
 * `sph2pipe' will produce usable RIFF versions of sphere files so
 * that the waveform data is accessible using common tools on these
 * systems.  It can also produce the Mac-specific AIFF file format,
 * the AU format often used on Sun/sparc and Next systems, or raw
 * (headerless) sample data.
 *
 * Input conditions: 
 *  - input can be any valid sphere file, or any raw (headerless)
 *      sample data file when a suitable sphere header is provided
 *      separately, via the "-h hdrfile" option
 *  - input must be read from disk or cdrom, not from stdin
 *  - may be shorten compressed, or not
 *  - may be single- or two-channel
 *  - may be pcm or ulaw
 *  - if 2-byte pcm, may be either byte-order (HL/10 or LH/01)
 *  - may be any sample rate (typically 10, 11.025, 12.5, 16, 20, 22.1 KHz)
 *  - may be any size (from several KB to hundreds of MB)
 *
 * Output conditions:
 *  - output is written to stdout, unless an output file name is given
 *  - always uncompressed
 *  - formats: SPH,AU,WAV/RIFF,MAC/AIFF,RAW [-f sph/au/(wav|rif)/(mac|aif)/raw]
 *  - if two-channel, allow demux (output user-selected channel) [-c 1/2]
 *  - allow conversion to linear pcm [-p], alaw [-a] or mulaw [-u]
 *  - if writing pcm, byte order is set by output format or machine byte-order
 *  - allow selection (sec or samples) of start and end boundaries for output
 *
 * Overall method of operation:
 *  - determine native byte order of the machine we're running on
 *  - get user selections (from command line):
 *     -- input file name (and optional output file name)
 *     -- output file format (default="native" format of user's system)
 *     -- output channel (ignored for 1-ch input; default="both" for 2-ch)
 *     -- SPHERE header file (default=read SPHERE header from input file)
 *     -- force pcm or mular output (default=same as input)
 *  - read input sphere header for sample rate, etc.
 *  - create and output desired target file header, if any
 *  - loop over input data; for each buffer read from input:
 *     -- uncompress via "shorten extract" if necessary
 *     -- skip or seek past unwanted portions if necessary
 *     -- demux (discard one channel) if necessary
 *     -- convert to ulaw or to pcm if necessary
 *     -- invert byte order if necessary
 *     -- write to output
 *  - close input file and exit
 *
 * The program includes source code for "shorten-compressed" data extraction;
 * the shorten source code is copyright 1991-1999, Anthony J. Robinson.
 */

/* VERSION information:

 * This is version 2.4, intended to work on Wintel (MS Windows
 * 95/98/NT), linux, solaris -- also works on MacOS X, but not
 * intended for ealier Macintosh systems.

 * Revision history:
 *  - v1.1 was called "sph_convert", worked on one file at a time and allowed
 *    output to a named file; as of v1.2, sph_convert became a very different
 *    application (including mac support), and sph2pipe branched off.
 *  - sph2pipe v1.2 added sphere-header output (not available in sph_convert)
 *  - sph2pipe v2.0 was a major re-organization of the code, to simplify the
 *    maintenance of the "separate-but-almost-equal" sph_convert utility
 *  - sph2pipe v2.1 fixed a subtle bug in handling two-channel data (the fix
 *    was also incorporated into sph_convert v2.0)
 *  - sph2pipe v2.2 added options for AU and AIFF output formats
 *  - sph2pipe v2.3 added "-s bsamp:esamp" and "-t bsec:esec" for selecting
 *    ranges and the "-h hdrfile" option for using "stand-off" sphere headers
 *    with raw sample data as input.
 *  - sph2pipe v2.4 added alaw support (only for non-shortened data, because
 *    shorten does not support alaw), and fixed a bug in the logic involving
 *    the use of "-h hdrfile".
 */

#define _SPH_CONVERT_MAIN_

#include "sph_convert.h"
#include "ulaw.h"

static double bgnsec, endsec;
static int bgnsamp, endsamp;
static char *hdrfile;

int main( int argc, char **argv )
{
    int ret, n;
    char *usage =
	"Usage: sph2pipe [-h hdr] [-t|-s b:e] [-c 1|2] [-p|-u|-a] [-f typ] infile [outfile]\n\n\
   default conditions (for 'sph2pipe infile'):\n\
       * input file contains sphere header\n\
       * output full duration of input file\n\
       * output all channels from input file\n\
       * output same sample coding as input file\n\
       * output format is WAV on Wintel machines, SPH elsewhere\n\
       * output is written to stdout\n\n\
   optional controls (items bracketed separately above can be combined):\n\
       -h hdr -- treat infile as headerless, get sphere info from file 'hdr'\n\
       -t b:e -- output portion between b and e sec (floating point)\n\
       -s b:e -- output portion between b and e samples (integer)\n\
       -c 1   -- only output first channel\n\
       -c 2   -- only output second channel\n\
       -p     -- force conversion to 16-bit linear pcm\n\
       -u     -- force conversion to 8-bit ulaw\n\
       -a     -- force conversion to 8-bit alaw\n\
       -f typ -- select alternate output header format 'typ'\n\
                 five types: sph, raw, au, rif(wav), aif(mac)\n";

/* find out what the native byte order is:
 */
    short_order.i2 = 1;
    nativorder = ( short_order.ch[0] ) ? "01" : "10";
#ifdef MSDOS
    def_outheader = "RIFF";
#else
    def_outheader = "SPH";
#endif

/* command line options will decide the output conditions
 */
    if ( getUserOpts( argc, argv )) {
	fputs( usage, stderr );
	exit(1);
    }
/* make the data buffers
 */
    if (( outbuf = (char *) malloc( STD_BUF_SIZE*2 )) == NULL ||
	( inpbuf = (char *) malloc( STD_BUF_SIZE*2 )) == NULL ) {
	fprintf( stderr, "Not enough memory for %d byte buffer\n",
		STD_BUF_SIZE*4 );
	exit(1);
    }

/* When adapting to handle multiple files in one run, the following
 * function call would need to be placed into an appropriate loop or
 * directory-tree-walk function (and output filename handling would
 * most likely need to be added):
 */
    ret = doConversion( inpname, outname );

    exit(ret);
}
/* end of main() */


int getUserOpts( int ac, char **av )
{
    int i, nfn;
    char *cln;
    extern char *optarg;
    extern int optind;

/* set initial default values for command-line controls
 */
    hdrfile = NULL;
    debug = 0;
    endsec = bgnsec = 0;
    endsamp = bgnsamp = 0;
    typeout = 0;  /* will be interpreted as "same as input sample type" */
    chanout = 2;  /* will be interpreted as "same as input channel count" */
    outheader = def_outheader;  /* OS dependent (see sph_convert.h) */

    while (( i = getopt( ac, av, "daupf:c:t:s:h:" )) != EOF )
	switch ( i )
	{
	  case 'd':
	    debug = 1;
	    break;
	  case 'p':  /* force pcm output, regardless of input sample type */
	    typeout = PCM;
	    break;
	  case 'u':  /* force ulaw output, regardless of input sample type */
	    typeout = ULAW;
	    break;
	  case 'a':  /* force alaw output, regardless of input sample type */
	    typeout = ALAW;
	    break;
	  case 'c':  /* output just one channel, if input is two-channel */
	    chanout = ( *optarg == '1' ) ?  0 : ( *optarg == '2' ) ?  1 : -1;
	    break;
	  case 'h':
	    hdrfile = strdup( optarg );
	    break;
	  case 'f':  /* force a particular output format */
	    if ( strncasecmp( optarg, "RIF", 3 ) == 0 ||
		 strncasecmp( optarg, "WAV", 3 ) == 0 )
		outheader = "RIFF";
	    else if ( strncasecmp( optarg, "RAW", 3 ) == 0 )
		outheader = "RAW";
	    else if ( strncasecmp( optarg, "SPH", 3 ) == 0 )
		outheader = "SPH";
	    else if ( strncasecmp( optarg, "AU", 2 ) == 0 )
		outheader = "AU";
	    else if ( strncasecmp( optarg, "AIF", 3 ) == 0 ||
		      strncasecmp( optarg, "MAC", 3 ) == 0 )
		outheader = "AIF";
	    else
		outheader = NULL;
	    break;
	  case 't':  /* output only a portion of the file's timeline */
	    if (( cln = index( optarg, ':' )) == NULL ) {
		fprintf( stderr, "invalid arg for -t -- missing ':'\n" );
		return 1;
	    }
	    if ( cln > optarg ) { /* arg did not start with colon */
		*cln = '\0';
		if ( sscanf( optarg, "%lf", &bgnsec ) != 1 ) {
		    fprintf( stderr, "invalid first arg for -t\n" );
		    return 1;
		}
	    }
	    cln++;
	    if ( *cln != '\0' ) { /* arg did not end with colon */
		if ( sscanf( cln, "%lf", &endsec ) != 1 ) {
		    fprintf( stderr, "invalid second arg for -t\n" );
		    return 1;
		}
	    }
	    if ( bgnsec > 0 && endsec > 0 && bgnsec >= endsec ) {
		fprintf( stderr, "bgnsec %lf >= endsec %lf\n",
			bgnsec, endsec );
		return 1;
	    }
	    break;
	  case 's':  /* output only a portion of the file's samples */
	    if (( cln = index( optarg, ':' )) == NULL ) {
		fprintf( stderr, "invalid arg for -s -- missing ':'\n" );
		return 1;
	    }
	    if ( cln > optarg ) { /* arg did not start with colon */
		*cln = '\0';
		if ( sscanf( optarg, "%d", &bgnsamp ) != 1 ) {
		    fprintf( stderr, "invalid first arg for -s\n" );
		    return 1;
		}
	    }
	    cln++;
	    if ( *cln != '\0' ) { /* arg did not end with colon */
		if ( sscanf( cln, "%d", &endsamp ) != 1 ) {
		    fprintf( stderr, "invalid second arg for -s\n" );
		    return 1;
		}
	    }
	    if ( bgnsamp > 0 && endsamp > 0 && bgnsamp >= endsamp ) {
		fprintf( stderr, "bgnsamp %d >= endsamp %d\n",
			bgnsamp, endsamp );
		return 1;
	    }
	    break;
	  default:
	    return 1;
	}

/* A successful command line must provide one or two file names (input file,
 * output file), and recognized values for "-c" and/or "-f" if these are used
 */
    nfn = ac - optind;
    if (( nfn + 1 )/2 != 1 || outheader == NULL || chanout < 0 )
        return 1;

/* Output byte order will be HL for aif and au files, LH for riff files,
 * native order otherwise
 */
    outorder = ( !strcmp( outheader, "AU" ) || !strcmp( outheader, "AIF" )) ?  "10" :
	( !strcmp( outheader, "RIFF" )) ?  "01" : nativorder;

/* Output sample coding will be PCM for aif files (aif does not support ULAW)
 */
    if ( !strcmp( outheader, "AIF" ))
	typeout = PCM;

    if ( debug ) {
      fprintf( stderr, "command-line params: sizeout=%d, typeout=%d, outorder=%s, outheader=%s,\n",
	       sizeout, typeout, outorder, outheader );
      fprintf( stderr, "  bgnsamp=%d, bgnsec=%f, endsamp=%d, endsec=%f, chanout=%d\n", 
	       bgnsamp, bgnsec, endsamp, endsec, chanout );
    }
    inpname = strdup( av[optind] );
    outname = ( nfn == 2 ) ?  strdup( av[optind+1] ) : NULL;

    return 0;
}

int doConversion( char *inpname, char *outname )
{
    int ret;

    if (( fpin = fopen( inpname, "rb" )) == NULL ) {
	fprintf( stderr, "Unable to open %s as input\n", inpname );
	return 1;
    }
    if ( outname == NULL ) {
	fpout = stdout;
	outname = "stdout";
#ifdef MSDOS
	setmode(fileno(fpout), O_BINARY);
#endif
    }
    else if (( fpout = fopen( outname, "wb" )) == NULL ) {
	fprintf( stderr, "Unable to open %s as output\n", outname );
	return 1;
    }

/* input file header will set the input conditions (and some global variables)
 */
    if ( readSphHeader( hdrfile )) {
	fprintf( stderr, "Input file %s is not a valid SPHERE file\n",
		inpname );
	return 1;
    }
    if ( bgnsec > totalsec || bgnsamp > sampcount ) {
	fprintf( stderr, "start point > length of %s\n", inpname );
	return 1;
    }
    startout = ( bgnsamp ) ? bgnsamp : (int)( bgnsec * samprate );
    endout = ( endsamp > sampcount ||
	       endsec >= totalsec ||
	       endsamp + endsec == 0 ) ?  sampcount :
      ( endsamp > 0 ) ? endsamp : (int)( endsec * samprate );
/*
    fprintf( stderr, "startout=%d (bgnsec=%f) endout=%d (endsec=%f)\n",
	    startout, bgnsec, endout, endsec );
 */
    if ( typeout == 0 )      /* if command line didn't say...  */
	typeout = samptype;  /*    keep samptype same as input */
    if ( chancount == 1 )    /* if input is single-channel...  */
	chanout = chancount; /*    "-c" option doesn't matter  */
    if ( endout < 0 || endout > sampcount )
	endout = sampcount;
    if ( startout > 0  || endout < sampcount )
	sampcount = endout - startout;

    sizeout = typeout & 3;  /* yields 1 for ulaw/alaw, 2 for pcm */

    if ( debug ) {
      fprintf( stderr, "control params:  sizeout=%d, typeout=%d, outorder=%s, outheader=%s, chanout=%d,\n",
	       sizeout, typeout, outorder, outheader, chanout );
      fprintf( stderr, "  bgnsamp=%d, bgnsec=%f, endsamp=%d, endsec=%f, startout=%d, endout=%d, sampcount=%d\n", 
	       bgnsamp, bgnsec, endsamp, endsec, startout, endout, sampcount );
    }
/* now that we know what's coming in and going out, write the
 * appropriate output header, if any
 */
    if ( !strcmp( outheader, "RIFF" ))
	writeRIFFHeader();
    else if ( !strcmp( outheader, "AIF" ))
	writeAIFFHeader();
    else if ( !strcmp( outheader, "AU" ))
	writeAUHeader();
    else if ( !strcmp( outheader, "SPH" ))
	writeSphHeader();

/* now pass the data through */
    if ( doshorten )
	ret = shortenXtract();
    else
	ret = copySamples();

    if ( ret )
	fprintf( stderr, "conversion failed for %s\n", inpname );

    fclose( fpin );
    fclose( fpout );
    return ret;
}

/*************************************************************
 * pcm2ulaw
 *------------------------------------------------------------
 * Copied verbatim from Tony Robinson's "ulaw.c" (which in turn
 * was copied from Craig Reese)
 */

/*
** This routine converts from linear to ulaw.
**
** Craig Reese: IDA/Supercomputing Research Center
** Joe Campbell: Department of Defense
** 29 September 1989
**
** References:
** 1) CCITT Recommendation G.711  (very difficult to follow)
** 2) "A New Digital Technique for Implementation of Any
**     Continuous PCM Companding Law," Villeret, Michel,
**     et al. 1973 IEEE Int. Conf. on Communications, Vol 1,
**     1973, pg. 11.12-11.17
** 3) MIL-STD-188-113,"Interoperability and Performance Standards
**     for Analog-to_Digital Conversion Techniques,"
**     17 February 1987
**
** Input: Signed 16 bit linear sample
** Output: 8 bit ulaw sample
*/

#define ZEROTRAP    /* turn on the trap as per the MIL-STD */
#undef ZEROTRAP
#define BIAS 0x84   /* define the add-in bias for 16 bit samples */
#define CLIP 32635

uchar pcm2ulaw( short int sample )
{
    static int exp_lut[256] = {0,0,1,1,2,2,2,2,3,3,3,3,3,3,3,3,
			       4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
			       5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
			       5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
			       6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
			       6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
			       6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
			       6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
			       7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
			       7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
			       7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
			       7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
			       7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
			       7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
			       7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
			       7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7};
    int sign, exponent, mantissa;
    uchar ulawbyte;

    /* Get the sample into sign-magnitude. */
    sign = (sample >> 8) & 0x80; /* set aside the sign */
    if(sign != 0) sample = -sample; /* get magnitude */
    if(sample > CLIP) sample = CLIP; /* clip the magnitude */

    /* Convert from 16 bit linear to ulaw. */
    sample = sample + BIAS;
    exponent = exp_lut[( sample >> 7 ) & 0xFF];
    mantissa = (sample >> (exponent + 3)) & 0x0F;
    ulawbyte = ~(sign | (exponent << 4) | mantissa);
#ifdef ZEROTRAP
    if (ulawbyte == 0) ulawbyte = 0x02;	/* optional CCITT trap */
#endif

    return(ulawbyte);
}

/************************************************************
 * pcm2alaw
 *-----------------------------------------------------------
 * Adapted from "st_13linear2alaw()" function in SoX, as
 * found in "g711.c"
 */

#define SEG_SHIFT  (4)   /* Left shift for segment number */
#define QUANT_MASK (0xf) /* Quantization field mask */

uchar pcm2alaw( short int pcmval )
{
  short int mask, seg;
  uchar aval;
  static short int seg_end[8] = { 0x1F, 0x3F, 0x7F, 0xFF,
				  0x1FF,0x3FF,0x7FF,0xFFF };

  pcmval = pcmval >> 3; /* shift down to 13 bits */

  if ( pcmval >= 0 ) 
    mask = 0xd5;
  else {
    mask = 0x55;
    pcmval = -pcmval - 1;
  }
  for ( seg=0; seg<8; seg++ ) {
    if ( pcmval <= seg_end[seg] )
      break;
  }
  if ( seg == 8 )
    return (unsigned char) (0x7F ^ mask);
  else {
    aval = (unsigned char) seg << SEG_SHIFT;
    aval |= ( seg < 2 ) ?  (pcmval >> 1) & QUANT_MASK : (pcmval >> seg) & QUANT_MASK;
    return (aval ^ mask);
  }
}

int copySamples( void )
{
    int i, nb, ns, sampsdone;
    short int *sptr, *cnvptr, s;
    uchar *cptr, (*pcm2xptr)( short int );
    char *wptr;

    if ( startout > 0 )
	fseek( fpin, startout * sampsize * chancount, SEEK_CUR );

    sampsdone = 0;
    while (( nb = fread( inpbuf, 1, STD_BUF_SIZE, fpin )) > 0 &&
	   sampsdone < sampcount )
    {
	ns = nb / ( chancount * sampsize );
	if (( sampsdone + ns ) > sampcount ) {
	    ns = sampcount - sampsdone;
	    nb = ns * chancount * sampsize;
	}
	sampsdone += ns;
	if ( chancount > chanout ) { /* input chancount==2, chanout=0 or 1 */
	    demux( nb );
	    nb /= 2;
	}
	wptr = inpbuf;
	if ( sampsize < sizeout ) { /* convert ulaw or alaw to pcm */
	    cptr = inpbuf;
	    sptr = (short int *) outbuf;
	    cnvptr = ( samptype == ALAW ) ?  alaw2pcm : ulaw2pcm;
	    for ( i=0; i<nb; i++ )
		*sptr++ = cnvptr[*cptr++];
	    nb *= 2;
	    if ( strcmp( nativorder, outorder )) /* if output filetype needs */
		swab( outbuf, inpbuf, nb );      /* it, do byte swapping too */
	    else
		wptr = outbuf;
	}
	else if ( sampsize > sizeout ) { /* convert pcm to ulaw or alaw */
	    if ( strcmp( inporder, nativorder )) { /* if inp. filetype needs */
		swab( inpbuf, outbuf, nb );        /* it, do byte swap first */
		sptr = (short int *) outbuf;
		cptr = inpbuf;
	    } else {
		wptr = cptr = outbuf;
		sptr = (short int *) inpbuf;
	    }

	    if ( typeout == ALAW ) 
	      pcm2xptr = pcm2alaw;
	    else 
	      pcm2xptr = pcm2ulaw;

	    for ( i=0; i<nb; i+=2 )
	      *cptr++ = (*pcm2xptr)( *sptr++ );
	    nb /= 2;
	}
	else if ( samptype == ALAW && typeout == ULAW ) { /* convert alaw to ulaw */
	    cptr = inpbuf;
	    for ( i=0; i<nb; i++ ) {
		s = alaw2pcm[*cptr];
		*cptr++ = pcm2ulaw( s );
	    }
	}
	else if ( samptype == ULAW && typeout == ALAW ) { /* convert ulaw to alaw */
	    cptr = inpbuf;
	    for ( i=0; i<nb; i++ ) {
		s = ulaw2pcm[*cptr];
		*cptr++ = pcm2alaw( s );
	    }
	}
	else if ( samptype == 2 && strcmp( inporder, outorder )) {
	    swab( inpbuf, outbuf, nb );
	    wptr = outbuf;
	}
	if ( fwrite( wptr, 1, nb, fpout ) != nb ) {
	    fprintf( stderr, "Failed while writing sample data to %s\n",
		    outname );
	    exit( 1 );
	}
    }
    if ( sampsdone != sampcount )
	fprintf( stderr, "Warning: %d samples written, %d samples expected\n",
		sampsdone, sampcount );
    return( sampsdone != sampcount );
}

void demux( int ns )
{
    int i;
    short int *sptr, *sptr2;
    uchar *cptr, *cptr2;

/* To demultiplex, simply move the samples of the selected channel
 * so that they are adjacent starting at offset 0 of inpbuf; this
 * overwrites the unselected channel data.
 */
    if ( sampsize == 2 ) {
	ns /= 2;
	i = chanout;
	sptr = (short int *) inpbuf;
	sptr2 = sptr + chanout;
	if ( chanout == 0 ) {
	    i = 2;
	    sptr++;
	    sptr2 += 2;
	}
	for ( ; i<ns; i+=2 ) {
	    *sptr++ = *sptr2;
	    sptr2 += 2;
	}
    } else {	/* sampsize == 1 */
	i = chanout;
	cptr = inpbuf;
	cptr2 = cptr + chanout;
	if ( chanout == 0 ) {
	    i = 2;
	    cptr++;
	    cptr2 += 2;
	}
	for ( ; i<ns; i+=2 ) {
	    *cptr++ = *cptr2;
	    cptr2 += 2;
	}
    }
}
