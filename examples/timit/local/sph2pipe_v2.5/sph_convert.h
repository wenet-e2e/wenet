/*************************************************************
 * sph_convert.h
 *------------------------------------------------------------
 *		 primary header file for sph_convert.c
 *			and other related .c files
 *
 * This takes care of all other necessary #include's, all program-wide
 * #define's, all function declarations, and global variables
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>

#ifdef _SPH_CONVERT_MAIN_
#define GLOBAL
#else
#define GLOBAL extern
#endif

#define STD_BUF_SIZE 16384
#define PCM 2
#define ULAW 1
#define ALAW 5
#define UNKNOWN 0

/* relation of PCM, ULAW, ALAW values to sample size:
   PCM & 3 == 2; ULAW & 3 == 1; ALAW & 3 == 1 */

/* Functions used by sph_convert & sph2pipe:
 *   all int functions return 0 for success, non-zero (1) for failure
 */
int doConversion( char *, char * );   /* isolates file i/o from main() */
int getUserOpts( int, char ** ); /* parses command-line options */
int readSphHeader( char * );     /* as the name implies... */
void writeSphHeader( void );     /* exits on error */
void writeAUHeader( void );      /* exits on error */
void writeRIFFHeader( void );    /* exits on error */
void writeAIFFHeader( void );    /* exits on error */
int shortenXtract( void );       /* handles data i/o for shortened files */
int copySamples( void );         /* handles data i/o for normal files */
void demux( int );               /* demultiplex 2-channel buffer in-place */

/* Global variables:
 */

GLOBAL char *def_outheader;
GLOBAL double totalsec;
GLOBAL int chancount, samptype, sampsize, sampcount, samprate, doshorten;
GLOBAL int chanout, typeout, sizeout, startout, endout, debug;
GLOBAL char *nativorder, *inporder, *outorder, *outheader;
GLOBAL char mesgbuf[512];
GLOBAL FILE *fpin, *fpout;
GLOBAL char *inpname, *outname;
GLOBAL char *inpbuf, *outbuf;

GLOBAL union {
    char ch[2];
    short int i2;
} short_order;

GLOBAL union {
    char ch[4];
    int i4;
} long_order;

/* The following "pseudo-typedefs" are adopted for the sake of
 * working with Tony Robinson's "shorten" source code
 */
#undef	uchar
#define uchar	unsigned char
#undef	schar
#define schar	signed char
#undef	ushort
#define ushort	unsigned short
#undef	ulong
#define ulong	unsigned long


/* This routine, copied directly from Tony Robinson's "shorten"
 * package, converts from ulaw to 16 bit linear.
 *
 * Craig Reese: IDA/Supercomputing Research Center
 * 29 September 1989
 *
 * References:
 * 1) CCITT Recommendation G.711  (very difficult to follow)
 * 2) MIL-STD-188-113,"Interoperability and Performance Standards
 *     for Analog-to_Digital Conversion Techniques,"
 *     17 February 1987
 *
 * Input: 8 bit ulaw sample
 * Output: signed 16 bit linear sample
 *
 * Note: this version differs from Tony's (and Craig's?) by changing
 * all the "int" declarations to "short int".  It is commented out now,
 * because a "ulaw2pcm" array of 256 short-int values is being used
 * instead (see above); that lookup table was created using this code.
 * [DG/LDC]

short int ulaw2pcm(char ulawbyte)
{
  static int exp_lut[8] = { 0, 132, 396, 924, 1980, 4092, 8316, 16764 };
  short int sign, exponent, mantissa, sample;

  ulawbyte = ~ulawbyte;
  sign = (ulawbyte & 0x80);
  exponent = (ulawbyte >> 4) & 0x07;
  mantissa = ulawbyte & 0x0F;
  sample = exp_lut[exponent] + (mantissa << (exponent + 3));
  if(sign != 0) sample = -sample;

  return(sample);
}

 */
