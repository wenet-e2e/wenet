/*************************************************************
 * Source File:	file_headers.c
 * Compilation:	gcc -o sph2pipe sph2pipe.c shorten_x.c file_headers.c -lm
 * Author:	Dave Graff; LDC, University of Pennsylvania
 * Purpose:	functions to read/write SPHERE headers, and
 *		write RIFF, AU, AIFF headers
 */

#include "sph_convert.h"
#include <math.h>

static char hdr[90];
static int hdrsize, origSampcount;

/*************************************************************
 * readSphHeader
 *------------------------------------------------------------
 * get relevant fields from input sphere file header; return true (1) for success;
 * return false if:
 * - input file appears not to contain a sphere header
 * - header does not contain all of these fields:
 *      channel_count, sample_count, sample_rate, sample_n_bytes, sample_coding(*)
 *      (* if sample_coding is missing, assume uncompressed pcm)
 * - there is no data following the header
 * If these conditions pass, check first four bytes of data to confirm whether
 * file is "shorten" compressed, then assign values to corresponding globals
 */

int readSphHeader( char *hdrfname )
{
    size_t n;
    int nx, inphdrsize;
    char *field, fldname[24], fldtype[8], fldsval[32], cmpcheck[4];
    FILE *fphd;

    if ( hdrfname == NULL )
        fphd = fpin;
    else if (( fphd = fopen( hdrfname, "rb" )) == NULL ) {
        fprintf( stderr, "Unable to open %s as input header\n", hdrfname );
	return 1;
    }
    n = fread( inpbuf, 1, 1024, fphd );	/* nearly all sphere headers are 1024 bytes */

    if ( n != 1024 || strncmp( inpbuf, "NIST_1A", 7 ))
	return 1;

    if ( sscanf( &inpbuf[8], "%d", &hdrsize ) != 1 )
	return 1;

    if ( hdrsize > 1024 ) {
	if ( hdrsize >= STD_BUF_SIZE*2 ) { /* this should not happen */
	    fprintf( stderr, "Invalid header size (%d) in %s\n", hdrsize, hdrfname );
	    return 1;
	}
	fseek( fphd, 0, 0 );
	n = fread( inpbuf, 1, hdrsize, fphd );
	if ( n != hdrsize ) {
	    fprintf( stderr, "Couldn't read %d byte header in %s\n", hdrsize, hdrfname );
	    return 1;
	}
    }

    /* from now on, inphdrsize should represent an offset into the data file (fpin), 
	to the point where actual sample data begins; in the case where the header was
	read from a separate file (hdrfname is not null), this offset is zero
     */
    inphdrsize = hdrsize;
    if ( hdrfname != NULL ) {
	fclose( fphd );
	inphdrsize = 0;
    }

    /* having read the header, also read the first four bytes of sample data */
    if ( fread( cmpcheck, 1, 4, fpin ) != 4 ) {
        fprintf( stderr, "Unable to read sample data in %s\n", inpname );
	return 1;
    }
    fseek( fpin, inphdrsize, 0 );

    samptype = sampsize = sampcount = samprate = chancount = UNKNOWN;
    inporder = NULL;

    field = strtok( inpbuf, "\n" );
    while ( field != NULL && strcmp( field, "end_head" ))
    {
	if ( !strncmp( field, "channel_count -i ", 17 ))
	    sscanf( field, "%s %s %d", fldname, fldtype, &chancount );
	else if ( !strncmp( field, "sample_count -i ", 16 ))
	    sscanf( field, "%s %s %d", fldname, fldtype, &sampcount );
	else if ( !strncmp( field, "sample_rate -i ", 15 ))
	    sscanf( field, "%s %s %d", fldname, fldtype, &samprate );
	else if ( !strncmp( field, "sample_n_bytes -i ", 18 ))
	    sscanf( field, "%s %s %d", fldname, fldtype, &sampsize );
	else if ( !strncmp( field, "sample_byte_format -s", 21 )) {
	    sscanf( field, "%s %s %s", fldname, fldtype, fldsval );
	    inporder = strdup( fldsval );
	}
	else if ( !strncmp( field, "sample_coding -s", 16 )) {
	    sscanf( field, "%s %s %s", fldname, fldtype, fldsval );
	    samptype =
	      ( !strncmp( fldsval, "ulaw", 4 )) ?  ULAW :
	      ( !strncmp( fldsval, "alaw", 4 )) ?  ALAW :
	      ( !strncmp( fldsval, "pcm", 3 ))  ?  PCM  : UNKNOWN;
	}
	field = strtok( NULL, "\n" );
    }
    if ( strcmp( field, "end_head" )) /* this shouldn't happen */
	return 1;

    if ( !samptype &&
	 ( sampsize == 2 || ( inporder && strlen( inporder ) == 2 )))
        samptype = PCM;

    /* having done that, the following things must be known, or else
     * we don't really have a usable sphere file:
     */
    if ( !samptype || !sampcount || !samprate || !chancount ||
	( samptype == PCM && inporder == NULL ))
	return 1;
    
    /* if "sample_n_bytes" was not specified, we can set it based
     * on sample_coding ( samptype & 3 )
     */
    if ( sampsize == UNKNOWN )
      sampsize = samptype & 3;
    
    /* Now that we're done with the text data in the sphere header, look
     * at the first four bytes of waveform data to see if it's shortened
     */
    origSampcount = sampcount;
    if ( !strncmp( cmpcheck, "ajkg", 4 )) { /* the "magic number" for shorten */
	doshorten++;
	if ( samptype == ALAW )  /* this must be a mistake -- abort now */
	  return 1;
    }
    else {	/* when not shortened, we can try to check file size vs. header specs */
	struct stat statbuf;
	int fdin;
	fdin = fileno( fpin );
	if ( fstat( fdin, &statbuf ) < 0 )
	    fprintf( stderr, "Warning: unable to determine size of file %s\n",
		    inpname );
	else {
	  n = inphdrsize + chancount * sampcount * sampsize;
	    if ( statbuf.st_size != n ) { /* if they conflict, go with the file size */
		sampcount = ( statbuf.st_size - inphdrsize ) / ( chancount * sampsize );
		fprintf( stderr,
			 "Warning:%s: sample_count reset to %d to match size (%d bytes)\n",
			inpname, sampcount, statbuf.st_size );
	    }
	}
    }
/* compute total duration, leave file pointer at end of header (start of data)
 */
    totalsec = sampcount / (double) samprate;

    return 0;
}

/*************************************************************
 * writeSphHeader
 *------------------------------------------------------------
 * If used at all, this function is called almost immediately
 * after readSphHeader, so SPH header data is still present in
 * inpbuf.  Now, make adjustments to the header data as needed
 * (to reflect uncompression, and/or conversion between u/alaw 
 * and pcm, and/or demux), and adjust padding to assure the
 * correct header size on output.
 */
void writeSphHeader( void )
{
    char *field, *ohdr, *fldsval, extrahdr[16];
    int i, flen, hdrbytesOut = 0, didSBFormat = 0;

/* Header data is still in inpbuf after call to readSphHeader,
 * but first we have to undo the effects of strtok():
 */
    ohdr = inpbuf;
    for ( i=0; i<hdrsize; i++ ) {
	if ( *ohdr == '\0' )
	    *ohdr = '\n';
	ohdr++;
    }

    ohdr = outbuf;
    field = strtok( inpbuf, "\n" );
    while ( field != NULL && strcmp( field, "end_head" ))
    {
	if ( !strncmp( field, "sample_checksum ", 16 )) {
	    field = strtok( NULL, "\n" );  /* can't set a correct value here, so */
	    continue;                      /* we're better off leaving it out */
	}
	else if ( !strncmp( field, "channel_count -i ", 17 ) && chancount > chanout )
	    flen = sprintf( ohdr, "channel_count -i 1\n" );
	else if ( !strncmp( field, "sample_count -i ", 16 ) &&
		 sampcount != origSampcount )
	    flen = sprintf( ohdr, "sample_count -i %d\n", sampcount );
	else if ( !strncmp( field, "sample_n_bytes -i ", 18 ) &&
		 sampsize != sizeout )
	    flen = sprintf( ohdr, "sample_n_bytes -i %d\n", sizeout );
	else if ( !strncmp( field, "sample_byte_format -s", 21 ) &&
		 ( sampsize != sizeout || ( sizeout == 2 && strcmp( outorder, inporder )))) {
	    if ( sizeout == 1 ) {
		field = strtok( NULL, "\n" );  /* don't need this field for u/alaw output */
		continue;
	    }
	    flen = sprintf( ohdr, "sample_byte_format -s2 %s\n", outorder );
	    didSBFormat++;
	}
	else if ( !strncmp( field, "sample_coding -s", 16 ) &&
		 ( doshorten || ( samptype != typeout ))) {
	    if ( typeout == PCM ) {
		i = 3;
		fldsval = "pcm";
	    } else {
		i = 4;
		fldsval = ( typeout == ALAW ) ?  "alaw" : "ulaw";
	    }
	    flen = sprintf( ohdr, "sample_coding -s%d %s\n", i, fldsval );
	}
	else if ( !strncmp( field, "sample_sig_bits -i ", 19 ) ) {
	    int bits;
	    sscanf( &field[19], "%d", &bits );
	    if ( bits > 8 && typeout != PCM ) 
	      flen = sprintf( ohdr, "sample_sig_bits -i 8\n" );
	    else if ( bits == 8 && typeout == PCM )
	      flen = sprintf( ohdr, "sample_sig_bits -i 16\n" );
	    else
	      flen = sprintf( ohdr, "%s\n", field );  /* no change needed */
	}
	else
	    flen = sprintf( ohdr, "%s\n", field );
	
	ohdr += flen;
	hdrbytesOut += flen;
	field = strtok( NULL, "\n" );
    }

/* Minor detail: if input is ulaw/alaw AND output is pcm AND the input header
 * happens to lack the "sample_byte_format" field (because this is not needed
 * for ulaw/alaw data), now we have to add this field to the output header.
 */
    if ( didSBFormat == 0 && sampsize < sizeout ) {
	flen = sprintf( ohdr, "sample_byte_format -s2 %s\n", outorder );
	ohdr += flen;
	hdrbytesOut += flen;
    }

    flen = sprintf( ohdr, "end_head\n" );  /* add the end-of-header marker */
    ohdr += flen;
    hdrbytesOut += flen;

/* Final detail: it's possible that changing "sample_sig_bits" and/or
 * "sample_coding", and/or adding/changing "sample_byte_format" could
 * enlarge the header size beyond its current multiple of 1024 -- if so,
 * we need to expand the header size to the next multiple of 1024:
 */
    if ( hdrbytesOut > hdrsize ) {
        hdrsize += 1024;
	sprintf( extrahdr, "%7d", hdrsize );
	strncpy( &outbuf[8], extrahdr, 7 );
    }

/* Add white-space padding to complete the header block
 */
    while ( hdrbytesOut < hdrsize ) {
	*ohdr++ = (char)(( hdrbytesOut % 32 ) ? ' ' : '\n' );
	hdrbytesOut++;
    }
    if ( fwrite( outbuf, 1, hdrsize, fpout ) != hdrsize ) {
	fprintf( stderr, "Couldn't write %d byte output header from %s\n",
		hdrsize, inpname );
	exit(1);
    }
}


/*************************************************************
 * copycharr
 *------------------------------------------------------------
 * could've used memcopy, but this feels more stable/portable...
 */
void copycharr( char *from, char *to, int n )
{
    int i;
    for ( i=0; i<n; i++ )
	*to++ = *from++;
}

/*************************************************************
 * copylong
 *------------------------------------------------------------
 * copy a 4-byte int, making sure to use the intended byte
 * order for the destination, regardless of what the
 * native byte order is on the current machine
 */
void copylong( int val, char *dest, char *intended )
{
    int i, e, incr;

    if ( strcmp( nativorder, intended )) { 
	i = 3;
	e = -1;
	incr = -1;
    }
    else {
	i = 0;
	e = 4;
	incr = 1;
    }
    long_order.i4 = val;
    for ( ; i != e; i += incr )
	*dest++ = long_order.ch[i];
}

/*************************************************************
 * copyshort
 *------------------------------------------------------------
 * copy a 2-byte int, making sure to use the intended byte
 * order for the destination, regardless of what the
 * native byte order is on the current machine
 */
void copyshort( short int val, char *dest, char *intended )
{
    if ( strcmp( nativorder, intended ))
	swab((char *) &val, short_order.ch, 2 );
    else
	short_order.i2 = val;

    *dest++ = short_order.ch[0];
    *dest = short_order.ch[1];
}

/*************************************************************
 * writeAUHeader
 *------------------------------------------------------------
 * The folloing documentation about the AU header format has been
 * copied verbatim from www.wotsit.org (no copyright statement):
-------
 * [ From: mrose@dbc.mtview.ca.us (Marshall Rose) ]
 * 
 * Audio data is encoded in three parts: a header, containing fields that
 * describe the audio encoding format; a variable-length information field,
 * in which, for instance, ASCII annotation may be stored; and, the actual
 * encoded audio.  The header and data fields are written using big-endian
 * ordering.
 * 
 * The header part consists of six 32-bit quantities, in this order:
 * 
 * longword	field		description
 * --------	-----		-----------
 *  0		magic number	the value 0x2e736e64 (ASCII ".snd")
 * 
 *  1		data offset	the offset, in octets, to the data part.
 * 				The minimum valid number is 24 (decimal).
 * 
 *  2		data size	the size in octets, of the data part.
 * 				If unknown, the value 0xffffffff should
 * 				be used.
 * 
 *  3		encoding	the data encoding format:
 * 				    value	format
 * 				      1		8-bit ISDN u-law
 * 				      2		8-bit linear PCM [REF-PCM]
 * 				      3		16-bit linear PCM
 * 				      4		24-bit linear PCM
 * 				      5		32-bit linear PCM
 * 				      6		32-bit IEEE floating point
 * 				      7		64-bit IEEE floating point
 * 				     23		8-bit ISDN u-law compressed
 * 						using the CCITT G.721 ADPCM
 * 						voice data encoding scheme.
 * 
 *  4		sample rate	the number of samples/second (e.g., 8000)
 * 
 *  5		channels	the number of interleaved channels (e.g., 1)
 * 
 * The information part, consists of 0 or more octets, and starts 24 octets
 * after the beginning of the header part. The length of the information
 * part is calculated by subtracting 24 (decimal) from the data offset
 * field in the header part.
 * --
 *  Bill Janssen      janssen@parc.xerox.com      (415) 812-4763
 *  Xerox Palo Alto Research Center      FAX: (415) 812-4777
 *  3333 Coyote Hill Road, Palo Alto, California   94304
-----
 * The following approach was written by Dave Graff for the LDC
 */
void writeAUHeader( void )
{
    char *ordr = "10";  /* AU files use high-byte first */
    int dsize, nchan, enc;

    nchan = ( chanout < chancount ) ?  1 : chancount;
    hdrsize = 24;
    dsize = sampcount * nchan * sizeout;
    enc = ( sizeout == 1 ) ?  1 : 3;  /* either 8-bit u-law or 16-bit PCM */
    copycharr( ".snd", &hdr[0], 4 );
    copylong( hdrsize, &hdr[4], ordr );
    copylong( dsize, &hdr[8], ordr );
    copylong( enc, &hdr[12], ordr );
    copylong( samprate, &hdr[16], ordr );
    copylong( nchan, &hdr[20], ordr );

    if ( fwrite( hdr, 1, hdrsize, fpout ) != hdrsize ) {
	fprintf( stderr, "Failed to write AU header to %s\n", outname );
	exit(1);
    }
}

/*************************************************************
 * writeRIFFHeader
 *------------------------------------------------------------
 * The following documentation about the RIFF header format has been
 * copied verbatim from "wav.c" in sox-12.17; the copyright notice
 * contained in that source file is included, and applies to the
 * following commentary:

----- excerpt from sox-12.17/wav.c 
----- (available from http://home.sprynet.com/~cbagwell/sox.html)

 * Microsoft's WAVE sound format driver
 *
 * This source code is freely redistributable and may be used for
 * any purpose.  This copyright notice must be maintained. 
 * Lance Norskog And Sundry Contributors are not responsible for 
 * the consequences of using this software.
 *
 * ... [See note below.  -- DG/LDC]
 *
 * NOTE: Previous maintainers weren't very good at providing contact
 * information.
 *
 * Copyright 1992 Rick Richardson
 * Copyright 1991 Lance Norskog And Sundry Contributors
 *
 * ...
 *
 * Info for format tags can be found at:
 *   http://www.microsoft.com/asf/resources/draft-ietf-fleischman-codec-subtree-01.txt

... write .wav headers as follows:
 
bytes      variable      description
0  - 3     'RIFF'
4  - 7     wRiffLength   length of file minus the 8 byte riff header
8  - 11    'WAVE'
12 - 15    'fmt '
16 - 19    wFmtSize       length of format chunk minus 8 byte header 
20 - 21    wFormatTag     identifies PCM, ULAW, ALAW etc
22 - 23    wChannels      
24 - 27    wSamplesPerSecond   samples per second per channel
28 - 31    wAvgBytesPerSec     non-trivial for compressed formats
32 - 33    wBlockAlign         basic block size
34 - 35    wBitsPerSample      non-trivial for compressed formats

PCM formats then go straight to the data chunk:
36 - 39    'data'
40 - 43     wDataLength   length of data chunk minus 8 byte header
44 - (wDataLength + 43)    the data

non-PCM formats must write an extended format chunk and a fact chunk:

ULAW, ALAW formats:
36 - 37    wExtSize = 0  the length of the format extension
38 - 41    'fact'
42 - 45    wFactSize = 4  length of the fact chunk minus 8 byte header
46 - 49    wSamplesWritten   actual number of samples written out
50 - 53    'data'
54 - 57     wDataLength   length of data chunk minus 8 byte header
58 - (wDataLength + 57)    the data

...
----- end of excerpt

 * Note: The source code change history (and source code) was omitted;
 * Stan Brooks (stabro@megsinet.com) and Chris Bagwell
 * (cbagwell@sprynet.com) authored several recent improvements
 * to wav.c, including the documentation quoted above.
 *
 * The following approach, written by David Graff for the LDC,
 * supports output to stdout (this was not supported in sox-12.17,
 * probably because sox included support for various RIFF-based
 * forms of compression, which are not supported here).
 */
void writeRIFFHeader( void )
{
    char *ordr = "01";		/* RIFF header wants ints with low-byte first */
    int fsize, hsize, hoffs;
    short int nbyts, nchan, fmtyp;
  
    nchan = ( chanout < chancount ) ?  1 : chancount;
    nbyts = nchan * sizeout;
    fsize = sampcount * nbyts;
    if ( sizeout == 1 ) {  /* applies to ALAW and ULAW */
	hoffs = 18;
	hsize = 50;
	fmtyp = ( typeout == ALAW ) ?  0x0006 : 0x0007;
    } else {
	hoffs = 16;
	hsize = 36;
	fmtyp = 0x0001;
    }
    copycharr( "RIFF", &hdr[0], 4 );
    copylong( fsize + hsize, &hdr[4], ordr );
    copycharr( "WAVE", &hdr[8], 4 );
    copycharr( "fmt ", &hdr[12], 4 );
    copylong( hoffs, &hdr[16], ordr );
    copyshort( fmtyp, &hdr[20], ordr );
    copyshort( nchan, &hdr[22], ordr );
    copylong( samprate, &hdr[24], ordr );
    copylong( nbyts * samprate, &hdr[28], ordr );
    copyshort( nbyts, &hdr[32], ordr );
    copyshort( sizeout * 8, &hdr[34], ordr );
    if ( sizeout == 1 ) {  /* applies to ALAW and ULAW */
	copyshort( 0, &hdr[36], ordr );
	copycharr( "fact", &hdr[38], 4 );
	copylong( 4, &hdr[42], ordr );
	copylong( sampcount, &hdr[46], ordr );
    }
    hoffs = hsize;
    copycharr( "data", &hdr[hoffs], 4 );
    copylong( fsize, &hdr[hoffs+4], ordr );

    hsize += 8;	/* add in the first 8 bytes, which weren't included earlier */
    if ( fwrite( hdr, 1, hsize, fpout ) != hsize ) {
	fprintf( stderr, "Failed to write WAV header to %s\n", outname );
	exit(1);
    }
}

/* The following documentation and code for "ConvertToIeeeExtended"
 * has been copied verbatim from the SoX source code distribution.  
 * The function calls invoked here require the inclusion of the 
 * math library at compile-time ("-lm").
 */

/*
 * C O N V E R T   T O   I E E E   E X T E N D E D
 */

/* Copyright (C) 1988-1991 Apple Computer, Inc.
 * All rights reserved.
 *
 * Machine-independent I/O routines for IEEE floating-point numbers.
 *
 * NaN's and infinities are converted to HUGE_VAL or HUGE, which
 * happens to be infinity on IEEE machines.  Unfortunately, it is
 * impossible to preserve NaN's in a machine-independent way.
 * Infinities are, however, preserved on IEEE machines.
 *
 * These routines have been tested on the following machines:
 *    Apple Macintosh, MPW 3.1 C compiler
 *    Apple Macintosh, THINK C compiler
 *    Silicon Graphics IRIS, MIPS compiler
 *    Cray X/MP and Y/MP
 *    Digital Equipment VAX
 *
 *
 * Implemented by Malcolm Slaney and Ken Turkowski.
 *
 * Malcolm Slaney contributions during 1988-1990 include big- and little-
 * endian file I/O, conversion to and from Motorola's extended 80-bit
 * floating-point format, and conversions to and from IEEE single-
 * precision floating-point format.
 *
 * In 1991, Ken Turkowski implemented the conversions to and from
 * IEEE double-precision format, added more precision to the extended
 * conversions, and accommodated conversions involving +/- infinity,
 * NaN's, and denormalized numbers.
 */

#ifndef HUGE_VAL
# define HUGE_VAL HUGE
#endif /*HUGE_VAL*/

#define FloatToUnsigned(f) ((unsigned long)(((long)(f - 2147483648.0)) + 2147483647L + 1))

ConvertToIeeeExtended(num, bytes)
double num;
char *bytes;
{
    int sign;
    int expon;
    double fMant, fsMant;
    unsigned long hiMant, loMant;

    if (num < 0) {
        sign = 0x8000;
        num *= -1;
    } else {
        sign = 0;
    }

    if (num == 0) {
        expon = 0; hiMant = 0; loMant = 0;
    }
    else {
        fMant = frexp(num, &expon);
        if ((expon > 16384) || !(fMant < 1)) {    /* Infinity or NaN */
            expon = sign|0x7FFF; hiMant = 0; loMant = 0; /* infinity */
        }
        else {    /* Finite */
            expon += 16382;
            if (expon < 0) {    /* denormalized */
                fMant = ldexp(fMant, expon);
                expon = 0;
            }
            expon |= sign;
            fMant = ldexp(fMant, 32);
            fsMant = floor(fMant);
            hiMant = FloatToUnsigned(fsMant);
            fMant = ldexp(fMant - fsMant, 32);
            fsMant = floor(fMant);
            loMant = FloatToUnsigned(fsMant);
        }
    }
    
    bytes[0] = expon >> 8;
    bytes[1] = expon;
    bytes[2] = hiMant >> 24;
    bytes[3] = hiMant >> 16;
    bytes[4] = hiMant >> 8;
    bytes[5] = hiMant;
    bytes[6] = loMant >> 24;
    bytes[7] = loMant >> 16;
    bytes[8] = loMant >> 8;
    bytes[9] = loMant;
}

/* The following code has been adapted from the file "aiff.c" provided
 * in the SoX source code distribution.
 */
void writeAIFFHeader( void )
{
    char *ordr = "10";    /* AIFF header want high-byte first */
    int fsize, hsize, hoffs;
    short int nbyts, nchan, fmtyp;
    char ieeebuf[10];

    nchan = ( chanout < chancount ) ?  1 : chancount;
    nbyts = nchan * sizeout;
    hsize = 46;
    fsize = nbyts * sampcount;
    ConvertToIeeeExtended((double)samprate, ieeebuf);

    copycharr( "FORM", &hdr[0], 4 );
    copylong( fsize + hsize, &hdr[4], ordr );
    copycharr( "AIFF", &hdr[8], 4 );
    copycharr( "COMM", &hdr[12], 4 );
    copylong( 18, &hdr[16], ordr );
    copyshort( nchan, &hdr[20], ordr );
    copylong( sampcount, &hdr[22], ordr );
    copyshort( 16, &hdr[26], ordr );
    copycharr( ieeebuf, &hdr[28], 10 );
    copycharr( "SSND", &hdr[38], 4 );
    copylong( 8+fsize, &hdr[42], ordr );
    copylong( 0, &hdr[46], ordr );
    copylong( 0, &hdr[50], ordr );

    hsize += 8;	/* add in the first 8 bytes, which weren't included earlier */
    if ( fwrite( hdr, 1, hsize, fpout ) != hsize ) {
	fprintf( stderr, "Failed to write AIFF header to %s\n", outname );
	exit(1);
    }
}

