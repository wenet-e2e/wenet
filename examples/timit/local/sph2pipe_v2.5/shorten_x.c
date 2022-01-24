/*************************************************************
 * Source File:	shorten_x.c
 * Compilation:	gcc -o sph_convert sph_convert.c shorten_x.c file_headers.c
 * Authors:	Created/invented by Tony Robinson, adapted by Dave Graff
 * Purpose:	uncompress "shortened" waveform data, convert ulaw to pcm
 *		if necessary, demux 2-channel data (discard a channel)
 *		if necessary, and write to output file.
 *
 * The various portions of source code from Tony Robinson's
 * "shorten-2.0" package are used here by permission of Tony Robinson
 * and SoftSound, Inc., who reserve all rights.
 *
 * By agreement with Tony Robinson and SoftSound, Inc, the Linguistic
 * Data Consortium (LDC) grants permission to copy and use this
 * software for the purpose of reading "shorten"-compressed speech
 * data provided by the LDC or others using the NIST SPHERE file
 * format.
 *
 * No portion of this code may be copied or adapted for other uses
 * without permission from Tony Robinson and SoftSound, Inc. (web:
 * http://www.softsound.com, email: info@softsound.com)
 *
 */

#include "sph_convert.h"
#include "bitshift.h"
#include "ulaw.h"

#undef	uchar
#define uchar	unsigned char
#undef	schar
#define schar	signed char
#undef	ushort
#define ushort	unsigned short
#undef	ulong
#define ulong	unsigned long

#define  MAGIC			"ajkg"
#define  FORMAT_VERSION		2
#define  MIN_SUPPORTED_VERSION	1
#define  MAX_SUPPORTED_VERSION	2
#define  MAX_VERSION		7
#define  UNDEFINED_UINT		-1
#define  DEFAULT_BLOCK_SIZE	256
#define  DEFAULT_V0NMEAN	0
#define  DEFAULT_V2NMEAN	4
#define  DEFAULT_MAXNLPC	0
#define  DEFAULT_NCHAN		1
#define  DEFAULT_NSKIP		0
#define  DEFAULT_NDISCARD	0
#define  NBITPERLONG		32
#define  DEFAULT_MINSNR         256
#define  DEFAULT_MAXRESNSTR	"32.0"
#define  DEFAULT_QUANTERROR	0
#define  MINBITRATE		2.5

#undef BUFSIZ
#define  BUFSIZ 1024

#define  MAX_LPC_ORDER	64
#define  CHANSIZE	0
#define  ENERGYSIZE	3
#define  BITSHIFTSIZE	2
#define  NWRAP		3
#define  XBYTESIZE	7

#define  POSITIVE_ULAW_ZERO 0xff
#define  NEGATIVE_ULAW_ZERO 0x7f

#define  FNSIZE		2
#define  FN_DIFF0	0
#define  FN_DIFF1	1
#define  FN_DIFF2	2
#define  FN_DIFF3	3
#define  FN_QUIT	4
#define  FN_BLOCKSIZE	5
#define  FN_BITSHIFT	6
#define  FN_QLPC	7
#define  FN_ZERO	8

#define  TYPESIZE	4
#define  TYPE_AU1	0
#define  TYPE_S8	1
#define  TYPE_U8	2
#define  TYPE_S16HL	3
#define  TYPE_U16HL	4
#define  TYPE_S16LH	5
#define  TYPE_U16LH	6
#define  TYPE_ULAW	7
#define  TYPE_AU2	8
#define  TYPE_EOF	9
#define  TYPE_GENERIC_ULAW 128

#define  MASKTABSIZE	33
#define  ULONGSIZE	2
#define  NSKIPSIZE	1
#define  LPCQSIZE	2
#define  LPCQUANT	5
#define  NWRAP		3

#define  V2LPCQOFFSET (1 << LPCQUANT);

#ifndef	MIN
#define MIN(a,b) (((a)<(b))?(a):(b))
#endif

#ifndef	MAX
#define MAX(a,b) (((a)>(b))?(a):(b))
#endif

#define ROUNDEDSHIFTDOWN(x, n) (((n) == 0) ? (x) : ((x) >> ((n) - 1)) >> 1)

static uchar *getbuf;
static uchar *getbufp;
static int    nbyteget;
static ulong  gbuffer;
static int    nbitget;
static int sizeof_sample[TYPE_EOF];
static char *writebuf, *writefub;
static int  nwritebuf, nsampsdone, nsampswritten;
static int  badinput;
ulong masktab[MASKTABSIZE];

void init_sizeof_sample( void )
{
    sizeof_sample[TYPE_AU1]   = sizeof(char);
    sizeof_sample[TYPE_S8]    = sizeof(char);
    sizeof_sample[TYPE_U8]    = sizeof(char);
    sizeof_sample[TYPE_S16HL] = sizeof(short);
    sizeof_sample[TYPE_U16HL] = sizeof(short);
    sizeof_sample[TYPE_S16LH] = sizeof(short);
    sizeof_sample[TYPE_U16LH] = sizeof(short);
    sizeof_sample[TYPE_ULAW]  = sizeof(char);
    sizeof_sample[TYPE_AU2]   = sizeof(char);
}

ulong word_get()
{
    ulong buffer;

    if(nbyteget < 4) {
	nbyteget += fread((char*) getbuf, 1, BUFSIZ, fpin);
	if(nbyteget < 4) {
	    fprintf( stderr, "premature EOF on compressed stream in %s\n", inpname );
	    badinput++;
	    return(buffer);
	}
	getbufp = getbuf;
    }
    buffer = (((long) getbufp[0]) << 24) | (((long) getbufp[1]) << 16) |
	(((long) getbufp[2]) <<  8) | ((long) getbufp[3]);

    getbufp += 4;
    nbyteget -= 4;

    return(buffer);
}

long uvar_get( int nbin )
{
    long result;

    if(nbitget == 0) {
	gbuffer = word_get();
	nbitget = 32;
	if ( badinput )
	    return(result);
    }

    for(result = 0; !(gbuffer & (1L << --nbitget)); result++) {
	if(nbitget == 0) {
	    gbuffer = word_get();
	    nbitget = 32;
	    if ( badinput )
		return(result);
	}
    }

    while(nbin != 0) {
	if(nbitget >= nbin) {
	    result = (result << nbin) | ((gbuffer >> (nbitget-nbin)) & masktab[nbin]);
	    nbitget -= nbin;
	    nbin = 0;
	} 
	else {
	    result = (result << nbitget) | (gbuffer & masktab[nbitget]);
	    gbuffer = word_get(fpin);
	    nbin -= nbitget;
	    nbitget = 32;
	    if ( badinput )
		return(result);
	}
    }

    return(result);
}

ulong ulong_get( void )
{
    int nbit = uvar_get(ULONGSIZE);
    return(uvar_get(nbit));
}

void mkmasktab() 
{
    int i;
    ulong val = 0;

    masktab[0] = val;
    for(i = 1; i < MASKTABSIZE; i++) {
	val <<= 1;
	val |= 1;
	masktab[i] = val;
    }
}

long **long2d( ulong n0, ulong n1)
{
    long **array0;

    if((array0 = (long**) malloc(n0 * sizeof(long*) +n0*n1*sizeof(long)))!=NULL){
	long *array1 = (long*) (array0 + n0);
	int i;

	for(i = 0; i < n0; i++)
	    array0[i] = array1 + i * n1;
    }
    return(array0);
}

void init_offset( long **offset, int nchan, int nblock, int ftype )
{
    long mean = 0;
    int  chan, i;

    /* initialise offset */
    switch(ftype) {
      case TYPE_AU1:
      case TYPE_S8:
      case TYPE_S16HL:
      case TYPE_S16LH:
      case TYPE_ULAW:
      case TYPE_AU2:
	mean = 0;
	break;
      case TYPE_U8:
	mean = 0x80;
	break;
      case TYPE_U16HL:
      case TYPE_U16LH:
	mean = 0x8000;
	break;
      default:
	fprintf( stderr, "File %s has unknown/unsupported sample type: %d\n",
		inpname, ftype );
	badinput++;
	return;
    }

    for(chan = 0; chan < nchan; chan++)
	for(i = 0; i < nblock; i++)
	    offset[chan][i] = mean;
}

long var_get( int nbin )
{
    ulong uvar = uvar_get(nbin + 1);

    if(uvar & 1) return((long) ~(uvar >> 1));
    else return((long) (uvar >> 1));
}

void fwrite_type( long **data, int ftype, int nchan, int nitem )
{
    int hiloint = 1, hilo = !(*((char*) &hiloint));
    int i, nwrite = 0, chan, offset, limit;
    long *data0 = data[0];
    int nchanout = ( chanout ) ? chanout : 1;

    nsampsdone += nitem;
    if ( startout > nsampsdone || endout <= ( nsampsdone - nitem ))
	return;

    if(nwritebuf < nchan * nitem * sizeout) {
	nwritebuf = nchan * nitem * sizeout;
	if(writebuf != NULL) free(writebuf);
	if(writefub != NULL) free(writefub);
	writebuf = (char*) malloc((ulong) nwritebuf);
	writefub = (char*) malloc((ulong) nwritebuf);
	if ( writebuf == NULL || writefub == NULL ) {
	    fprintf( stderr, "No memory for %d byte output buffer on %s\n",
		    nwritebuf, inpname );
	    exit(1);
	}
    }

/* For "sph_convert", this next part is where uncompressed samples need extra (or
 * different) handling to support output options not provided by original "shorten":
 *  -- if original data was 2-channel, user may output both (dflt), or chan.1 or chan.2
 *  -- if output format is pcm RIFF, byte order must be LH, regardless of native order
 *  -- user may force ulaw data (AU2) to pcm output, or vice-versa
 *  -- user may force ulaw or pcm data to alaw output (alaw data cannot be shortened)
 *  -- user may select a time-slice from the file, skipping initial and/or final portions
 */
    switch (ftype) {
      case TYPE_AU1:		/* leave the conversion to fix_bitshift() */
      case TYPE_AU2:
	if ( typeout == PCM ) /* convert ulaw to pcm */
	{
	    short int *writebufp = (short int*) writebuf;
	    if(nchan == 1)
		for(i = 0; i < nitem; i++)
		    *writebufp++ = ulaw2pcm[data0[i]];
	    else if ( nchan > chanout )	/* this block (in each case) handles demux */
		for(i = 0; i < nitem; i++)
		    *writebufp++ = ulaw2pcm[data[chanout][i]];
	    else
		for(i = 0; i < nitem; i++)
		    for(chan = 0; chan < nchan; chan++)
			*writebufp++ = ulaw2pcm[data[chan][i]];
	} else if ( typeout == ALAW ) {  /* convert ulaw to alaw */
	    uchar *writebufp = (uchar*) writebuf;
	    if(nchan == 1)
		for(i = 0; i < nitem; i++)
		    *writebufp++ = pcm2alaw( ulaw2pcm[data0[i]] );
	    else if ( nchan > chanout )
		for(i = 0; i < nitem; i++)
		    *writebufp++ = pcm2alaw( ulaw2pcm[data[chanout][i]] );
	    else
		for(i = 0; i < nitem; i++)
		    for(chan = 0; chan < nchan; chan++)
		        *writebufp++ = pcm2alaw( ulaw2pcm[data[chan][i]] );
	} else {		/* leave ulaw as-is */
	    uchar *writebufp = (uchar*) writebuf;
	    if(nchan == 1)
		for(i = 0; i < nitem; i++)
		    *writebufp++ = data0[i];
	    else if ( nchan > chanout )
		for(i = 0; i < nitem; i++)
		    *writebufp++ = data[chanout][i];
	    else
		for(i = 0; i < nitem; i++)
		    for(chan = 0; chan < nchan; chan++)
			*writebufp++ = data[chan][i];
	}
	break;
      case TYPE_U8: {
	  uchar *writebufp = (uchar*) writebuf;
	  if(nchan == 1)
	      for(i = 0; i < nitem; i++)
		  *writebufp++ = data0[i];
	  else if ( nchan > chanout )
	      for(i = 0; i < nitem; i++)
		  *writebufp++ = data[chanout][i];
	  else
	      for(i = 0; i < nitem; i++)
		  for(chan = 0; chan < nchan; chan++)
		      *writebufp++ = data[chan][i];
	  break;
      }
      case TYPE_S8: {
	  char *writebufp = (char*) writebuf;
	  if(nchan == 1)
	      for(i = 0; i < nitem; i++)
		  *writebufp++ = data0[i];
	  else if ( nchan > chanout )
	      for(i = 0; i < nitem; i++)
		  *writebufp++ = data[chanout][i];
	  else
	      for(i = 0; i < nitem; i++)
		  for(chan = 0; chan < nchan; chan++)
		      *writebufp++ = data[chan][i];
	  break;
      }
      case TYPE_S16HL:
      case TYPE_S16LH: {
	  if ( typeout == ULAW ) { /* convert pcm to ulaw */
	      char *writebufp = writebuf;
	      if(nchan == 1)
		  for(i = 0; i < nitem; i++)
		      *writebufp++ = pcm2ulaw( data0[i] );
	      else if ( nchan > chanout )
		  for(i = 0; i < nitem; i++)
		      *writebufp++ = pcm2ulaw( data[chanout][i] );
	      else
		  for(i = 0; i < nitem; i++)
		      for(chan = 0; chan < nchan; chan++)
			  *writebufp++ = pcm2ulaw( data[chan][i] );
	  } else if ( typeout == ALAW ) {  /* convert pcm to alaw */
	      char *writebufp = writebuf;
	      if(nchan == 1)
		  for(i = 0; i < nitem; i++)
		      *writebufp++ = pcm2alaw( data0[i] );
	      else if ( nchan > chanout )
		  for(i = 0; i < nitem; i++)
		      *writebufp++ = pcm2alaw( data[chanout][i] );
	      else
		  for(i = 0; i < nitem; i++)
		      for(chan = 0; chan < nchan; chan++)
			  *writebufp++ = pcm2alaw( data[chan][i] );
	  } else {   /* leave pcm as-is */
	      short *writebufp = (short*) writebuf;
	      if(nchan == 1)
		  for(i = 0; i < nitem; i++)
		      *writebufp++ = data0[i];
	      else if ( nchan > chanout )
		  for(i = 0; i < nitem; i++)
		      *writebufp++ = data[chanout][i];
	      else
		  for(i = 0; i < nitem; i++)
		      for(chan = 0; chan < nchan; chan++)
			  *writebufp++ = data[chan][i];
	  }
	  break;
      }
      case TYPE_U16HL:
      case TYPE_U16LH: {
	  ushort *writebufp = (ushort*) writebuf;
	  if(nchan == 1)
	      for(i = 0; i < nitem; i++)
		  *writebufp++ = data0[i];
	  else if ( nchan > chanout )
	      for(i = 0; i < nitem; i++)
		  *writebufp++ = data[chanout][i];
	  else
	      for(i = 0; i < nitem; i++)
		  for(chan = 0; chan < nchan; chan++)
		      *writebufp++ = data[chan][i];
	  break;
      }
      case TYPE_ULAW: {
	  uchar *writebufp = (uchar*) writebuf;
	  if(nchan == 1)
	      for(i = 0; i < nitem; i++)
		  *writebufp++ = pcm2ulaw(data0[i] << 3);
	  else if ( nchan > chanout )
	      for(i = 0; i < nitem; i++)
		  *writebufp++ = data[chanout][i];
	  else
	      for(i = 0; i < nitem; i++)
		  for(chan = 0; chan < nchan; chan++)
		      *writebufp++ = pcm2ulaw(data[chan][i] << 3);
	  break;
      }
    }
    if ( startout > ( nsampsdone - nitem ))
	offset = startout - ( nsampsdone - nitem );
    else
	offset = 0;
    if ( endout < nsampsdone )
	limit = ( endout - ( nsampsdone - nitem ));
    else
	limit = nitem;
    limit -= offset;
    offset *= sizeout * nchanout;
	
    switch(ftype) {
      case TYPE_AU1:
      case TYPE_S8:
      case TYPE_U8:
      case TYPE_ULAW:
      case TYPE_AU2:
	if( sizeout == PCM && strcmp( nativorder, outorder )) {
	    swab(writebuf, writefub, sizeout * nchanout * nitem);
	    nwrite = fwrite( &writefub[offset], sizeout * nchanout, limit, fpout);
	} else
	    nwrite = fwrite( &writebuf[offset], sizeout * nchanout, limit, fpout);
	break;
      case TYPE_S16HL:
      case TYPE_U16HL:
	if( sizeout == PCM && strcmp( nativorder, outorder )) {
	    swab(writebuf, writefub, sizeout * nchanout * nitem);
	    nwrite = fwrite( &writefub[offset], sizeout * nchanout, limit, fpout);
	} else
	    nwrite = fwrite( &writebuf[offset], sizeout * nchanout, limit, fpout);
	break;
      case TYPE_S16LH:
      case TYPE_U16LH:
	if( sizeout == PCM && strcmp( nativorder, outorder )) {
	    swab(writebuf, writefub, sizeout * nchanout * nitem);
	    nwrite = fwrite( &writefub[offset], sizeout * nchanout, limit, fpout);
	}
	else
	    nwrite = fwrite( &writebuf[offset], sizeout * nchanout, limit, fpout);
	break;
    }

    if(nwrite != limit) {
	fprintf( stderr, "failure writing uncompressed data from %s to %s\n",
		inpname, outname );
	exit(1);
    }
}

void fix_bitshift( long *buffer, int nitem, int bitshift, int ftype )
{
    int i;

    if(ftype == TYPE_AU1)
	for(i = 0; i < nitem; i++)
	    buffer[i] = ulaw_outward[bitshift][buffer[i] + 128];
    else if(ftype == TYPE_AU2)
	for(i = 0; i < nitem; i++) {
	    if(buffer[i] >= 0)
		buffer[i] = ulaw_outward[bitshift][buffer[i] + 128];
	    else if(buffer[i] == -1)
		buffer[i] =  NEGATIVE_ULAW_ZERO;
	    else
		buffer[i] = ulaw_outward[bitshift][buffer[i] + 129];
	}
    else
	if(bitshift != 0)
	    for(i = 0; i < nitem; i++)
		buffer[i] <<= bitshift;
}

/*************************************************************
 * shortenXtract
 *------------------------------------------------------------
 * Adapted by Dave Graff from a portion of Tony Robinson's "shorten.c";
 *  -- changed variable names as needed to work with sph_convert
 *  -- added logic to handle ulaw->pcm conversion and demux
 */
int shortenXtract( void )
{
    long **buffer, *buffer1, **offset;
    int version, nmean, ftype, nchan, blocksize, nskip, nwrap, chan;
    int *qlpc, maxnlpc, lpcqoffset, i, cmd, nscan, bitshift = 0;
    char *magic = MAGIC;

    nscan = 0;

    version = MAX_VERSION + 1;
    while(version > MAX_VERSION) {
	int byte = getc(fpin);
	if(magic[nscan] != '\0' && byte == magic[nscan]) nscan++;
	else if(magic[nscan] == '\0' && ( byte == 1 || byte == 2 )) version = byte;
	else {
	    fprintf( stderr, "Can't interpret start of %s as usable shortened data\n",
		    inpname );
	    return 1;
	}
    }
    /* set up the default nmean */
    nmean = (version < 2) ? DEFAULT_V0NMEAN : DEFAULT_V2NMEAN;

    /* initialise the variable length file read for the compressed stream */
    mkmasktab();
    getbuf   = (uchar*) malloc( BUFSIZ );
    getbufp  = getbuf;
    nbyteget = 0;
    gbuffer  = 0;
    nbitget  = 0;
    
    /* initialise the fixed length file write for the uncompressed stream */
    init_sizeof_sample();
    writebuf  = (char*) NULL;
    writefub  = (char*) NULL;
    nwritebuf = 0;
    nsampsdone = nsampswritten = 0;

    /* get file type and set up appropriately */
    ftype = ulong_get();

    if ( ftype >= TYPE_EOF ) {
	fprintf( stderr, "%s contains shortened sample-type %d; can't handle it\n",
		inpname, ftype );
	return 1;
    }

    nchan = ulong_get();
    blocksize = ulong_get();
    maxnlpc = ulong_get();
    nmean = ulong_get();
    nskip = ulong_get();
    for ( i=0; i<nskip; i++ ) {
	int byte = uvar_get(XBYTESIZE);
	if ( putc( byte, fpout ) != byte ) {
	    fprintf( stderr,"write failed on %s\n", outname );
	    exit(1);
	}
    }

    nwrap = MAX(NWRAP, maxnlpc);

    /* grab some space for the input buffer */
    buffer  = long2d((ulong) nchan, (ulong) (blocksize + nwrap));
    offset  = long2d((ulong) nchan, (ulong) MAX(1, nmean));

    for(chan = 0; chan < nchan; chan++) {
	for(i = 0; i < nwrap; i++) buffer[chan][i] = 0;
	buffer[chan] += nwrap;
    }

    if(maxnlpc > 0)
	qlpc = (int*) malloc((ulong) (maxnlpc * sizeof(*qlpc)));

    if(version > 1)
	lpcqoffset = V2LPCQOFFSET;

    init_offset(offset, nchan, MAX(1, nmean), ftype);
    if ( badinput )
	return 1;

    /* get commands from file and execute them */
    chan = 0;
    while((cmd = uvar_get(FNSIZE)) != FN_QUIT)
	switch(cmd) {
	  case FN_ZERO:
	  case FN_DIFF0:
	  case FN_DIFF1:
	  case FN_DIFF2:
	  case FN_DIFF3:
	  case FN_QLPC: {
	      long coffset, *cbuffer = buffer[chan];
	      int resn, nlpc, j;

	      if(cmd != FN_ZERO)
		  resn = uvar_get(ENERGYSIZE);

	      /* find mean offset : N.B. this code duplicated */
	      if(nmean == 0) coffset = offset[chan][0];
	      else {
		  long sum = (version < 2) ? 0 : nmean / 2;
		  for(i = 0; i < nmean; i++) sum += offset[chan][i];
		  if(version < 2)
		      coffset = sum / nmean;
		  else
		      coffset = ROUNDEDSHIFTDOWN(sum / nmean, bitshift);
	      }

	      switch(cmd) {
		case FN_ZERO:
		  for(i = 0; i < blocksize; i++)
		      cbuffer[i] = 0;
		  break;
		case FN_DIFF0:
		  for(i = 0; i < blocksize; i++)
		      cbuffer[i] = var_get(resn) + coffset;
		  break;
		case FN_DIFF1:
		  for(i = 0; i < blocksize; i++)
		      cbuffer[i] = var_get(resn) + cbuffer[i - 1];
		  break;
		case FN_DIFF2:
		  for(i = 0; i < blocksize; i++)
		      cbuffer[i] = var_get(resn) + (2 * cbuffer[i - 1] -
						    cbuffer[i - 2]);
		  break;
		case FN_DIFF3:
		  for(i = 0; i < blocksize; i++)
		      cbuffer[i] = var_get(resn) + 3 * (cbuffer[i - 1] -
							cbuffer[i - 2]) + cbuffer[i - 3];
		  break;
		case FN_QLPC:
		  nlpc = uvar_get(LPCQSIZE);

		  for(i = 0; i < nlpc; i++)
		      qlpc[i] = var_get(LPCQUANT);
		  for(i = 0; i < nlpc; i++)
		      cbuffer[i - nlpc] -= coffset;
		  for(i = 0; i < blocksize; i++) {
		      long sum = lpcqoffset;

		      for(j = 0; j < nlpc; j++)
			  sum += qlpc[j] * cbuffer[i - j - 1];
		      cbuffer[i] = var_get(resn) + (sum >> LPCQUANT);
		  }
		  if(coffset != 0)
		      for(i = 0; i < blocksize; i++)
			  cbuffer[i] += coffset;
		  break;
	      }

	      /* store mean value if appropriate : N.B. Duplicated code */
	      if(nmean > 0) {
		  long sum = (version < 2) ? 0 : blocksize / 2;

		  for(i = 0; i < blocksize; i++)
		      sum += cbuffer[i];

		  for(i = 1; i < nmean; i++)
		      offset[chan][i - 1] = offset[chan][i];
		  if(version < 2)
		      offset[chan][nmean - 1] = sum / blocksize;
		  else
		      offset[chan][nmean - 1] = (sum / blocksize) << bitshift;
	      }

	      /* do the wrap */
	      for(i = -nwrap; i < 0; i++) cbuffer[i] = cbuffer[i + blocksize];

	      fix_bitshift(cbuffer, blocksize, bitshift, ftype);

	      if(chan == nchan - 1) {
		  fwrite_type( buffer, ftype, nchan, blocksize );
		  if ( nsampsdone >= endout )
		      return 0;
	      }
	      chan = (chan + 1) % nchan;
	  }
	    break;
	  case FN_BLOCKSIZE:
	    blocksize = ulong_get();
	    break;
	  case FN_BITSHIFT:
	    bitshift = uvar_get( BITSHIFTSIZE );
	    break;
	  default:
	    fprintf( stderr, "sanity check fails trying to decode function %d in %s\n",
		    cmd, inpname);
	}
    return 0;
}
