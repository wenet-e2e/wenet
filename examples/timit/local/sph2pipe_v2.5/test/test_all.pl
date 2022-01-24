#!/usr/bin/env perl

# Test plan to verify sph2pipe (v2.4)

# We start with 7 distinct input files:

# 123_2pcle_shn.sph -- 2 channel, little-endian PCM, shorten-compressed
# 123_2pcbe_shn.sph -- 2 channel, big-endian PCM, shorten-compressed
# 123_1pcle_shn.sph -- 1 channel, little-endian PCM, shorten-compressed
# 123_1pcbe_shn.sph -- 1 channel, big-endian PCM, shorten-compressed
# 123_2ulaw_shn.sph -- 2 channel, ulaw, shorten-compressed
# 123_1ulaw_shn.sph -- 1 channel, ulaw, shorten-compressed
# 123_2alaw.sph     -- 2 channel, alaw (shorten does not apply to alaw)

# We'll test these groups of conditions in all possible combinations:

# 8 input format types: pcm-BE-shn, pcm-LE-shn, ulaw-shn,
#                       pcm-BE, pcm-LE, ulaw, alaw, alaw with external header
# 2 input channel types: single-channel, two-channel

# 3 output range selections: full-file, -t bsec:esec, -s bsmp:esmp
# 3 output channel selections: both-channels, channel-A, channel-B

# That's 144 combinations to test, for each of the following
# conditions of output format and sample type:

# 3 raw: alaw, ulaw, pcm-NATIVE (i.e. byte-order of current cpu)
# 3 sph: alaw, ulaw, pcm-NATIVE
# 3 wav: alaw, ulaw, pcm-LE
# 2 au:  ulaw, pcm-BE
# 1 aif: pcm-BE

# That's 12 output types * 144 combinations per type = 1728 tests to run
# plus 7 more, to cover the following extra cases:

# - an extra-long sphere header (2 runs)
# - final colon in the -t and -s opions (2 runs)
# - initial colon in the -t and -s options (2 runs)
# - redirection to a file vs. named output file (1 run)

# The chosen test file has a sample_rate of 20_000, so we'll simplify
# the -t and -s range tests by using equivalent ranges: '-t 0.9:1.2'
# and '-s 18000:24000' should produce the same output.

use Digest::MD5;

my @t_opts = ( '', '-t 0.9:1.2', '-s 18000:24000' );
my @c_opts = ( '', '-c 1', '-c 2' );
my @f_opts = ( '-f sph -p', '-f sph -u', '-f sph -a',
	       '-f raw -p', '-f raw -u', '-f raw -a',
	       '-f wav -p', '-f wav -u', '-f wav -a',
	       '-f au -p',  '-f au -u',  '-f aif' );

my @shn_inputs = ( qw/
		  123_1pcle_shn.sph 123_2pcle_shn.sph
		  123_1pcbe_shn.sph 123_2pcbe_shn.sph
		  123_1ulaw_shn.sph 123_2ulaw_shn.sph
		  / );
my @unc_inputs = ( qw/
		  123_1pcle.sph 123_2pcle.sph
		  123_1pcbe.sph 123_2pcbe.sph
		  123_1ulaw.sph 123_2ulaw.sph
		  123_1alaw.sph 123_2alaw.sph
		  / );
my @raw_inputs = ( qw/
		  123_1alaw.raw  123_2alaw.raw
		  / );

open( my $SH, "| /bin/sh" ) or die "can't start sub-shell /bin/sh: $!";
open( my $CKSUMS, ">outfile-md5.list" ) or die "can't write md5 file: $!";

my $s2p = "./sph2pipe";
my %output_map;

for my $i ( 0..$#shn_inputs ) {
    print_SH( "$s2p -f sph $shn_inputs[$i] $unc_inputs[$i]\n" );
}
print_SH( "$s2p -f sph -c 1 123_2alaw.sph 123_1alaw.sph\n" );
print_SH( "$s2p -f raw 123_1alaw.sph 123_1alaw.raw\n" );
print_SH( "$s2p -f raw 123_2alaw.sph 123_2alaw.raw\n" );

my $out = "0000";

for my $f ( @f_opts ) {
    my $ext = ( split( / /, $f ))[1];
    for my $c ( @c_opts ) {
	for my $t ( @t_opts ) {
	    for my $in ( @shn_inputs, @unc_inputs ) {
		$out++;
		print_SH( "$s2p $f $c $t $in $out.$ext\n" );
	    }
	    for my $in ( @raw_inputs ) {
		my ( $nc ) = ( $in =~ /_([12])/ ); 
		$out++;
		print_SH( "$s2p $f $c $t -h std$nc.hdr $in $out.$ext\n" );
		$out++;
		print_SH( "$s2p $f $c $t -h big$nc.hdr $in $out.$ext\n" );
	    }
	}
    }
}
close $SH;

my $md5 = Digest::MD5->new;
for my $f ( sort keys %output_map ) {
    $md5->reset;
    if ( -s $f > 0 ) {
        open( my $CHK, $f );
    	printf $CKSUMS ( "%s %-8s <- %s\n", $md5->addfile( $CHK )->b64digest,
	       	         $f, $output_map{$f} );
    	close $CHK;
    } else {
	warn "$output_map{$f} -> $f: zero-length output\n" unless -s $f;
    }
}

sub print_SH
{
    my @cmd = split( ' ', $_[0] );
    my $msg = join " ", @cmd[1..$#cmd];
    my $outname = pop @cmd;
    $output_map{$outname} = pop @cmd;

    print $SH "echo $msg\n";
    print $SH $_[0];
}
