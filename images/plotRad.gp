set term pngcairo

datafile='rad400.dat'

stats datafile

unset border
unset tics

set lmargin 0
set rmargin 0
set tmargin 0
set bmargin 0

unset colorbox

total_block = STATS_blocks-2

do for [ii=0:total_block] {
   filename = sprintf ( "rad%03d.png",ii )
   set output filename	   
   plot datafile index ii matrix with image
   system( sprintf("echo plotting image %d of %d",ii,total_block) ) 
}