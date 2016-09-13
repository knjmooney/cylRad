cylRad
======

cylRad models a cylindrical radiator. The radiator has a column running straight 
down that is heated from the top, from the column spans circular rows which make
the cylindrical shape of the radiator. The column can be imagined as a human
spine and the rows the rib cages, except the rows complete a full circle. The
rows are insulated from each other. The heat in the column stays constant and 
tends to propagate to the right. 

This radiator can be modeled by a 2D matrix with values rad(i,j,t) where i and j 
are the indexes of the matrix and t is the time that has elapsed since the 
radiator was turned on. If j=0,1 is the column, the radiator can be initialised 
with

    rad(i,0,0) = 1.00*(i+1) / n
    rad(i,1,0) = 0.75*(i+1) / n.
    rad(i,j,0) = 0.00 for j != 0,1. 

Then the heat propagation is modeled by 

    rad(i,j,t)   =     1.85*oldRad(i,j-2)
        	     + 1.40*oldRad(i,j-1)
		     + 1.00*oldRad(i,j  )
		     + 0.60*oldRad(i,j+1)
		     + 0.15*oldRad(i,j+2)

    rad(i,j,t)   =   rad(i,j) / 5.0;

    for all j != 0,1 and t > 0
      
cylRad is modelled using a c++11 and CUDA 6.5. To install, run

    make

It was compiled and tested on Ubuntu 14.04.4 LTS with the following 

Software:
    nvcc V6.5.12
    gcc V4.8.4

Hardware:
    Tesla K40c

To generate an animation, run

    make animation

The software required to generate the animation is

    gnuplot V4.6
    ImageMagick V6.7.7
   
The animation is saved as images/rad.gif. The following help message is printed
when cylRad is run with the -h flag

    Usage: ././bin/cylRad [options] ...
    Options:
       -f str                    file to print radiator to
       -h                        show this help
       -m int                    set the number of rows
       -n int                    set the number of cols
       -p int                    set the number of iterations
       -v                        verbose

The number of rows and columns must be a multiple of 32. If run with the -v flag,
the program will print the number of discrepancies between the GPU and CPU results