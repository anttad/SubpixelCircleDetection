The Canny/Devernay algorithm
============================

Version 1.0 - October 10, 2017
by Rafael Grompone von Gioi <grompone@gmail.com>
   and Gregory Randall <randall@fing.edu.uy>


Introduction
------------

This is an implementation of Canny/Devernay's sub-pixel edge detector. This
code is part of the following publication and was subject to peer review:

  "A Sub-Pixel Edge Detector: an Implementation of the Canny/Devernay Algorithm"
  by Rafael Grompone von Gioi and Gregory Randall,
  Image Processing On Line, 2017. DOI:10.5201/ipol.2017.216
  http://dx.doi.org/10.5201/ipol.2017.216


Files
-----

README.txt     - This file
COPYING        - GNU AFFERO GENERAL PUBLIC LICENSE Version 3
Makefile       - Compilation instructions for 'make'
devernay.c     - Devernay module ANSI C89 code (peer reviewed)
devernay.h     - Devernay module ANSI C89 header (peer reviewed)
devernay_cmd.c - Command line interface for Devernay, ANSI C89
io.c           - Input/Output functions for command interface, ANSI C89
io.h           - Input/Output functions header, ANSI C89
image.pgm      - Test image in PGM format
image.asc      - Test image in ASC format
image_out.txt  - Expected result for the test image as an TXT file
image_out.pdf  - Expected result for the test image as a PDF file
image_out.svg  - Expected result for the test image as a SVG file

The files 'devernay.c' and 'devernay.h' were subject to peer review as part of
the acceptance process of the IPOL article and are the official version of
Devernay.


Compiling
---------

Devernay is an ANSI C89 Language program and can be used as a module to
be called from a C language program or as an independent command.

In the distribution is included a Makefile file with instructions to build the
command line program 'devernay'. A C compiler must be installed on your system
as well as the program 'make'. Devernay only uses the standard C library so it
should compile in any ANSI C89 Language environment. In particular, it should
compile in a Unix like system.

The compiling instruction is just

  make

from the directory where the source codes and the Makefile are located. If the
compilation needs to be made manually, the usual command to do it is:

  cc -o devernay devernay_cmd.c io.c devernay.c -lm

To verify a correct compilation you can apply Devernay to the test image
'image.pgm' and compare the result to the provided ones. This can be done by
executing:

  make test


Running the Command Line Interface
----------------------------------

The simplest Devernay command execution is just

  devernay

(use ./devernay if the command is not included in the current path). That
should print Devernay version number and a description of the command line
interface, including the available options. The input image formats handled are
PGM (in its two versions, ASCII and Binary) and the ASC format (as defined by
the CImg Library and as described below). A typical execution would be:

  devernay image.pgm -t output.txt

That should give the result as an TXT file 'output.txt' which consists of two
columns of real numbers in ASCII format (the numbers are separated by a
space). Each row corresponds to a contour point, the first column gives the x
coordinate and the second row gives the y coordinate. Contour points of
consecutive rows are part of the same curve, they are chained. Each curve is
ended by a row "-1 -1", which mark the end of the curve or chain. A new chain
may start on the next row. Then end of the file indicates that no more curves
are present. For closed curves, the first point of the curve is repeated again
as the last point of the chain. The following is an example of output:

  9 5.57116
  10 5.73102
  11 6.64051
  11.9352 8
  12.0939 9
  11.9844 10
  11.4029 11
  10 12.0179
  9 12.119
  8 11.9399
  6.62612 11
  5.65646 10
  5.51349 9
  5.62126 8
  7 6.54667
  8 5.68361
  9 5.57116
  -1 -1
  2.09193 5
  2.17282 4
  3 2.7223
  4 2.13559
  5 1.99507
  -1 -1

It corresponds to two curves, each one ended by a "-1 -1" row. The first one is
a closed curve described by 16 points. Please note that the list has 17 points,
but the first and last are the same, "9 5.57116", indicating a closed
curve. The second curve is an open curve described by five points. Note that
one of the two coordinates is always an integer. This is due the modified
Devernay sub-pixel correction, in which the sub-pixel interpolation is always
performed along the vertical or horizontal axis, but not both; thus, one of the
two coordinates is not interpolated and remain an integer.

For easy visualization of the result, the command line interface can also
provide the output in PDF and SVG file formats:

  devernay image.pgm -p output.pdf

will produce the PDF file 'output.pdf' and

  devernay image.pgm -g output.svg

will produce the SVG file 'output.svg'. Using all these options will generate
TXT, PDF and SVG outputs:

  devernay image.pgm -p output.pdf -t output.txt -g output.svg

Note that the line width used in the PDF and SVG output is arbitrary. The
default value is 1.3 (in pixel units) but can be modified with the -w
option. By using a smaller value, the sub-pixel accuracy of the result is
better appreciated:

  devernay image.pgm -p output.pdf -w 0.5

The width value can be set to zero:

  devernay image.pgm -p output.pdf -g output.svg -w 0

In this case, the PDF and SVG standards determine different behavior. In PDF,
a zero width means that the software rendering the PDF will choose the smallest
possible line width that can be rendered in the current device. This is useful
to see details but it is not recommended for figures to be distributed as the
rendering is device dependent. In SVG, however, line with zero width are not
drawn.

The Canny/Devernay algorithm depends on three parameters: sigma, th_low,
th_high. The default value of these parameters is zero for the three of them.
Their value can be modified individually using the -s, -l and -h options of the
command line interface. For example:

  devernay image.pgm -p output.pdf -s 1 -l 5 -h 15

will set the standard deviation of the Gaussian filtering to 1, the low
gradient threshold to 5 and the high gradient threshold to 15.


The ASC file format
-------------------

The ASC image file format was defined by the CImg Library (http://cimg.eu/). It
is a simple ASCII format allowing to store 4D arrays of floating point values
(and not just integer as in PGM). In addition, multiple channels and multiple
frames are possible; the command line interface of Devernay, however, only
handles a single gray level frame. The main reason to include this file format
as a possible input is to be able to use floating point images in a simple
way. As an example, the file 'image.asc' in this distribution contains the same
image as 'image.pgm' but in ASC file format.

To explain the ASC file format, the following is an example of a 6x6 floating
point gray-level in ASC format:

  6 6 1 1
  0.0022 0.1119 0.9455 1.7084 0.9455 0.1119
  0.1119 1.7270 5.8871 7.4892 5.8871 1.7270
  0.9455 5.8871 9.6757 9.9586 9.6757 5.8888
  1.7084 7.4892 9.9586 9.9999 9.9586 7.4934
  0.9455 5.8871 9.6757 9.9586 9.6757 5.8888
  0.1119 1.7270 5.8888 7.4934 5.8888 1.7271

An ASC file consists of a simple header followed by the data, all written as
numbers in ASCII format. The header is just four integer numbers separated by
spaces and ending in a new line. The four numbers of the header correspond to
the size of each of the 4 dimensions X, Y, Z and C and in that order: X is the
width of the image; Y is the height of the image; Z is the number of frames; C
is the number of channels. For single frame and single channel (as in the
example above and as required by Devernay), Z=1 and C=1.

The header is followed by the data written in standard ASCII floating point
notation. Any format accepted by the standard C function scanf is valid; for
example: 0, 692315, 0.34230295282, -30.3423, +45, 1e10, 5E4, 3.14e-34. The
numbers should be separated by any combination of spaces, tabs, or end-of-line.
The order of the numbers is important. The pixels have coordinates (x,y,z,c),
with 0 <= x < X and 0 <= y < Y and 0 <= z < Z and 0 <= c < C. The pixel
(x,y,z,c) will be stored at position x + y*X + z*X*Y + c*X*Y*Z. Thus, the first
number following the header corresponds to pixel (0,0,0,0) which is the upper
left pixel of the first channel of the first frame. The second value is the
second pixel on the same row (1,0,0,0) and the following values complete the
first row (up to X-1,0,0,0). Then comes the second row of the first channel of
the first frame, from (0,1,0,0) to (X-1,1,0,0). Then the following rows and so
on until the last row of the frame, (0,Y-1,0,0) to (X-1,Y-1,0,0). This would
complete the image if only one frame and one channel are present (as required
in Devernay and as in the example above).

In multiple frame and multiple channel (Z>1 and/or C>1), the next values will
store, in the same way, the first channel of the second frame. When all the
frames are complete, the same procedure will be done for the values of the
second channel, and so on.


Copyright and License
---------------------

Copyright (c) 2016-2017 Rafael Grompone von Gioi and Gregory Randall

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.


Thanks
------

We would be grateful to receive any comment, especially about errors, bugs,
or strange results.
