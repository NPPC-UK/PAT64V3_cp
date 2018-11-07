# PAT64V3_cp

This software partially analyses images taken by the 
[Medium to Large dynamic phenotyping platform](https://www.plant-phenomics.ac.uk/index.php/resources/index.php/resources/lemnatec-system/) 
at the [NPPC](https://www.plant-phenomics.ac.uk). 

This software is based on several earlier versions of PAT64V3 in order to
correctly analyse images taken with white or black backgrounds. It has also been
ported to run on Linux (though it is likely to compile and run correctly on Unix 
systems and MS Windows with little to no modification).

## Outputs

PAT64V3 will calculate things like the height of the plant, as well as the area presented to the camera.

## Dependencies/Installing/Running

This software depends on [OpenCV](https://opencv.org/), [Boost.Filesystem](https://www.boost.org/doc/libs/1_39_0/libs/filesystem/doc/index.htm),
and compiles with [GCC](https://gcc.gnu.org/).  

To compile, ensure that the dependencies exist on your system and run `make all` in the project directory.  This will produce three executables:
`PAT32_wb`, `PAT32_bb` and `PAT32_clover` for analysing data from the Medium to Large dynamic phenotyping platform taken against white and dark backgrounds
as well as data taken from clover grown in gutters respectively.

## Author

This software was initially written by [Dr. Jiwan Han](https://www.plant-phenomics.ac.uk/index.php/about/meet-the-team/). 
It was refactored, ported and the file handling and I/O operations rewritten by [Maximilian Friedersdorff](https://git.friedersdorff.com/explore/repos)
