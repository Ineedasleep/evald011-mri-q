UCR CS217 Final Project
Ethan Valdez
NetID: evald011

Computation of a matrix Q, representing the scanner configuration, used in 
a 3D magnetic resonance image reconstruction algorithm in non-Cartesian space.

Our task is to accelerate this computation using GPGPUs.  Your goal is to 
make the GPU kernel execution as fast as you can with the following restriction:

    The results must be deterministic and match the result of the
    sequential code (within rounding errors).  This means you may not use the fast math
	versions of sin and cos, and the order of accumulation
	operations must be the same.  While some optimizations can
	trade off accuracy for speed, we're asking you to maintain
	current semantics exactly.

The given interface for the application is as follows:

    You must specify using the -i option the input file.  The dataset 
    directory includes three different size input files.

    You may specify the option -S to get more accurate timings (inserts 
    synchronization after non-blocking events).  This is how we will 
    measure your final speed. (NOT CURRENTLY IMPLEMENTED)

    You may specify an output file using the -o option.  You can then 
    analyse the output file however you like, including comparing it to 
    other output files using compareFiles.cc in the mri-q directory.

    You may specify as the last command line parameter an integer number 
    to limit the number of input samples used.  This can be useful in testing
    or verifying your code in a shorter amount of time.  For reference, we 
    also provide correct output files for using 512 or 10000 samples.  Keep in 
    mind that your optimizations should not put restrictions on the number of 
    samples you may be provided with as input, although you could potentially
    pad or otherwise handle it internally.
