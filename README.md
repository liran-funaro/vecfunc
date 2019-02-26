# vecfunc

Vectorized function representation for a generic function.

# Features
- Represent a multi dimensional function using a vector/matrix/tensor.
- Calculate the function values in its domain boundaries using various interpolation methods.
- Inspect function properties.
- Calculate gradients.
- Calculate expected value given a distribution function (in the form of a CDF).
- Visualize the function (up to 3D functions).
- Generate random multi dimensional functions.
  * Function smoothing using Chaikin's Corner Cutting Scheme.
- Specific compiled binary (C++) for different number of dimensions and different data type.
  * Unrolls loops (over each dimension) for best performance.

In addition, an implementation of Bresenham's line algorithm in N-dimensions is included.


# Install (beta)
Install `g++-8`:
```bash
apt-get install g++-8 
```

The Python library will compile the binary upon usage.
To precompile for all data types and for all dimensions, use the included script: `vecfunclib/makeall.sh`.

See Python's requirements in the [REQUIREMENTS](REQUIREMENTS.txt) file.

Finally, install the package in developer mode:
```bash
python setup.py develop --user
```


# License
[GPL](LICENSE.txt)