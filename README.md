# mugrid
Demonstration of 1D multigrid solver. For a full description, see the [post on my site.][post]

# Contents
The two main files are

1. linoplib.py: This file has the code which generates the linear operators needed for multigrid, including functions which generate the discrete laplacian, prolongation, and restriction operators. Particularly useful is `get_good_grid_sizes(N)`, which lists grid sizes which work well with multigrid, i.e. they can be coarsened many times. If you use a grid size that can't be coarsened with these operators very much, you won't see much speedup (if any) over the Jacobi method alone.
2. solvers.py: The Jacobi, VMG, and FMG solvers are implemented here.

In the workspace directory, you'll find a bunch of scripts which I used to generate the plots in the article I wrote.

[post]: https://peytondmurray.netlify.app/blog/2019-02-12-multigrid-solve-odes/
