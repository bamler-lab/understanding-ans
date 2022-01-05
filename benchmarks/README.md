# Benchmark Code And Results

## Results Reported in the Paper

You can find the benchmark results from which the plots in the paper were generated, as well as the code to generate the plots in the subdirectory [`plots`](https://github.com/bamler-lab/understanding-ans/blob/main/benchmarks/plots).

## Rerunning the Benchmark

To compile and run the benchmarks yourself, do the following.

- `cd` into this directory and run `./download-dwe.sh` to download the data set.
- [Install Rust](https://rustup.rs/).
- Run the benchmarks by typing (in this directory):
  - to get just the entropies and bitrates of each one of the 209 slices:<br>
    `cargo run --release -- --bitrates > my_bitrates.py`
  - to also measure run time performance (takes much longer because it measures each run
    time 10 times and introduces some cool-down time in between, just to be on the safe side):<br>
  `cargo run --release -- --runtimes --runtimes-count 10 --pause 15 > my_runtimes.py`

Benchmark results will be written out as valid python files, so you can easily just `import`
them into a jupyter notebook to plot them (see [jupyter
notebook](https://github.com/bamler-lab/understanding-ans/blob/main/benchmarks/plots/plots.ipynb)
in the subdirectory `plots`).
