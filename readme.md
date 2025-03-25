# Automater for running Experiments

Just follow this readme and run the code snippets, everything is automated.

>ALL data including raw CSV, graphs, decision tree PDFs, rules of the tree and node info for each experiment run are stored in the `CSV` folder of the repo.


### 0. directory structure

This code assumes a user called `gpuio` is present in the`/home` directory.

```bash
cd /home/gpuio/
git clone https://github.com/aalhadsawane/gpuIO.git
cd gpuIO
```

### 1. install hdf5 LTS in gpuIO user.

```bash
./build_hdf5.sh
```

### 2. install h5bench - the benchmark suite

```bash
./build_h5bench.sh
```

*NOTE*: What has been done till now (Step 0 to 2) needs to be done on a node only once as setup.

The steps ahead will be reused every experiment.

### 3. Set experiment parameters

>Edit `benchmark_config.conf` in source dir of repo.

>Set number of IO threads, dataset sizes, block sizes and IO modes there.

### 4. Run and get h5bench data in csv

Run benchmarking script to the conf file mentioned above *in background* 

```bash
nohup ./run_h5bench.sh > h5bench_output.log 2>&1 &
```

The output (raw_output.csv) is stored in a new folder `/home/gpuio/gpuIO/CSV/Run*/` 

where `Run*` indicates latest Run folder (made automatically)

A txt file called `nodename.txt` inside the Run folder is also made which stores the hostname (ex `colva2`).

u can check the `h5bench_output.log` file to check where the experiment has reached. (Look for `Benchmark progress`).

To kill the experiment midway: ```ps aux | grep run_h5bench.sh``` and kill the PID.
### 5. Plot the csv

Create a venv with
```bash
virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> My plotting script is given as `plot.py` in source dir of repo. Please make a copy of it if u wanna edit it; dont change this one.
#### Usage:

```bash
sudo apt-get install graphviz
python plot.py
```

It will take the input csv as `CSV/Run*/raw_output.csv` which is the latest output CSV made by the `run_h5bench.sh` script.

It will store all `PNG` images in `CSV/Run*/graphs`
where `Run*` indicates latest Run.

It will also make a decision tree as a `PDF` and a `txt` file defining rules of the decision tree in `CSV/RUN*`, 

###$ Custom csv path and output path:

>plot.py takes 2 arguments:

> --data-path : 
>	default it will use the `raw_output.csv` in latest run folder `CSV/Run*`

>--output-dir : 
>	default it will make a `graphs` dir in the latest run folder `CSV/Run*`

example of using custom output directory for graphs:
```python
python plot.py --output-dir /home/gpuio/gpuIO/custom_graphs
```

### 6. Push to github

> If u make any large file, please make sure to add them to .gitignore to avoid committing them to the repository.

> *Push after every succesful run.*

### 7. Handling Anomalies in Benchmark Results

Sometimes individual benchmark configurations may fail or produce anomalous results (blank entries or 0 values) in the raw_output.csv file. writing a script to handle that, should be ready soon.