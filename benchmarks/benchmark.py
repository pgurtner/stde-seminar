import subprocess
import time
import psutil
import os


def memory_uss_tree(proc):
    total = 0
    try:
        procs = [proc] + proc.children(recursive=True)
        for p in procs:
            try:
                total += p.memory_full_info().uss
            except psutil.NoSuchProcess:
                pass
    except psutil.NoSuchProcess:
        pass
    return total


def benchmark_with_cheap_rss_mem():
    LOG_DIR = "logs-rss-mem"
    os.makedirs(f"benchmarks/{LOG_DIR}", exist_ok=True)
    for random_batch_size in [6, 8, 12, 16, 24, 32]:
        # rather large number to avg out fluctuations from laptop benchmarking
        for i in range(10):
            start = time.time()
            print(f"Running {i}th iteration of random batch size {random_batch_size}")
            with open(f"benchmarks/{LOG_DIR}/log_{random_batch_size}.{i}.txt", "w") as f:
                process = subprocess.Popen([
                    "./scripts/insep.sh",
                    "--config.eqn_cfg.rand_batch_size", f"{random_batch_size}",
                    "--config.eqn_cfg.hess_diag_method", "sparse_stde",
                    "--config.eqn_cfg.dim", "100",
                    "--config.eqn_cfg.name", "AllenCahnTwobody",
                    "--n_runs=1",
                    "--config.gd_cfg.epochs", "10000",
                ], stdout=f, stderr=f)

                ps_process = psutil.Process(process.pid)
                mem_samples = []

                while process.poll() is None:
                    try:
                        mem_info = ps_process.memory_info()
                        mem_samples.append(mem_info.rss)
                        time.sleep(0.1)
                    except psutil.NoSuchProcess:
                        break

            process.wait()

            duration = time.time() - start
            max_mem = max(mem_samples)
            mean_mem = sum(mem_samples) / len(mem_samples) if len(mem_samples) > 0 else 0

            with open(f"benchmarks/{LOG_DIR}/log_{random_batch_size}.{i}.txt", "a") as f:
                f.write(f"Runtime {duration:.2f} s\n")
                f.write(f"Max mem {max_mem / 2 ** 20:.3f} MB\n")
                f.write(f"Mean mem {mean_mem / 2 ** 20:.3f} MB\n")


def benchmark_with_mem():
    LOG_DIR = "logs-uss-mem"
    os.makedirs(f"benchmarks/{LOG_DIR}", exist_ok=True)
    for random_batch_size in [6, 8, 12, 16, 24, 32]:
        # rather large number to avg out fluctuations from laptop benchmarking
        for i in range(10):
            start = time.time()
            print(f"Running {i}th iteration of random batch size {random_batch_size}")
            with open(f"benchmarks/{LOG_DIR}/log_{random_batch_size}.{i}.txt", "w") as f:
                process = subprocess.Popen([
                    "./scripts/insep.sh",
                    "--config.eqn_cfg.rand_batch_size", f"{random_batch_size}",
                    "--config.eqn_cfg.hess_diag_method", "sparse_stde",
                    "--config.eqn_cfg.dim", "100",
                    "--config.eqn_cfg.name", "AllenCahnTwobody",
                    "--n_runs=1",
                    "--config.gd_cfg.epochs", "10000",
                ], stdout=f, stderr=f)

                ps_process = psutil.Process(process.pid)
                mem_samples = []

                while process.poll() is None:
                    try:
                        mem_samples.append(memory_uss_tree(ps_process))
                        time.sleep(0.5)
                    except psutil.NoSuchProcess:
                        break

            process.wait()

            duration = time.time() - start
            max_mem = max(mem_samples)
            mean_mem = sum(mem_samples) / len(mem_samples) if len(mem_samples) > 0 else 0

            with open(f"benchmarks/{LOG_DIR}/log_{random_batch_size}.{i}.txt", "a") as f:
                f.write(f"Runtime {duration:.2f} s\n")
                f.write(f"Max mem {max_mem / 2 ** 20:.3f} MB\n")
                f.write(f"Mean mem {mean_mem / 2 ** 20:.3f} MB\n")


def benchmark_no_mem():
    LOG_DIR = "logs-normal"
    os.makedirs(f"benchmarks/{LOG_DIR}", exist_ok=True)
    for random_batch_size in [6, 8, 12, 16, 24, 32]:
        # rather large number to avg out fluctuations from laptop benchmarking
        for i in range(10):
            start = time.time()
            print(f"Running {i}th iteration of random batch size {random_batch_size}")
            with open(f"benchmarks/{LOG_DIR}/log_{random_batch_size}.{i}.txt", "w") as f:
                process = subprocess.Popen([
                    "./scripts/insep.sh",
                    "--config.eqn_cfg.rand_batch_size", f"{random_batch_size}",
                    "--config.eqn_cfg.hess_diag_method", "sparse_stde",
                    "--config.eqn_cfg.dim", "100",
                    "--config.eqn_cfg.name", "AllenCahnTwobody",
                    "--n_runs=1",
                    "--config.gd_cfg.epochs", "10000",
                ], stdout=f, stderr=f)

            process.wait()

            duration = time.time() - start
            max_mem = 0
            mean_mem = 0

            with open(f"benchmarks/{LOG_DIR}/log_{random_batch_size}.{i}.txt", "a") as f:
                f.write(f"Runtime {duration:.2f} s\n")
                f.write(f"Max mem {max_mem / 2 ** 20:.3f} MB\n")
                f.write(f"Mean mem {mean_mem / 2 ** 20:.3f} MB\n")


print("benchmarks without memory tracking")
benchmark_no_mem()

print("benchmarks with rss tracking")
benchmark_with_cheap_rss_mem()

print("benchmarks with uss tracking")
benchmark_with_mem()
