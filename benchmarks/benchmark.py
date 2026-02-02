import subprocess
import time
import psutil


def benchmark_with_mem():
    start = time.time()

    for random_batch_size in [6]:  # , 8, 12, 16, 24, 32, 64]:

        with open(f"benchmarks/logs/log_{random_batch_size}.txt", "w") as f:
            process = subprocess.Popen([
                "./scripts/insep.sh",
                "--config.eqn_cfg.rand_batch_size", "16",
                "--config.eqn_cfg.hess_diag_method", "sparse_stde",
                "--config.eqn_cfg.dim", "100",
                "--config.eqn_cfg.name", "AllenCahnTwobody",
                "--n_runs=1",
                "--config.gd_cfg.epochs", "1000",
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

        print(f"Runtime {duration:.2f} s")
        print(f"Max mem {max_mem / 2**20:.3f} MB")
        print(f"Mean mem {mean_mem / 2**20:.3f} MB")

benchmark_with_mem()