import re
import glob
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

LOG_DIR = "benchmarks/logs"
PLOT_DIR = "benchmarks/plots"

patterns = {
    "runtime": re.compile(r"Runtime\s+([\d.]+)\s*s"),
    "max_mem": re.compile(r"Max mem\s+([\d.]+)\s*MB"),
    "mean_mem": re.compile(r"Mean mem\s+([\d.]+)\s*MB"),
    "l1": re.compile(r"l1:\s*([\d.E+-]+)"),
    "l2": re.compile(r"l2:\s*([\d.E+-]+)"),
}

results = defaultdict(lambda: defaultdict(list))

for log_path in glob.glob(f"{LOG_DIR}/log_*.txt"):
    fname = log_path.split("/")[-1]
    batch_size = int(fname.split("_")[1].split(".")[0])

    with open(log_path, "r") as f:
        text = f.read()

    for key, pattern in patterns.items():
        match = pattern.search(text)
        if match:
            results[batch_size][key].append(float(match.group(1)))

batch_sizes = sorted(results.keys())

avg_runtime = np.array([
    np.mean(results[b]["runtime"]) for b in batch_sizes
])

avg_max_mem = np.array([
    np.mean(results[b]["max_mem"]) for b in batch_sizes
])

avg_mean_mem = np.array([
    np.mean(results[b]["mean_mem"]) for b in batch_sizes
])

avg_l1 = np.array([
    np.mean(results[b]["l1"]) for b in batch_sizes
])

avg_l2 = np.array([
    np.mean(results[b]["l2"]) for b in batch_sizes
])

for i, b in enumerate(batch_sizes):
    print(
        f"Batch {b:>3} | "
        f"runtime={avg_runtime[i]:.2f}s | "
        f"max_mem={avg_max_mem[i]:.2f}MB | "
        f"mean_mem={avg_mean_mem[i]:.2f}MB | "
        f"l1={avg_l1[i]:.3e} | "
        f"l2={avg_l2[i]:.3e}"
    )

#################################################################################
# plotting
batch_sizes = np.array(batch_sizes)

###################################
# acc per second
acc_per_second = 1.0 / (avg_runtime * avg_l2)
plt.figure()
plt.plot(batch_sizes, acc_per_second, marker="o")
plt.xlabel("Random batch size")
plt.ylabel("1 / (runtime × l2)")
plt.title("1 / (Runtime × L2) vs Batch Size")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/acc_per_second.png")
# plt.show()

###################################
# memory
plt.figure()
plt.plot(batch_sizes, avg_max_mem, marker="o", label="Max memory (MB)")
plt.plot(batch_sizes, avg_mean_mem, marker="s", label="Mean memory (MB)")
plt.xlabel("Random batch size")
plt.ylabel("Memory (MB)")
plt.title("Memory Usage vs Batch Size")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.ylim(bottom=0, top=max(max(avg_max_mem), max(avg_mean_mem))*1.2)
plt.savefig(f"{PLOT_DIR}/memory_usage.png")
# plt.show()

###################################
# runtime, l1, l2
_, ax_acc_per_s = plt.subplots()

line_acc_per_s, = ax_acc_per_s.plot(
    batch_sizes,
    avg_runtime,
    marker="o",
    label="Runtime (s)"
)
ax_acc_per_s.set_xlabel("Random batch size")
ax_acc_per_s.set_ylabel("Runtime (s)")
ax_acc_per_s.grid(True)

ax_error = ax_acc_per_s.twinx()
line_l1, = ax_error.plot(
    batch_sizes,
    avg_l1,
    marker="s",
    linestyle="--",
    label="L1"
)
line_l2, = ax_error.plot(
    batch_sizes,
    avg_l2,
    marker="^",
    linestyle="--",
    label="L2"
)
ax_error.set_ylabel("Error (L1, L2)")

lines = [line_acc_per_s, line_l1, line_l2]
labels = [line.get_label() for line in lines]
ax_acc_per_s.legend(lines, labels, loc="best")

plt.title("Runtime and Errors vs Batch Size")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/runtime_l1_l2.png")
# plt.show()

###################################
# acc per second with l2 error

_, ax_acc_per_s = plt.subplots()

acc_per_second = 1.0 / (avg_runtime * avg_l2)
line_acc_per_s, = ax_acc_per_s.plot(
    batch_sizes,
    acc_per_second,
    marker="o",
    label="1 / (runtime × l2)"
)
ax_acc_per_s.set_xlabel("Random batch size")
ax_acc_per_s.set_ylabel("1 / (runtime × l2)")
ax_acc_per_s.grid(True)

ax_error = ax_acc_per_s.twinx()

line_l2, = ax_error.plot(
    batch_sizes,
    avg_l2,
    marker="^",
    linestyle="--",
    label="L2"
)

ax_error.set_ylabel("L2-Error")

lines = [line_acc_per_s, line_l2]
labels = [line.get_label() for line in lines]
ax_acc_per_s.legend(lines, labels, loc="best")

plt.title("1 / (runtime × l2) and L2-Error vs. Batch size")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/acc_per_second_l2.png")
# plt.show()
