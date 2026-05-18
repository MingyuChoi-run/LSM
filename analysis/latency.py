import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
from torch.backends import cudnn

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

import threading
import time 
from torch.profiler import profile, record_function, ProfilerActivity

from model_zoo.lsm import buildLSMS, buildLSM
from model_zoo.lsm_light import buildLSM_light
from model_zoo.swinIR import buildSwinIR, buildSwinIR_light
from model_zoo.mambaIR import buildMambaIR, buildMambaIR_light
from model_zoo.mambairv2 import buildMambaIRv2S, buildMambaIRv2B
from model_zoo.mambairv2light import buildMambaIRv2Light
from model_zoo.atd import buildATD, buildATD_light
from model_zoo.cat import buildCATR, buildCATA
from model_zoo.dat import buildDATS, buildDAT



class PowerMeasurer:
    def __init__(self, tick_interval=1):
        self.tick_interval = tick_interval
        self.power_usage = []
        self.stop_event = threading.Event()
        self.power_thread = None
    
    def start(self):
        def get_power_usage():
            time.sleep(1)
            while not self.stop_event.is_set():
                self.power_usage.append(torch.cuda.power_draw(0))
                time.sleep(self.tick_interval)
        self.stop_event.clear()
        self.power_thread = threading.Thread(target=get_power_usage)
        self.power_thread.start()
    
    def stop(self):
        self.stop_event.set()
        if self.power_thread is not None:
            self.power_thread.join()
            self.power_thread = None
    
    def average(self):
        if self.power_usage:
            return np.mean(self.power_usage)/1000
        return None


# @calc_average_power_usage()
def test_direct_metrics(
    model,
    input_shape,
    n_repeat=100,
    use_float16=False,
    jit_compile=False,
    scale=None,
    report_throughput=True
):
    cudnn.benchmark = True

    print(f'CUDNN Benchmark: {cudnn.benchmark}')
    if use_float16:
        context = torch.cuda.amp.autocast
        print('Using AMP(FP16) for testing ...')
    else:
        context = nullcontext
        print('Using FP32 for testing ...')

    x = torch.FloatTensor(*input_shape).uniform_(0., 1.).cuda()
    print(f'Input shape: {x.shape}')
    model = model.cuda().eval()

    if jit_compile:
        with torch.no_grad():
            model = torch.jit.trace(model, x)

    # measure_power = PowerMeasurer()

    with context():
        with torch.inference_mode():
            print('warmup ...')
            for _ in tqdm.tqdm(range(100)):
                model(x)
                torch.cuda.synchronize()

            print('testing ...')
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

            # measure_power.start()

            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            timings = np.zeros((n_repeat, 1), dtype=np.float64)

            for rep in tqdm.tqdm(range(n_repeat)):
                starter.record()
                model(x)
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)  # ms
                timings[rep] = curr_time

            # measure_power.stop()

    avg_ms = float(np.sum(timings) / n_repeat)
    med_ms = float(np.median(timings))
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # [ADDED] Throughput calculations
    B, C, H_lr, W_lr = x.shape
    avg_s = avg_ms / 1000.0

    img_per_s = None
    lr_mpix_per_s = None
    hr_mpix_per_s = None

    if report_throughput and avg_s > 0:
        img_per_s = B / avg_s
        lr_mpix_per_s = (B * H_lr * W_lr) / avg_s / 1e6
        if scale is not None:
            H_hr, W_hr = H_lr * int(scale), W_lr * int(scale)
            hr_mpix_per_s = (B * H_hr * W_hr) / avg_s / 1e6

    print('------------ Results ------------')
    print(f'Average time: {avg_ms:.5f} ms')
    print(f'Median time: {med_ms:.5f} ms')
    print(f'Maximum GPU memory Occupancy: {torch.cuda.max_memory_allocated() / 1024**2:.5f} MB')
    print(f'Maximum GPU memory Reserved: {torch.cuda.max_memory_reserved() / 1024**2:.5f} MB')
    print(f'Params: {params / 1000:.3f}K')
    # print(f'Average power usage: {measure_power.average()} W')

    # [ADDED] Print throughput
    if report_throughput and img_per_s is not None:
        print('------------ Throughput ------------')
        print(f'Throughput: {img_per_s:.3f} images/s (B={B})')
        print(f'Throughput: {lr_mpix_per_s:.3f} MPix/s (LR={H_lr}x{W_lr})')
        if hr_mpix_per_s is not None:
            print(f'Throughput: {hr_mpix_per_s:.3f} MPix/s (HR={H_lr*int(scale)}x{W_lr*int(scale)}, scale={int(scale)})')
        else:
            print('Throughput(HR): scale not provided, skipping HR MPix/s')
        print('-----------------------------------')

    print('---------------------------------')


def profile_one_run(model, x, use_amp=False, out_dir="./profiler_out"):
    model.eval().cuda()
    x = x.cuda()

    # warmup
    with torch.inference_mode():
        for _ in range(10):
            _ = model(x)
        torch.cuda.synchronize()

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        with_modules=True,
    ) as prof:
        with torch.inference_mode():
            with record_function("forward_total"):
                _ = model(x)
            torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30))

    os.makedirs(out_dir, exist_ok=True)
    prof.export_chrome_trace(f"{out_dir}/trace.json")
    print(f"Saved trace to {out_dir}/trace.json")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H = 512
    W = 512
    scale = 4

    init_model = buildRGTviaLRU_semantic9_19tF_UNet(upscale=scale).to(device)

    # ── measure latency & memory & throughput ──
    test_direct_metrics(
        model=init_model,
        input_shape=(1, 3, H // scale, W // scale),
        n_repeat=10,
        use_float16=False,
        jit_compile=False,
        scale=scale,
        report_throughput=True
    )
