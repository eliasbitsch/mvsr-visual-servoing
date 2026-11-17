#!/usr/bin/env python3
"""Runs a whole evaluation campaign unattended (WSL side, controls the containers).

    python3 scripts/run_campaign.py experiments/final.yaml            # everything
    python3 scripts/run_campaign.py experiments/final.yaml --smoke    # 1 run per block
    python3 scripts/run_campaign.py experiments/final.yaml --blocks det_sam3 lost

Results: data/eval/<campaign>/<block>/ (run_eval output), logs next to them. Finished blocks
(DONE file) are skipped, interrupted ones resume, so the same command continues a campaign.
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
COMPOSE = 'docker-compose.yml:docker-compose.fm.yml:docker-compose.wsl.yml'
ROS_ENV = ('source /opt/ros/jazzy/setup.bash && source /ws/install/setup.bash && '
           'source /ws/student/install/setup.bash')
# The Ollama on the Windows host (Vulkan): llama.cpp's CUDA kernels crash sporadically on this
# laptop's RTX 3080 Ti (illegal memory access), so not the compose service
OLLAMA_URL = 'http://localhost:11434'
CLI_VLM_URL = 'http://localhost:11436'
MIN_FREE_GB = 3.0
RESUME_FREE_GB = 8.0
MAX_MEMORY_WAIT = 2 * 3600
LOG = None


def log(msg):
    line = time.strftime('%H:%M:%S ') + msg
    print(line, flush=True)
    if LOG:
        with open(LOG, 'a') as fh:
            fh.write(line + '\n')


def sh(cmd, check=False, capture=True, timeout=None):
    env = {**os.environ, 'COMPOSE_FILE': COMPOSE}
    out = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=REPO, env=env, text=True,
                         capture_output=capture, timeout=timeout)
    if check and out.returncode != 0:
        raise RuntimeError(f'{cmd}: {out.stderr.strip()[-500:]}')
    return out


def http_ok(url, timeout=3):
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True
    except Exception:
        return False


def wait_for(cond, timeout, what, step=2.0):
    end = time.time() + timeout
    while time.time() < end:
        res = cond()
        if res:
            return res
        time.sleep(step)
    raise RuntimeError(f'timeout waiting for {what}')


def free_gb():
    """Smaller of WSL and Windows available memory (the host hit 100 % RAM once)."""
    with open('/proc/meminfo') as fh:
        wsl = next(int(l.split()[1]) for l in fh if l.startswith('MemAvailable')) / 2**20
    try:
        out = sh(['powershell.exe', '-NoProfile', '-Command',
                  '(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory'], timeout=30)
        win = int(out.stdout.strip()) / 2**20
    except Exception:
        win = wsl
    return min(wsl, win)


# --- services ---------------------------------------------------------------------------------

def container_running(name):
    out = sh(['docker', 'inspect', '-f', '{{.State.Running}}', name])
    return out.stdout.strip() == 'true'


def start_sim():
    log('starting sim (vs-lab)')
    sh(['docker', 'rm', '-f', 'vs-lab'])
    sh(['docker', 'compose', 'run', '-d', '--name', 'vs-lab', 'sim', 'bash', '-lc',
        'ros2 launch vs_sim vs_sim.launch.py target:=red_can gazebo_gui:=false '
        'image_view:=false camera_rate:=15 > /tmp/launch.log 2>&1'], check=True)
    wait_for(lambda: sh(['docker', 'exec', 'vs-lab', 'grep', '-q',
                         'Configured and activated joint_state_broadcaster',
                         '/tmp/launch.log']).returncode == 0, 180, 'sim controllers')
    time.sleep(5)


class Fm:
    """The perception + IBVS container, recreated whenever its launch config changes."""

    def __init__(self):
        self.key = None

    def start(self, launch):
        key = json.dumps(launch, sort_keys=True)
        if key == self.key and container_running('vs-fm') and self.perception_alive():
            return
        log(f'starting stack {launch}')
        sh(['docker', 'stop', '-t', '15', 'vs-fm'])
        sh(['docker', 'rm', '-f', 'vs-fm'])
        args = ' '.join(f'{k}:={str(v).lower() if isinstance(v, bool) else v}'
                        for k, v in launch.items() if not k.startswith('vlm_'))
        env = []
        if launch.get('vlm_model'):
            url = CLI_VLM_URL if 'claude' in launch['vlm_model'] else OLLAMA_URL
            env = ['-e', f'VS_VLM_URL={url}/v1', '-e', f'VS_VLM_MODEL={launch["vlm_model"]}']
        cmd = ('cd /ws/student && colcon build --symlink-install && source install/setup.bash '
               f'&& exec ros2 launch vs_bringup servo.launch.py {args}')
        sh(['docker', 'compose', 'run', '-d', '--name', 'vs-fm', *env, 'fm', 'bash', '-lc', cmd],
           check=True)

        def ready():
            res = sh(['docker', 'logs', 'vs-fm'])
            logs = res.stdout + res.stderr
            if 'models ready' in logs:
                return True
            if 'process has died' in logs or 'Failed   <<<' in logs:
                raise RuntimeError('stack failed:\n' + logs[-3000:])
            return False
        # First start of a model downloads its weights
        wait_for(ready, 1200, 'models ready', step=5)
        self.key = key

    def perception_alive(self):
        return sh(['docker', 'exec', 'vs-fm', 'pgrep', '-f', 'perception_node']).returncode == 0

    def stop(self):
        sh(['docker', 'stop', '-t', '15', 'vs-fm'])
        self.key = None


def unload_host_ollama():
    """An Ollama on the Windows host (port 11434) may still hold a model in VRAM."""
    try:
        with urllib.request.urlopen('http://localhost:11434/api/ps', timeout=3) as r:
            models = [m['name'] for m in json.load(r).get('models', [])]
    except Exception:
        return
    for name in models:
        body = json.dumps({'model': name, 'keep_alive': 0}).encode()
        urllib.request.urlopen(urllib.request.Request(
            'http://localhost:11434/api/generate', body,
            {'Content-Type': 'application/json'}), timeout=30).read()
        log(f'unloaded {name} from the host Ollama')


def ensure_vlm_server(model):
    """Start only what the block needs; everything else stays off (VRAM, quota)."""
    if model and 'claude' in model:
        if not http_ok(f'{CLI_VLM_URL}/v1/models'):
            log('starting the CLI VLM shim on the host')
            win_path = sh(['wslpath', '-w', str(REPO / 'scripts' / 'cli_vlm_server.py')]
                          ).stdout.strip()
            subprocess.Popen(['python.exe', win_path, '--idle-exit', '3600'],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True)
            wait_for(lambda: http_ok(f'{CLI_VLM_URL}/v1/models'), 60, 'CLI VLM shim')
    if model and 'claude' not in model:
        # Briefly gone while the driver resets the GPU after a fault
        wait_for(lambda: http_ok(f'{OLLAMA_URL}/api/tags'), 180, f'Ollama at {OLLAMA_URL}')
        if model != 'ollama':
            log(f'loading {model} into ollama')
            body = json.dumps({'model': model, 'prompt': '', 'keep_alive': '30m'}).encode()
            urllib.request.urlopen(urllib.request.Request(
                f'{OLLAMA_URL}/api/generate', body, {'Content-Type': 'application/json'}),
                timeout=600).read()
    else:
        unload_host_ollama()  # its VRAM belongs to the detector of this block


# --- blocks -----------------------------------------------------------------------------------

def merge(defaults, block):
    return {k: {**defaults.get(k, {}), **(block.get(k) or {})} for k in ('launch', 'eval')}


def eval_args(opts):
    args = []
    for k, v in opts.items():
        args.append('--' + k.replace('_', '-'))
        args += [str(x) for x in v] if isinstance(v, list) else [str(v)]
    return args


def wait_for_memory(fm):
    """Something else (another job on the host) took the RAM: free ours, wait, then resume."""
    log(f'less than {MIN_FREE_GB} GB RAM free: stopping the containers and waiting')
    fm.stop()
    sh(['docker', 'stop', '-t', '15', 'vs-lab'])
    end = time.time() + MAX_MEMORY_WAIT
    while time.time() < end:
        time.sleep(60)
        if free_gb() >= RESUME_FREE_GB:
            log(f'{free_gb():.1f} GB free again, resuming')
            return
    raise MemoryError(f'RAM not back above {RESUME_FREE_GB} GB within {MAX_MEMORY_WAIT // 60} min')


FAULTS = ('detector failed', 'CUDA error', 'illegal memory access', 'illegal instruction')
MAX_FAULTS = 3


def summary_rows(path):
    return sum(1 for _ in open(path)) if path.exists() else 0


def truncate_summary(path, rows):
    """Drop the runs written since the last check without a fault (header counts as a row)."""
    if path.exists():
        lines = open(path).readlines()[:rows]
        path.write_text(''.join(lines))


def stack_fault():
    res = sh(['docker', 'logs', 'vs-fm'])
    log_text = res.stdout + res.stderr
    return next((l.strip()[-300:] for l in log_text.splitlines()
                 if any(f in l for f in FAULTS)), None), log_text


def run_block(name, cfg, out_dir, fm, attempts=6):
    host_dir = REPO / 'data' / 'eval' / out_dir / name
    if (host_dir / 'DONE').exists():
        log(f'{name}: done, skipped')
        return True
    host_dir.mkdir(parents=True, exist_ok=True)
    ensure_vlm_server(cfg['launch'].get('vlm_model'))
    container_dir = f'/root/vs_data/eval/{out_dir}/{name}'
    cmd = (f'{ROS_ENV} && ros2 run vs_evaluation run_eval --resume {container_dir} '
           + ' '.join(shlex.quote(a) for a in eval_args(cfg['eval'])))
    summary = host_dir / 'summary.csv'
    attempt = faults = 0
    while attempt < attempts:
        attempt += 1
        if free_gb() < MIN_FREE_GB:
            wait_for_memory(fm)
        if not container_running('vs-lab'):
            start_sim()
        fm.start(cfg['launch'])
        log(f'{name}: attempt {attempt}')
        with open(host_dir.parent / f'{name}.log', 'a') as fh:
            proc = subprocess.Popen(['docker', 'exec', 'vs-fm', 'bash', '-c', cmd],
                                    stdout=fh, stderr=subprocess.STDOUT)
            last_check, low_mem, fault = time.time(), False, None
            good_rows = summary_rows(summary)
            while proc.poll() is None:
                time.sleep(10)
                if time.time() - last_check < 30:
                    continue
                last_check = time.time()
                if free_gb() < MIN_FREE_GB:
                    low_mem = True
                # A GPU fault (Xid 13 on this laptop every few hours) or a detector raising
                # turns every later run into "not detected"; catch it while the block runs
                fault, _ = stack_fault()
                if low_mem or fault:
                    proc.kill()
                    proc.wait()
                    break
                good_rows = summary_rows(summary)
        end_fault, stack_log = stack_fault()
        fault = fault or end_fault
        (host_dir.parent / f'{name}.stack.log').write_text(stack_log)
        if low_mem:
            truncate_summary(summary, good_rows)
            wait_for_memory(fm)
            attempt -= 1  # not the sim's fault
            continue
        if fault:
            faults += 1
            truncate_summary(summary, good_rows)
            log(f'{name}: stack fault ({faults}/{MAX_FAULTS}), runs since the last check '
                f'discarded: {fault}')
            if faults >= MAX_FAULTS:
                raise RuntimeError(f'detector keeps failing: {fault}')
            fm.stop()  # fresh process, fresh CUDA context
            attempt -= 1
            continue
        if proc.returncode == 0:
            (host_dir / 'DONE').write_text(time.strftime('%Y-%m-%d %H:%M:%S\n'))
            log(f'{name}: finished')
            return True
        # Usual cause under WSL: the lab's ros_gz_bridge lost all Gazebo topics; sometimes the
        # perception node used up its respawns after CUDA faults
        log(f'{name}: run_eval exit {proc.returncode}, restarting the sim')
        start_sim()
        if not container_running('vs-fm') or not fm.perception_alive():
            fm.key = None
    log(f'{name}: giving up after {attempts} attempts')
    return False


def main():
    global LOG
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--name', help='campaign folder under data/eval (default: config name)')
    ap.add_argument('--blocks', nargs='+', help='only these blocks')
    ap.add_argument('--smoke', action='store_true', help='one short run per block')
    ap.add_argument('--keep-running', action='store_true', help='leave the containers up')
    args = ap.parse_args()

    spec = yaml.safe_load((REPO / args.config).read_text()
                          if not os.path.isabs(args.config) else open(args.config).read())
    name = args.name or Path(args.config).stem
    if args.smoke:
        name += '_smoke'
    out = REPO / 'data' / 'eval' / name
    out.mkdir(parents=True, exist_ok=True)
    LOG = out / 'campaign.log'
    (out / 'campaign.yaml').write_text(yaml.safe_dump(spec, sort_keys=False))

    blocks = {k: merge(spec.get('defaults', {}), v or {}) for k, v in spec['blocks'].items()}
    if args.blocks:
        blocks = {k: v for k, v in blocks.items() if k in args.blocks}
    if args.smoke:
        for cfg in blocks.values():
            cfg['eval'].update(targets=['red_can'], starts=[1], timeout=40)
    order = list(blocks)

    log(f'campaign {name}: {len(order)} blocks, free RAM {free_gb():.1f} GB')
    unload_host_ollama()
    # Latencies are only comparable on an otherwise idle GPU: record what else runs on it
    gpu = sh('nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,clocks.sm --format=csv,noheader'
             ' && nvidia-smi --query-compute-apps=process_name --format=csv,noheader')
    (out / 'gpu_preflight.txt').write_text(gpu.stdout)
    load = int(gpu.stdout.split(',')[0].strip().rstrip('%') or 0) if gpu.stdout else 0
    if load > 10:
        log(f'WARNING: GPU already {load} % busy before the campaign, latencies will be inflated: '
            + ', '.join(l.rsplit('\\', 1)[-1] for l in gpu.stdout.splitlines()[1:]))
    fm, failed = Fm(), []
    try:
        for block in order:
            try:
                if not run_block(block, blocks[block], name, fm):
                    failed.append(block)
            except MemoryError as exc:
                log(f'ABORT: {exc}')
                failed.append(f'{block} (aborted, memory)')
                raise
            except Exception as exc:  # one broken model must not cost the rest of the night
                log(f'{block}: FAILED: {exc}')
                (out / f'{block}.error').write_text(str(exc))
                failed.append(block)
                fm.key = None
        if spec.get('bench') and not args.blocks:
            log('offline detector benchmark')
            fm.stop()
            ensure_vlm_server('claude')
            ensure_vlm_server('ollama')
            bench = sh(['docker', 'compose', 'run', '--rm', '-e', f'VS_VLM_URL={OLLAMA_URL}/v1',
                        'fm', 'bash', '-lc',
                        f'cd /ws/student && colcon build --symlink-install >/dev/null && '
                        f'{ROS_ENV} && $VS_PYTHON -m vs_evaluation.bench_detectors '
                        f'--campaign /root/vs_data/eval/{name}' + (' --smoke' if args.smoke else '')],
                       capture=False)
            if bench.returncode != 0:
                failed.append('bench')
        if spec.get('notebook') and not args.blocks:
            log('executing the notebook')
            nb = sh(['docker', 'compose', 'run', '--rm', '-v', f'{REPO}:/repo', '-w', '/repo/notebooks',
                     '-e', f'VS_CAMPAIGN={name}', 'fm', 'bash', '-lc',
                     f'unset PYTHONPATH; $VS_PYTHON -m jupyter nbconvert --to notebook --execute --inplace '
                     f'/repo/{spec["notebook"]}'], capture=False)
            if nb.returncode != 0:
                failed.append('notebook')
    finally:
        ensure_vlm_server(None)
        if not args.keep_running:
            fm.stop()
            sh(['docker', 'stop', '-t', '15', 'vs-lab'])
        log(f'campaign {name} ended, failed: {failed or "none"}')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
