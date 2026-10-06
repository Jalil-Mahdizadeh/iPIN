"""Run the one frozen benchmark after full training; restart committed phases."""
import json
import subprocess
import sys
from common import ROOT, require_continue, stop_requested


def main():
    import prepare_selected
    prepare_selected.main()
    if (ROOT / 'completed.json').exists():
        from write_report import finish_closeout
        finish_closeout(json.loads((ROOT / 'completed.json').read_text()))
        print('V3 already finalized; verified final artifacts.', flush=True)
        return
    require_continue()
    import qualify
    qualify.main()
    require_continue()
    subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone',
        '--nnodes=1', '--nproc-per-node=4', str(ROOT / 'scripts/infer.py')], check=True)
    require_continue()
    import analyze
    analyze.main()
    require_continue()
    import write_report
    write_report.main()


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError:
        if stop_requested(): raise SystemExit(75)
        raise
