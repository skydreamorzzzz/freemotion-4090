"""Python 3.8 compatible, read-only environment/source audit. Never installs packages."""
import argparse
import ast
import difflib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

OFFICIAL = 'https://github.com/VankouF/FreeMotion-Codes.git'


def command(argv, cwd=None):
    env = os.environ.copy()
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    try:
        p = subprocess.run(argv, cwd=str(cwd) if cwd else None, env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           universal_newlines=True, timeout=180)
        return {'command': argv, 'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {'command': argv, 'returncode': None, 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True, help='Remote FreeMotion checkout')
    parser.add_argument('--output', type=Path, default=Path('artifacts/remote_environment_audit'))
    args = parser.parse_args()
    repo, out = args.repo.resolve(), args.output.resolve()
    if not (repo / 'requirements.txt').is_file():
        parser.error('--repo must contain FreeMotion requirements.txt')
    out.mkdir(parents=True, exist_ok=True)
    report = {'python_executable': sys.executable, 'python_version': sys.version,
              'conda_default_env': os.environ.get('CONDA_DEFAULT_ENV'),
              'conda_prefix': os.environ.get('CONDA_PREFIX'), 'repo': str(repo),
              'policy': 'Audit only: no install, upgrade, reset, source import or training',
              'commands': {}, 'official_source_verified': False}
    commands = {
        'python_version': [sys.executable, '--version'],
        'pip_version': [sys.executable, '-m', 'pip', '--version'],
        'pip_list': [sys.executable, '-m', 'pip', 'list', '--format=json'],
        'pip_check': [sys.executable, '-m', 'pip', 'check'],
        'conda_list': [os.environ.get('CONDA_EXE', 'conda'), 'list', '--prefix', sys.prefix, '--json'],
        'gpu': ['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
        'local_head': ['git', 'rev-parse', 'HEAD'],
        'local_status': ['git', 'status', '--short'],
        'local_diff': ['git', 'diff', 'HEAD', '--'],
        'local_diff_stat': ['git', 'diff', 'HEAD', '--stat'],
    }
    for name, argv in commands.items():
        result = command(argv, repo)
        report['commands'][name] = result
        (out / (name + '.json')).write_text(json.dumps(result, indent=2), encoding='utf-8')
    # Import only the installed torch package in a subprocess; no GPU training or matmul.
    probe = ('import json,torch; print(json.dumps({"torch":torch.__version__, '
             '"cuda_runtime":torch.version.cuda,"cudnn":torch.backends.cudnn.version(),'
             '"cuda_available":torch.cuda.is_available()}))')
    report['torch_probe'] = command([sys.executable, '-B', '-c', probe])
    packages = []
    for dist in metadata.distributions():
        name = dist.metadata.get('Name', '')
        packages.append({'name': name, 'version': dist.version,
                         'requires_python': dist.metadata.get('Requires-Python'),
                         'requires_dist': dist.requires or []})
    report['installed_metadata'] = packages
    # pip carries packaging itself; no dependency is installed for the audit.
    try:
        from packaging.requirements import Requirement
    except ImportError:
        from pip._vendor.packaging.requirements import Requirement
    with tempfile.TemporaryDirectory(prefix='freemotion_official_audit_') as directory:
        official = Path(directory) / 'official'
        clone = command(['git', 'clone', '--depth', '1', OFFICIAL, str(official)])
        report['official_clone'] = clone
        if clone.get('returncode') == 0:
            report['official_source_verified'] = True
            report['official_commit'] = command(['git', 'rev-parse', 'HEAD'], official)['stdout'].strip()
            req = (official / 'requirements.txt').read_text(encoding='utf-8')
            report['official_requirements'] = req
            (out / 'official_requirements.txt').write_text(req, encoding='utf-8')
            tracked = command(['git', 'ls-files'], official)['stdout'].splitlines()
            source_diffs, dependency_uses, snapshots = [], [], {}
            for name in tracked:
                path = official / name
                if path.suffix not in ('.py', '.yaml', '.yml', '.txt', '.md', '.sh'):
                    continue
                text = path.read_text(encoding='utf-8', errors='replace')
                if name == 'README.md' or name.endswith('.sh') or name.startswith('configs/'):
                    snapshots[name] = text
                local_path = repo / name
                if not local_path.is_file():
                    source_diffs.append({'path': name, 'status': 'missing locally'})
                else:
                    local = local_path.read_text(encoding='utf-8', errors='replace')
                    if local != text:
                        diff = ''.join(difflib.unified_diff(text.splitlines(True), local.splitlines(True),
                                                          fromfile='official/' + name, tofile='local/' + name))
                        source_diffs.append({'path': name, 'status': 'different', 'diff': diff})
                if path.suffix == '.py':
                    for line_no, line in enumerate(text.splitlines(), 1):
                        if re.search(r'\b(torch|torchvision|torchaudio|lightning|mmcv)\b', line):
                            dependency_uses.append({'file': name, 'line': line_no, 'text': line.strip()})
            report['official_readme_scripts_configs'] = snapshots
            report['official_dependency_uses'] = dependency_uses
            report['source_differences'] = source_diffs
            differences = []
            for line in req.splitlines():
                line = line.strip()
                if not line or line.startswith(('#', '--')):
                    continue
                if line.startswith('git+'):
                    try:
                        dist = metadata.distribution('clip')
                        direct = dist.read_text('direct_url.json')
                        row = {'requirement': line, 'installed': dist.version,
                               'status': 'provenance_check_required', 'direct_url': json.loads(direct) if direct else None}
                    except metadata.PackageNotFoundError:
                        row = {'requirement': line, 'installed': None, 'status': 'missing'}
                else:
                    requirement = Requirement(line)
                    try:
                        version = metadata.version(requirement.name)
                        row = {'requirement': line, 'installed': version,
                               'status': 'matches' if requirement.specifier.contains(version, prereleases=True) else 'version_difference_keep_until_tested'}
                    except metadata.PackageNotFoundError:
                        row = {'requirement': line, 'installed': None, 'status': 'missing'}
                differences.append(row)
            report['dependency_comparison'] = differences
        else:
            report['blocked'] = 'Official clone failed; no local file assumed to be official. See official_clone.'
    report['protected_packages'] = {name: next((p for p in packages if p['name'].lower().replace('_', '-') == name), None)
                                    for name in ['torch', 'torchvision', 'torchaudio', 'lightning', 'pytorch-lightning', 'mmcv', 'mmcv-full']}
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# FreeMotion 环境审计（未安装任何依赖）', '',
             '- Python: `' + sys.version.splitlines()[0] + '`', '- Interpreter: `' + sys.executable + '`',
             '- 官方源已核验: ' + str(report['official_source_verified']), '',
             '| 官方要求 | 当前实际版本 | 状态 |', '|---|---|---|']
    for row in report.get('dependency_comparison', []):
        lines.append('| {} | {} | {} |'.format(row['requirement'], row['installed'], row['status']))
    lines += ['', 'missing 仅代表缺包，不是自动安装授权；版本差异不能直接视为不兼容。',
              'MMCV 与 mmcv-full 的模块冲突、CLIP 来源、pip check 报错及代码 diff 需人工核查。',
              '此脚本未 import 全部核心包或运行模型；安装决策和 smoke test 留待审计后。']
    if report.get('blocked'): lines.append(report['blocked'])
    (out / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Audit saved to {}'.format(out))
    print('Official source verified: {}'.format(report['official_source_verified']))
    print('No packages installed; no model/source changes; no training.')


if __name__ == '__main__':
    main()
