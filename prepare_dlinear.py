from pathlib import Path
import hashlib,json,shutil
root=Path.cwd(); repo=root/'.tmp/tslib_repo'; base=root/'vendor/tslib'
commit='4e938a1767106324dd753b2a44832bf870a0252e'
manifest={'repository':'https://github.com/thuml/Time-Series-Library','commit':commit,'files':{}}
for relative in ['models/DLinear.py','layers/Autoformer_EncDec.py','LICENSE']:
    payload=(repo/relative).read_bytes(); (base/relative).write_bytes(payload)
    manifest['files'][relative]={'url':f'https://github.com/thuml/Time-Series-Library/blob/{commit}/{relative}','sha256':hashlib.sha256(payload).hexdigest()}
(base/'PROVENANCE.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
for relative in ['models/__init__.py','layers/__init__.py']: (base/relative).touch()
previous=json.loads((root/'artifacts/baselines/output_manifest.json').read_text(encoding='utf-8')); verified=0
for entry in previous:
    assert hashlib.sha256((root/entry['path']).read_bytes()).hexdigest()==entry['sha256'],entry['path']
    verified+=1
(root/'MIGRATION.md').write_text(f'# Project migration\n\nMoved from C:/Users/lenovo/Desktop/CUMCM2026C_TS_Baseline to D:/数学建模/CUMCM2026C_TS_Baseline.\n\n{verified} baseline outputs verified against previous SHA256 manifest. Project inputs, virtual environment and outputs now on D:. Original supplied ZIP stays on Desktop. Virtual environment refreshed at new path.\n',encoding='utf-8')
print('Pinned upstream',commit,'Migrated artifacts verified:',verified)
