from pathlib import Path
import json, traceback
from audit_data import run as audit
from audit_extras import run as extras

if __name__ == '__main__':
    root=Path(__file__).resolve().parent
    config=json.loads((root/'configs/audit.json').read_text(encoding='utf-8-sig'))
    status_path=root/'artifacts/data_audit/run_status.json'
    status_path.parent.mkdir(parents=True,exist_ok=True)
    status_path.write_text(json.dumps({'stage':'audit_only','status':'running'}),encoding='utf-8')
    try:
        assert config['stage']=='audit_only'
        assert config['allow_training'] is False
        assert config['allow_template_writes'] is False
        audit()
        extras()
    except Exception:
        status_path.write_text(json.dumps({'stage':'audit_only','status':'failed','error':traceback.format_exc()},ensure_ascii=False,indent=2),encoding='utf-8')
        raise
    status_path.write_text(json.dumps({'stage':'audit_only','status':'complete','forecasting_leakage_tests':'NOT RUN','baselines':'NOT RUN'},indent=2),encoding='utf-8')
