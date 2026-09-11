from pathlib import Path
path=Path('build_baseline_report.py')
source=path.read_text(encoding='utf-8-sig')
source=source.replace("frame[column]=frame[column].map(lambda number:f'{number:.4f}' if pd.notna(number) else 'N/A')", "if column in ['n','horizon','tests','failed','skipped']:\n            frame[column]=frame[column].map(lambda number:f'{int(number):,}' if pd.notna(number) else 'N/A')\n        elif column=='nmae':\n            frame[column]=frame[column].map(lambda number:f'{number:.2%}' if pd.notna(number) else 'N/A')\n        else:\n            frame[column]=frame[column].map(lambda number:f'{number:.4f}' if pd.notna(number) else 'N/A')")
source=source.replace("def interval(horizon):", "def change_word(fraction):\n    return f'降低{fraction:.2%}' if fraction>=0 else f'升高{-fraction:.2%}'\n\ndef interval(horizon):")
source=source.replace('MAE降低{reduction:.2%}', 'MAE{change_word(reduction)}')
source=source.replace('MAE变化为降低{trend:.2%}', 'MAE{change_word(trend)}')
path.write_text(source,encoding='utf-8')
