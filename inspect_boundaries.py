from pathlib import Path
from collections import Counter
from openpyxl import load_workbook
import fitz, json
root=Path.cwd()
for path in (root/'inputs').rglob('*.xlsx'):
    workbook=load_workbook(path,data_only=True)
    if path.name in ['附件1.xlsx','附件2.xlsx','附件4.xlsx']:
        for sheet in workbook:
            labels=[row[0] for row in list(sheet.values)[1:]] if path.name=='附件1.xlsx' else list(sheet.values)[0][1:]
            print(path.name,sheet.title,dict(Counter(type(label).__name__ for label in labels)))
            print('string samples',[(i,str(label)) for i,label in enumerate(labels) if isinstance(label,str)][:10])
    if path.name=='result2.xlsx':
        print('TAIL LABELS',list(workbook.worksheets[0].values)[0][-7:])
        for sheet in workbook.worksheets[1:]: print(sheet.title,[str(row[0]) for row in list(sheet.values)[1:] if row[0] is not None])
path=next((root/'inputs').rglob('*.pdf'))
(root/'artifacts/data_audit/pdf_pages').mkdir(exist_ok=True)
with fitz.open(path) as document:
    for index,page in enumerate(document):
        page.get_pixmap(matrix=fitz.Matrix(1.3,1.3)).save(root/f'artifacts/data_audit/pdf_pages/page_{index+1}.png')
