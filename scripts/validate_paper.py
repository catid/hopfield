"""Structural/math extraction and prose-coverage checks for the reviewed conversion.
Run with uv run --with beautifulsoup4 --with lxml --with pypandoc_binary python scripts/validate_paper.py
The coverage check is a supplement to manual PDF/LaTeX review, not a proof of fidelity.
"""
from pathlib import Path
from collections import Counter
import json,re,unicodedata
import pypandoc
from bs4 import BeautifulSoup
ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'references'
md=(base/'2611.17309v1.md').read_text()
ast=json.loads(pypandoc.convert_text(md,'json','gfm+tex_math_dollars+footnotes'))
math=[]
def walk(x):
 if isinstance(x,dict):
  if x.get('t')=='Math': math.append({'display':x['c'][0]['t']=='DisplayMath','tex':x['c'][1]})
  for v in x.values(): walk(v)
 elif isinstance(x,list):
  for v in x: walk(v)
walk(ast)
(base/'review/math_ast.json').write_text(json.dumps(math,indent=2)+'\n')
anchors=re.findall(r'<a id="([^"]+)"></a>',md)
links=set(re.findall(r'\]\(#([^\)]+)\)',md))
figures=re.findall(r'!\[Figure (\d+)\]\(([^)]+)\)',md)
tables=re.findall(r'^Table (\d+):',md,re.M)
equations=re.findall(r'\\tag\{(\d+)\}',md)
refs=re.findall(r'^\\\[(\d+)\\\]',md,re.M)
missing_files=[]
for link in re.findall(r'\]\(([^)]+)\)',md):
 if not link.startswith(('http','#','mailto:')) and not (base/link).exists(): missing_files.append(link)
source=BeautifulSoup((base/'2611.17309v1.html').read_text(),'lxml').find('article')
for tag in list(source.select('.ltx_phantom,[style*="visibility:hidden"]')):
 if tag.parent: tag.decompose()
for tag in source.find_all('math'): tag.replace_with(' ')
for tag in list(source.select('.ltx_note')): tag.decompose()
html=pypandoc.convert_text(md,'html','gfm+tex_math_dollars+footnotes',extra_args=['--mathjax'])
(base/'review/rendered-body.html').write_text(html)
out=BeautifulSoup(html,'lxml')
for tag in out.select('.math'): tag.replace_with(' ')
def norm(s):
 s=unicodedata.normalize('NFKD',s)
 return ''.join(c.lower() for c in s if c.isalnum())
whole=norm(out.get_text(' ',strip=True))
# Cross-reference renderer changes (Sec. -> App.) and whitespace do not affect meaning.
coverage=[]
for tag in source.select('p.ltx_p, figcaption, li.ltx_bibitem'):
 if tag.find_parent('figure',class_='ltx_table') and tag.name!='figcaption': continue
 for a in tag.select('a.ltx_ref'):
  if a.get('href','').startswith('#A'):
   for word in a.find_all(string=True): word.replace_with(str(word).replace('Sec.','App.'))
 text=tag.get_text(' ',strip=True)
 clean=norm(text)
 if len(clean)>30 and clean not in whole:
  coverage.append({'id':tag.get('id'),'tag':tag.name,'text':text})
report={
 'math_expressions':len(math),'display_equation_tags':equations,'tables':tables,
 'figures':[n for n,_ in figures],'bibliography_entries':refs,
 'missing_internal_anchors':sorted(links-set(anchors)),
 'duplicate_anchors':[a for a,n in Counter(anchors).items() if n>1],
 'missing_local_assets':missing_files,
 'unconverted_markup':bool(re.search(r'PAPERTOKEN|<table|<tr>|<td>|<figcaption',md)),
 'prose_coverage_candidates':coverage,
}
(base/'review/validation.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='prose_coverage_candidates'},indent=2))
print('Prose coverage candidates for manual inspection:',len(coverage))
for item in coverage: print(item['id'],item['tag'],item['text'][:220])
assert equations==[str(n) for n in range(1,16)]
assert tables==[str(n) for n in range(1,9)]
assert [n for n,_ in figures]==[str(n) for n in range(1,7)]
assert refs==[str(n) for n in range(1,67)]
assert not any(report[k] for k in ['missing_internal_anchors','duplicate_anchors','missing_local_assets','unconverted_markup'])
