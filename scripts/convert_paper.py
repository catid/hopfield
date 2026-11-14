"""Create review drafts from authoritative arXiv HTML; manual review follows.
Run: uv run --with beautifulsoup4 --with lxml --with pypandoc_binary python scripts/convert_paper.py
Does not overwrite the final reviewed Markdown.
"""
from pathlib import Path
import re
import json
import pypandoc
from bs4 import BeautifulSoup, NavigableString

ROOT = Path(__file__).resolve().parents[1]
soup = BeautifulSoup((ROOT/'references/2611.17309v1.html').read_text(), 'lxml')
article = soup.find('article')
replacements = {}
def token(value):
    key = f'PAPERTOKEN{len(replacements):05d}END'
    replacements[key] = value
    return key

def tex(s):
    # Preserve role colors with portable, named hex syntax rather than rgb decimals.
    colors = {'0.1641,0.4688,0.8398':'#2A78D6','0.9219,0.4063,0.2031':'#EB6834','0.0938,0.6406,0.4453':'#18A372','0.2891,0.2266,0.6523':'#4A3AA7'}
    def col(m):
        key=m.group(1)
        if key not in colors:
            vals=[float(v) for v in key.split(',')]
            return r'\color{'+''.join(['#']+[f'{round(v*256):02X}' for v in vals])+'}'
        return r'\color{'+colors[key]+'}'
    return re.sub(r'\\color\[rgb\]\{([^}]+)\}',col,s)

# Remove invisible layout phantoms, not substantive content.
for e in list(article.select('.ltx_phantom, [style*="visibility:hidden"]')):
    if e.parent: e.decompose()

# Keep stable anchors for every cross-reference target.
targets={a.get('href')[1:] for a in article.find_all('a',href=True) if a.get('href').startswith('#')}
for e in list(article.find_all(id=True)):
    if e['id'] in targets:
        anchor = NavigableString(token(f'\n<a id="{e["id"]}"></a>\n'))
        if e.name == 'section': e.insert(0,anchor)
        else: e.insert_before(anchor)

# Display math is split into multiple cells by LaTeXML. Rejoin each row.
for eq in list(article.select('table.ltx_eqn_table')):
    rows=[]
    for row in eq.select('tr.ltx_eqn_row'):
        cells=[]
        for cell in row.find_all(['td','th'],recursive=False):
            mts=cell.find_all('math')
            if mts: cells.append(' '.join(tex(m.get('alttext','')) for m in mts))
        tag=row.select_one('.ltx_tag_equation')
        # A tag belongs to the row and must not be dropped when joining cells.
        tag_text=tag.get_text('',strip=True).strip('()') if tag else ''
        if cells: rows.append((' & '.join(cells),tag_text))
    if len(rows)==1:
        body=rows[0][0].replace(' & ',' ')
        if rows[0][1]: body+=r' \tag{'+rows[0][1]+'}'
    else:
        body=r'\begin{aligned}'+'\n'+(' \\\n'.join(r[0] for r in rows))+'\n'+r'\end{aligned}'
        tags=[r[1] for r in rows if r[1]]
        if tags: body+=r' \tag{'+','.join(tags)+'}'
    eq.replace_with(NavigableString(token('\n\n$$\n'+body+'\n$$\n\n')))

for math in list(article.find_all('math')):
    expression=tex(math.get('alttext',''))
    delim='$$' if math.get('display')=='block' else '$'
    math.replace_with(NavigableString(token(delim+expression+delim)))

figures={'S1.F1':'overview','S4.F2':'compute-couplings','S4.F3':'difficulty-data-efficiency','S4.F4':'alt-training','A3.F5':'hand-set-energies','A6.F6':'maze-hard'}
for fid,name in figures.items():
    fig=article.find(id=fid)
    for e in list(fig.find_all(['img','object'])): e.decompose()
    number=re.search(r'F(\d+)$',fid)[1]
    fig.insert(0,NavigableString(token(f'\n\n![Figure {number}](figures/{name}.png)\n\n[Original vector figure](source/figures/{name}.pdf)\n\n')))

# Strip CSS-only wrappers but retain structural emphasis.
for span in list(article.find_all('span')):
    if not span.parent: continue
    classes=span.get('class',[])
    if 'ltx_font_bold' in classes: span.name='strong'; span.attrs={}
    elif 'ltx_font_italic' in classes: span.name='em'; span.attrs={}
    elif 'ltx_font_typewriter' in classes: span.name='code'; span.attrs={}
    else: span.unwrap()
for e in article.find_all(True):
    for k in list(e.attrs):
        if k not in ['href','src','alt','colspan','rowspan']: del e[k]

parts=[('00_front',list(article.find_all(recursive=False))[:4]),
       ('01_main',[article.find(id=x) for x in []]),]
# IDs have been removed; identify top-level sections by their original order.
children=list(article.find_all(recursive=False))
sections=[e for e in children if e.name=='section']
parts=[('00_front',[e for e in children if e.name!='section' and e.name!='div' or e.name=='div' and e.find('section') is None][:3]),
       ('01_main',sections[:6]),('02_references',sections[6:7]),
       ('03_appendices_abc',sections[7:10]),('04_appendices_def',sections[10:13])]
# Front matter includes affiliation converted from the standalone footnote span.
first_section=sections[0]
front=[]
for e in article.children:
    if e is first_section: break
    front.append(str(e))
parts[0]=('00_front',front)
for name,els in parts:
    html='\n'.join(str(e) for e in els)
    fragment=BeautifulSoup(html,'lxml')
    for el in fragment.find_all('figcaption'): el.name='p'
    for el in list(fragment.find_all(['div','section','figure'])): el.unwrap()
    html=str(fragment)
    md=pypandoc.convert_text(html,'gfm+tex_math_dollars','html',extra_args=['--wrap=none'])
    for k,v in replacements.items(): md=md.replace(k,v)
    md=re.sub(r'\n{3,}','\n\n',md)
    (ROOT/'references/review'/f'{name}.md').write_text(md)
(ROOT/'references/review/conversion_inventory.json').write_text(json.dumps({'source_math_count':458,'figure_count':6,'table_count':8,'bibliography_count':66,'replacement_count':len(replacements)},indent=2)+'\n')
