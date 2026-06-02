"""
Transfer 论文.md content into 230-A.docx template with proper styles.
Fixed parser: handles single-line $$...$$ math without consuming subsequent lines.
"""
import docx
from docx import Document
from docx.shared import Pt, Inches, Cm, Emu
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
import os, re

BASE = 'D:/数学建模比赛/20260506校赛/zjy'
MD_FILE = f'{BASE}/论文/论文.md'
TEMPLATE_FILE = f'{BASE}/论文/230-A.docx'
OUTPUT_FILE = f'{BASE}/论文/230-A修改中.docx'

# ============================================================
# Parse MD line by line
# ============================================================
def parse_md(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    blocks = []
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()

        # Separator
        if stripped == '---':
            blocks.append(('sep', None))
            i += 1; continue

        # Heading (#, ##, ###, ####)
        m = re.match(r'^(#{1,4})\s+(.+)', line)
        if m:
            blocks.append(('heading', {'level': len(m.group(1)), 'text': m.group(2).strip()}))
            i += 1; continue

        # Image (![alt](path)) — must be at line start
        m = re.match(r'^!\[(.*?)\]\((.*?)\)', line)
        if m:
            blocks.append(('image', {'alt': m.group(1), 'path': m.group(2)}))
            i += 1; continue

        # Table (|...|...|)
        if stripped.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                cells = [c.strip() for c in lines[i].strip().split('|')[1:-1]]
                if not all(re.match(r'^[-:]+$', c) for c in cells if c):
                    rows.append(cells)
                i += 1
            if rows:
                blocks.append(('table', rows))
            continue

        # Display math $$ (single-line: $$...$$, multi-line: $$ \n ... \n $$)
        if stripped.startswith('$$'):
            # Check if single-line (closing $$ on same line)
            if stripped.endswith('$$') and len(stripped) > 4:
                blocks.append(('math', stripped))
                i += 1; continue
            # Multi-line math
            math_lines = [stripped]
            i += 1
            while i < len(lines):
                math_lines.append(lines[i].rstrip())
                if lines[i].strip().startswith('$$'):
                    i += 1; break
                i += 1
            blocks.append(('math', '\n'.join(math_lines)))
            continue

        # Empty line
        if not stripped:
            blocks.append(('blank', None))
            i += 1; continue

        # Regular paragraph (may contain inline $...$ or **...**)
        blocks.append(('para', stripped))
        i += 1

    return blocks


# ============================================================
# Build DOCX
# ============================================================
def build_docx(blocks, template_path, output_path):
    doc = Document(template_path)

    # Clear existing paragraphs (keep first empty paragraph as anchor)
    for p in list(doc.paragraphs):
        p._element.getparent().remove(p._element)
    for t in list(doc.tables):
        t._element.getparent().remove(t._element)

    # ---- Helper functions ----
    def add_run_to_paragraph(p, text, bold=False, size=None, color=None):
        """Add a run to paragraph, handling special characters."""
        if not text:
            return
        run = p.add_run(text)
        if bold:
            run.bold = True
        if size:
            run.font.size = Pt(size)
        if color:
            run.font.color.rgb = RGBColor(*color)
        return run

    def add_styled_para(text, style_name, font_size=None, alignment=None):
        """Add paragraph with a specific style, handling **bold** markers."""
        p = doc.add_paragraph(style=style_name)
        if alignment is not None:
            p.alignment = alignment

        # Split text by **bold** markers
        parts = re.split(r'(\*\*.*?\*\*)', text)
        for part in parts:
            if part.startswith('**') and part.endswith('**'):
                add_run_to_paragraph(p, part[2:-2], bold=True, size=font_size)
            else:
                # Remove stray $ from inline math
                clean = part.replace('$', '')
                add_run_to_paragraph(p, clean, size=font_size)
        return p

    def add_image(alt, rel_path):
        """Insert image with centered alignment."""
        candidates = [
            os.path.join(BASE, 'code', rel_path),
            os.path.join(BASE, 'code', 'output', 'figures', os.path.basename(rel_path)),
        ]
        img_path = None
        for c in candidates:
            if os.path.exists(c):
                img_path = c; break

        if img_path:
            try:
                p = doc.add_paragraph(style='正')
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.add_run().add_picture(img_path, width=Cm(14))
                # Caption goes after image as a separate paragraph
            except Exception as e:
                doc.add_paragraph(f'[Image error: {alt} — {e}]', style='正')
        else:
            doc.add_paragraph(f'[Image: {alt}]', style='正')

    def add_caption(text):
        """Add figure caption (centered, small font). The ** markers are already in text."""
        p = doc.add_paragraph(style='正')
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        parts = re.split(r'(\*\*.*?\*\*)', text)
        for part in parts:
            if part.startswith('**') and part.endswith('**'):
                add_run_to_paragraph(p, part[2:-2], bold=True, size=9)
            else:
                add_run_to_paragraph(p, part, size=9)
        return p

    def add_table(rows):
        if not rows: return
        ncols = max(len(r) for r in rows)
        table = doc.add_table(rows=len(rows), cols=ncols)
        table.style = 'Table Grid'
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, row_data in enumerate(rows):
            for j in range(ncols):
                cell_text = row_data[j] if j < len(row_data) else ''
                cell = table.cell(i, j)
                cell.paragraphs[0].clear()
                run = cell.paragraphs[0].add_run(cell_text)
                run.font.size = Pt(7.5)
                if i == 0:
                    run.bold = True
                try:
                    cell.paragraphs[0].style = doc.styles['表内']
                except:
                    pass
        doc.add_paragraph('', style='Normal')

    # ---- Process each block ----
    for btype, bdata in blocks:
        if btype == 'sep':
            doc.add_paragraph('', style='Normal')
            continue

        if btype == 'blank':
            continue  # Skip blank lines (heading spacing is handled by styles)

        if btype == 'heading':
            level = bdata['level']
            text = bdata['text']
            if level == 1:
                add_styled_para(text, 'Title', font_size=18, alignment=WD_ALIGN_PARAGRAPH.CENTER)
            elif level == 2:
                if text in ('摘要', '参考文献', '附录'):
                    add_styled_para(text, 'Heading 1')
                elif text.startswith('附录'):
                    add_styled_para(text, 'Heading 2')
                else:
                    add_styled_para(text, 'Heading 1')
            elif level == 3:
                add_styled_para(text, 'Heading 2')
            elif level == 4:
                add_styled_para(text, 'Heading 3')
            continue

        if btype == 'image':
            add_image(bdata['alt'], bdata['path'])
            continue

        if btype == 'table':
            add_table(bdata)
            continue

        if btype == 'math':
            # Display math as italic paragraph
            clean = bdata.replace('$$', '').strip()
            p = doc.add_paragraph(style='正')
            add_run_to_paragraph(p, clean, size=10)
            continue

        if btype == 'para':
            text = bdata

            # Keywords line
            if text.startswith('关键词'):
                p = doc.add_paragraph(style='Normal')
                add_run_to_paragraph(p, '关键词：', bold=True, size=10.5)
                rest = text.replace('关键词：', '').replace('**', '').replace('；', '；')
                add_run_to_paragraph(p, rest, size=10.5)
                continue

            # Figure caption (**图X ...**)
            if re.match(r'^\*\*图\d+', text):
                add_caption(text)
                continue

            # Regular paragraph
            # Clean inline math markers
            if '$' in text:
                text = re.sub(r'\$([^$]+)\$', r'\1', text)
            add_styled_para(text, '正', font_size=10.5)
            continue

    # ---- Post-processing ----
    # Set abstract paragraph to 中 style
    for idx, p in enumerate(doc.paragraphs):
        if p.text.strip() == '摘要' and p.style.name == 'Heading 1':
            for j in range(idx+1, min(idx+5, len(doc.paragraphs))):
                if doc.paragraphs[j].text.strip():
                    doc.paragraphs[j].style = doc.styles['中']
                    break

    doc.save(output_path)
    print(f'Saved: {output_path}')


if __name__ == '__main__':
    print(f'Parsing {MD_FILE}...')
    blocks = parse_md(MD_FILE)
    print(f'Total blocks: {len(blocks)}')

    types = {}
    for bt, bd in blocks:
        types[bt] = types.get(bt, 0) + 1
    for t, c in sorted(types.items()):
        print(f'  {t}: {c}')

    print(f'\nBuilding DOCX from template...')
    build_docx(blocks, TEMPLATE_FILE, OUTPUT_FILE)
    print('Done!')
