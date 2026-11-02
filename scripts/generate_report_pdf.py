"""Compiles docs/report.md into a professional academic PDF (docs/report.pdf) using ReportLab."""

import os
import re
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#58626B"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 750, "Edge Health Monitor: Final Academic Project Report")
            self.drawRightString(612 - 54, 750, "TinyML Healthcare System")
            self.setStrokeColor(colors.HexColor("#CBC8BF"))
            self.setLineWidth(0.5)
            self.line(54, 744, 612 - 54, 744)
            
        # Footer
        self.setStrokeColor(colors.HexColor("#CBC8BF"))
        self.setLineWidth(0.5)
        self.line(54, 45, 612 - 54, 45)
        self.drawString(54, 32, "Introduction to AI and Embedded Systems")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(612 - 54, 32, page_str)
        self.restoreState()

def build_pdf():
    report_md_path = Path("docs/report.md")
    pdf_out_path = Path("docs/report.pdf")
    
    if not report_md_path.exists():
        print(f"Error: {report_md_path} does not exist.")
        return False

    with open(report_md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    doc = SimpleDocTemplate(
        str(pdf_out_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    
    # Custom engineering palette styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#1E2328'),
        spaceAfter=12
    )
    
    h1_style = ParagraphStyle(
        'Heading1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#1F6F78'),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'Heading2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=13,
        textColor=colors.HexColor('#1E2328'),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#1E2328'),
        spaceAfter=6
    )

    code_style = ParagraphStyle(
        'Code',
        parent=styles['Code'],
        fontName='Courier',
        fontSize=7,
        leading=9.5,
        textColor=colors.HexColor('#1E2328'),
        spaceBefore=4,
        spaceAfter=6
    )

    meta_style = ParagraphStyle(
        'Meta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#58626B'),
        spaceAfter=4
    )
    
    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7,
        leading=9,
        textColor=colors.HexColor('#1E2328')
    )
    
    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7,
        leading=9,
        textColor=colors.HexColor('#1E2328')
    )

    story = []
    
    lines = md_text.split("\n")
    i = 0
    in_code_block = False
    code_lines = []
    
    while i < len(lines):
        line = lines[i]
        
        # Handle code blocks
        if line.startswith("```"):
            if in_code_block:
                in_code_block = False
                code_text = "<br/>".join([c.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace(" ", "&nbsp;") for c in code_lines])
                story.append(Paragraph(code_text, code_style))
                story.append(Spacer(1, 4))
                code_lines = []
            else:
                in_code_block = True
            i += 1
            continue
            
        if in_code_block:
            code_lines.append(line)
            i += 1
            continue
            
        # Skip horizontal rules
        if line.strip() in ("---", "***"):
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBC8BF"), spaceAfter=8, spaceBefore=4))
            i += 1
            continue
            
        # Title
        if line.startswith("# "):
            title_text = line[2:].strip()
            story.append(Paragraph(title_text, title_style))
            i += 1
            continue
            
        # Headings
        if line.startswith("## "):
            story.append(Paragraph(line[3:].strip(), h1_style))
            i += 1
            continue
        if line.startswith("### "):
            story.append(Paragraph(line[4:].strip(), h2_style))
            i += 1
            continue
        if line.startswith("#### "):
            story.append(Paragraph(f"<b>{line[5:].strip()}</b>", h2_style))
            i += 1
            continue
            
        # Metadata lines
        if line.startswith("**Course:**") or line.startswith("**Project:**") or line.startswith("**Document Type:**") or line.startswith("**Date:**") or line.startswith("**System Status:**"):
            meta_html = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', line)
            story.append(Paragraph(meta_html, meta_style))
            i += 1
            continue
            
        # Markdown Table Detection
        if "|" in line and i + 1 < len(lines) and "|---" in lines[i+1]:
            header_cells = [c.strip() for c in line.strip().strip("|").split("|")]
            i += 2 # Skip separator line
            table_rows = []
            while i < len(lines) and "|" in lines[i] and not lines[i].startswith("#"):
                row_cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                table_rows.append(row_cells)
                i += 1
                
            # Render Table
            table_data = []
            table_data.append([Paragraph(f"<b>{c}</b>", table_header_style) for c in header_cells])
            for r in table_rows:
                # pad or trim
                row_pars = []
                for c in r:
                    formatted_c = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', c)
                    formatted_c = re.sub(r'\*([^*]+)\*', r'<i>\1</i>', formatted_c)
                    row_pars.append(Paragraph(formatted_c, table_cell_style))
                while len(row_pars) < len(header_cells):
                    row_pars.append(Paragraph("", table_cell_style))
                table_data.append(row_pars[:len(header_cells)])
                
            num_cols = len(header_cells)
            available_width = 504.0 # 612 - 108
            col_width = available_width / num_cols
            t = Table(table_data, colWidths=[col_width] * num_cols)
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ECEAE4')),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
                ('TOPPADDING', (0, 0), (-1, -1), 3),
                ('LEFTPADDING', (0, 0), (-1, -1), 4),
                ('RIGHTPADDING', (0, 0), (-1, -1), 4),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBC8BF')),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ]))
            story.append(Spacer(1, 4))
            story.append(KeepTogether(t))
            story.append(Spacer(1, 6))
            continue
            
        # Bullet list
        if line.strip().startswith("- ") or line.strip().startswith("* "):
            bullet_text = line.strip()[2:].strip()
            bullet_html = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', bullet_text)
            bullet_html = re.sub(r'\*([^*]+)\*', r'<i>\1</i>', bullet_html)
            story.append(Paragraph(f"• {bullet_html}", body_style))
            i += 1
            continue
            
        # Numbered list
        num_match = re.match(r'^(\d+)\.\s+(.*)', line.strip())
        if num_match:
            num_text = num_match.group(2)
            num_html = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', num_text)
            num_html = re.sub(r'\*([^*]+)\*', r'<i>\1</i>', num_html)
            story.append(Paragraph(f"{num_match.group(1)}. {num_html}", body_style))
            i += 1
            continue
            
        # Standard paragraph
        if line.strip():
            clean_p = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', line.strip())
            clean_p = re.sub(r'\*([^*]+)\*', r'<i>\1</i>', clean_p)
            clean_p = re.sub(r'`([^`]+)`', r'<font face="Courier">\1</font>', clean_p)
            # KaTeX / math cleanup for plain reportlab
            clean_p = clean_p.replace("$$", "").replace("$", "")
            story.append(Paragraph(clean_p, body_style))
            
        i += 1

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Generated PDF successfully: {pdf_out_path} ({os.path.getsize(pdf_out_path) / 1024:.1f} KB)")
    return True

if __name__ == "__main__":
    build_pdf()
