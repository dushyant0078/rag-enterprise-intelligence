"""Generate small, realistic sample documents across every supported format
and domain, so the ingestion pipeline (Milestone 3) has something concrete
and locally-available to run against.

This exists because this development sandbox's network allowlist doesn't
reach sec.gov or arxiv.org (see scripts/download_datasets.py for the real
bulk-download script, meant to be run in an environment with open internet
access, not here). These samples are small and synthetic but structurally
representative: multi-page PDFs, a table-bearing DOCX, a slide deck with
speaker notes, and a genuinely scanned (image-only, no text layer) PDF to
exercise the OCR fallback path end-to-end.

Run: python scripts/generate_sample_documents.py
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
from docx import Document
from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation

OUT = Path("documents")


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"  wrote {path}")


def make_hr_docx() -> None:
    path = OUT / "hr" / "remote_work_policy.docx"
    path.parent.mkdir(parents=True, exist_ok=True)

    doc = Document()
    doc.add_heading("Remote Work Policy", level=1)
    doc.add_paragraph("Effective Date: 2025-01-01   Revision: 3   Owner: People Operations")
    doc.add_paragraph(
        "This policy governs remote and hybrid work arrangements for all full-time employees. "
        "It applies from the employee's first day of employment; there is no minimum tenure "
        "requirement for eligibility."
    )
    doc.add_heading("Eligibility by Level", level=2)
    table = doc.add_table(rows=1, cols=3)
    table.style = "Light Grid Accent 1"
    hdr = table.rows[0].cells
    hdr[0].text, hdr[1].text, hdr[2].text = "Job Level", "Remote Days / Week", "Approval Required"
    rows = [
        ("Individual Contributor", "Up to 5", "Manager"),
        ("Team Lead", "Up to 4", "Manager + HRBP"),
        ("Director and above", "Up to 3", "VP + HRBP"),
    ]
    for level, days, approval in rows:
        cells = table.add_row().cells
        cells[0].text, cells[1].text, cells[2].text = level, days, approval

    doc.add_page_break()
    doc.add_heading("Parental Leave Interaction", level=2)
    doc.add_paragraph(
        "Employees on parental leave who return to a remote or hybrid arrangement retain their "
        "prior remote-days eligibility for the first 90 days back, after which standard "
        "level-based eligibility applies."
    )
    doc.save(str(path))
    print(f"  wrote {path}")


def make_hr_pdf() -> None:
    """A genuine multi-page, native-text PDF (no OCR needed)."""
    path = OUT / "hr" / "employee_handbook_excerpt.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)

    pages_text = [
        "Employee Handbook — Section 4: Leave Policies\n\n"
        "4.1 Parental Leave\n"
        "Employees with at least one year of continuous service are eligible for 26 weeks of "
        "paid parental leave. Employees in their first year of employment are eligible for "
        "12 weeks of paid parental leave, per the 2025 handbook revision.",
        "4.2 Sick Leave\n\n"
        "All employees accrue 1.5 days of sick leave per month of employment, up to a maximum "
        "carryover of 15 days per calendar year. Unused sick leave does not convert to a cash "
        "payout upon separation.",
        "4.3 Bereavement Leave\n\n"
        "Employees are entitled to 5 paid bereavement days for an immediate family member and "
        "3 paid bereavement days for an extended family member, as defined in Appendix C.",
    ]

    pdf = pymupdf.open()
    for text in pages_text:
        page = pdf.new_page(width=595, height=842)  # A4
        page.insert_text((72, 72), text, fontsize=11, fontname="helv")
    pdf.save(str(path))
    pdf.close()
    print(f"  wrote {path}")


def make_legal_contract_txt() -> None:
    path = OUT / "legal" / "vendor_msa_sample.txt"
    _write(
        path,
        "MASTER SERVICES AGREEMENT (Sample)\n\n"
        'This Master Services Agreement ("Agreement") is entered into as of March 1, 2025.\n\n'
        "Section 8.2 — Termination for Convenience\n"
        "Either party may terminate this Agreement for convenience upon sixty (60) days' prior "
        "written notice to the other party.\n\n"
        "Section 12.1 — Auto-Renewal\n"
        "This Agreement automatically renews for successive one-year terms unless either party "
        "provides written notice of non-renewal at least forty-five (45) days before the end of "
        "the then-current term.\n\n"
        "Section 15.4 — Limitation of Liability\n"
        "Neither party's aggregate liability under this Agreement shall exceed the total fees "
        "paid in the twelve (12) months preceding the claim.\n",
    )


def make_banking_policy_markdown() -> None:
    path = OUT / "banking" / "aml_policy.md"
    _write(
        path,
        "# Anti-Money Laundering Policy\n\n"
        "**Effective date:** 2025-06-01  \n"
        "**Owner:** Compliance Department\n\n"
        "## Enhanced Due Diligence Thresholds\n\n"
        "| Transaction Type | Threshold | EDD Required |\n"
        "|---|---|---|\n"
        "| Domestic wire transfer | $10,000 | Yes |\n"
        "| International wire transfer | $3,000 | Yes |\n"
        "| Cash deposit | $10,000 | Yes |\n\n"
        "## Ongoing Monitoring\n\n"
        "Accounts flagged under Enhanced Due Diligence are reviewed quarterly by the Compliance "
        "Department. Any account showing a sustained pattern of transactions within 10% of a "
        "reporting threshold is escalated to the AML Officer for manual review.\n",
    )


def make_insurance_html() -> None:
    path = OUT / "insurance" / "homeowners_policy_faq.html"
    _write(
        path,
        "<html><head><title>Homeowners Policy FAQ</title></head><body>"
        "<h1>Homeowners Policy Form HO-3 — Frequently Asked Questions</h1>"
        "<h2>Water Damage</h2>"
        "<p>Sudden and accidental water damage from a burst pipe is covered under Coverage A "
        "(Dwelling) and Coverage C (Personal Property), subject to your deductible.</p>"
        "<p><strong>Vacancy exclusion:</strong> if the property was vacant or unoccupied for "
        "more than 60 consecutive days immediately before the loss, water damage coverage is "
        "excluded unless the Vacancy Permit endorsement was purchased.</p>"
        "<table><caption>Standard Deductibles</caption>"
        "<tr><th>Peril</th><th>Deductible</th></tr>"
        "<tr><td>Water damage</td><td>$2,500</td></tr>"
        "<tr><td>Wind/Hail</td><td>2% of Coverage A</td></tr>"
        "<tr><td>All other perils</td><td>$1,000</td></tr>"
        "</table>"
        "</body></html>",
    )


def make_research_paper_txt() -> None:
    path = OUT / "research" / "imbalanced_classification_abstract.txt"
    _write(
        path,
        "Title: Adaptive Hybrid Oversampling for Severely Imbalanced Tabular Classification\n\n"
        "Abstract: Class imbalance remains a central obstacle to reliable classification in "
        "high-stakes tabular domains such as fraud detection and rare-disease diagnosis. We "
        "propose an adaptive hybrid oversampling strategy that combines density-based synthetic "
        "sample generation with a dynamically tuned minority-class weighting term, evaluated "
        "against SMOTE, ADASYN, and cost-sensitive baselines across five public benchmark "
        "datasets. Our evaluation methodology reports F1, PR-AUC, and Matthews correlation "
        "coefficient rather than accuracy alone, given severe class imbalance in every dataset "
        "considered.\n\n"
        "1. Introduction\n"
        "Standard classifiers trained on imbalanced data tend to optimize overall accuracy at "
        "the expense of minority-class recall, which is precisely the failure mode that matters "
        "most in fraud and risk applications...\n",
    )


def make_technical_sop_pptx() -> None:
    path = OUT / "technical" / "sop_lockout_tagout.pptx"
    path.parent.mkdir(parents=True, exist_ok=True)

    prs = Presentation()
    title_slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(title_slide_layout)
    slide.shapes.title.text = "SOP-014: Lockout-Tagout Procedure"
    slide.placeholders[1].text = "Model X-200 Servicing — Rev. 4"

    content_layout = prs.slide_layouts[1]

    slide2 = prs.slides.add_slide(content_layout)
    slide2.shapes.title.text = "Before Servicing"
    body = slide2.placeholders[1].text_frame
    body.text = "1. Notify affected personnel"
    for line in [
        "2. Shut down the Model X-200 using the normal stop procedure",
        "3. Isolate all energy sources at the primary disconnect",
        "4. Apply lock and tag to the primary disconnect",
        "5. Verify zero energy state before beginning work",
    ]:
        p = body.add_paragraph()
        p.text = line
        p.level = 0
    slide2.notes_slide.notes_text_frame.text = (
        "Do not restore line power until all locks and tags from every technician working on "
        "the unit have been removed. A single remaining lock means the unit stays de-energized."
    )

    slide3 = prs.slides.add_slide(content_layout)
    slide3.shapes.title.text = "Restoring Power"
    body3 = slide3.placeholders[1].text_frame
    body3.text = "1. Confirm all tools removed and guards reinstalled"
    for line in [
        "2. Confirm all personnel are clear of the equipment",
        "3. Remove locks and tags in reverse order of application",
        "4. Restore energy at the primary disconnect",
        "5. Return the Model X-200 to normal operation",
    ]:
        p = body3.add_paragraph()
        p.text = line
        p.level = 0

    prs.save(str(path))
    print(f"  wrote {path}")


def make_financial_csv() -> None:
    path = OUT / "financial" / "quarterly_metrics.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "quarter,revenue_usd_millions,gross_margin_pct,top_risk_factor\n"
        '2024-Q4,412.6,58.2,"Supply chain concentration"\n'
        '2025-Q1,438.1,59.0,"Regulatory change in EU market"\n'
        '2025-Q2,451.7,58.6,"Regulatory change in EU market"\n'
        '2025-Q3,467.3,60.1,"Foreign currency exposure"\n',
        encoding="utf-8",
    )
    print(f"  wrote {path}")


def make_scanned_pdf() -> None:
    """A genuinely image-only PDF (no text layer) — exercises the OCR fallback
    path in src/parsing/pdf_parser.py end-to-end.
    """
    path = OUT / "scanned" / "scanned_facility_notice.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)

    # Render text onto a raster image using PIL (this is the "scan").
    img = Image.new("RGB", (1240, 1754), color="white")  # ~150dpi A4
    draw = ImageDraw.Draw(img)
    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 34)
        font_body = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 26)
    except OSError:
        font_title = ImageFont.load_default()
        font_body = ImageFont.load_default()

    lines = [
        ("FACILITY SAFETY NOTICE", font_title),
        ("", font_body),
        ("Effective immediately, all visitors to Building 4 must sign in", font_body),
        ("at the main security desk and wear a visible visitor badge", font_body),
        ("at all times while on the premises.", font_body),
        ("", font_body),
        ("Contractors performing lockout-tagout work must present a", font_body),
        ("valid LOTO authorization card issued by Facilities Safety.", font_body),
        ("", font_body),
        ("Questions: facilities-safety@example-corp.internal", font_body),
    ]
    y = 90
    for text, font in lines:
        draw.text((90, y), text, fill="black", font=font)
        y += 55

    img_path = path.with_suffix(".png")
    img.save(img_path)

    pdf = pymupdf.open()
    page = pdf.new_page(width=img.width, height=img.height)
    page.insert_image(pymupdf.Rect(0, 0, img.width, img.height), filename=str(img_path))
    pdf.save(str(path))
    pdf.close()
    img_path.unlink()  # intermediate artifact only
    print(f"  wrote {path} (image-only — no text layer, requires OCR)")


def main() -> None:
    print("Generating sample documents in documents/ ...")
    make_hr_docx()
    make_hr_pdf()
    make_legal_contract_txt()
    make_banking_policy_markdown()
    make_insurance_html()
    make_research_paper_txt()
    make_technical_sop_pptx()
    make_financial_csv()
    make_scanned_pdf()
    print("Done.")


if __name__ == "__main__":
    main()
