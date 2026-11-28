
from io import BytesIO
from xml.sax.saxutils import escape

from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Flowable
)

from app.database import get_db
from app.models import User, Patient, ECGAnalysis


router = APIRouter(
    prefix="/api/reports",
    tags=["PDF Reports"]
)


class ECGWaveform(Flowable):

    def __init__(self, signal, width=480, height=150):
        super().__init__()
        self.signal = signal or []
        self.width = width
        self.height = height

    def wrap(self, available_width, available_height):
        self.width = min(self.width, available_width)
        return self.width, self.height

    def draw(self):
        canvas = self.canv

        left = 35
        right = self.width - 10
        bottom = 25
        top = self.height - 15

        canvas.setStrokeColor(colors.lightgrey)
        canvas.setLineWidth(0.5)

        for i in range(5):
            y = bottom + i * (top - bottom) / 4
            canvas.line(left, y, right, y)

        canvas.setStrokeColor(colors.HexColor("#087E83"))
        canvas.setLineWidth(1)

        values = []

        for value in self.signal:
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                continue

        if len(values) < 2:
            canvas.setFillColor(colors.grey)
            canvas.setFont("Helvetica", 9)
            canvas.drawString(left, self.height / 2, "Waveform unavailable")
            return

        minimum = min(values)
        maximum = max(values)

        if maximum == minimum:
            maximum += 1
            minimum -= 1

        path = canvas.beginPath()

        for index, value in enumerate(values):
            x = left + index * (right - left) / (len(values) - 1)
            y = bottom + (
                (value - minimum) / (maximum - minimum)
            ) * (top - bottom)

            if index == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)

        canvas.drawPath(path)

        canvas.setFillColor(colors.grey)
        canvas.setFont("Helvetica", 8)
        canvas.drawCentredString(
            self.width / 2,
            5,
            "ECG Signal Samples (downsampled)"
        )


@router.get("/ecg/{analysis_id}/pdf")
def generate_ecg_report(
    analysis_id: int,
    request: Request,
    db: Session = Depends(get_db)
):

    user_id = request.session.get("user_id")

    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="Please login first"
        )

    user = db.get(User, user_id)

    if not user:
        request.session.clear()
        raise HTTPException(
            status_code=401,
            detail="User not found"
        )

    patient = db.query(Patient).filter(
        Patient.user_id == user_id
    ).first()

    if not patient:
        raise HTTPException(
            status_code=404,
            detail="Patient profile not found"
        )

    analysis = db.query(ECGAnalysis).filter(
        ECGAnalysis.id == analysis_id,
        ECGAnalysis.patient_id == patient.id
    ).first()

    if not analysis:
        raise HTTPException(
            status_code=404,
            detail="ECG analysis not found"
        )

    buffer = BytesIO()

    pdf = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=45,
        leftMargin=45,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=24,
        textColor=colors.HexColor("#087E83"),
        alignment=TA_CENTER,
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontSize=9,
        alignment=TA_CENTER,
        textColor=colors.grey
    )

    heading_style = ParagraphStyle(
        "ReportHeading",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=colors.HexColor("#087E83"),
        spaceBefore=15,
        spaceAfter=8
    )

    story = []

    story.append(Paragraph("PulseAI", title_style))

    story.append(
        Paragraph(
            "AI-POWERED PATIENT HEALTH MONITORING SYSTEM",
            subtitle_style
        )
    )

    story.append(Spacer(1, 18))

    story.append(
        Paragraph("DETAILED ECG ANALYSIS REPORT", heading_style)
    )

    report_date = (
        analysis.created_at.strftime("%d %B %Y, %I:%M %p")
        if analysis.created_at else "Not available"
    )

    patient_data = [
        ["Patient Information", "Details"],
        ["Patient Name", escape(patient.name or "Not available")],
        ["Patient ID", str(patient.id)],
        ["Age", str(patient.age) if patient.age else "Not provided"],
        ["Gender", escape(patient.gender or "Not provided")],
        ["Email", escape(user.email)],
        ["Analysis ID", str(analysis.id)],
        ["Analysis Date", report_date]
    ]

    patient_table = Table(
        patient_data,
        colWidths=[180, 290]
    )

    patient_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#087E83")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#EAF5F5")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
    ]))

    story.append(patient_table)

    story.append(
        Paragraph("ECG Analysis Overview", heading_style)
    )

    metrics = [
        ["Parameter", "Recorded Result"],
        [
            "Heart Rate",
            f"{analysis.heart_rate:.2f} BPM"
            if analysis.heart_rate is not None else "Not available"
        ],
        [
            "Total Detected Beats",
            str(analysis.total_beats)
            if analysis.total_beats is not None else "Not available"
        ],
        [
            "Abnormal Beats",
            str(analysis.abnormal_beats)
            if analysis.abnormal_beats is not None else "Not available"
        ],
        [
            "Abnormal Beat Burden",
            escape(analysis.abnormal_beat_burden or "Not available")
        ],
        [
            "Average Model Confidence",
            f"{analysis.average_confidence:.2f}%"
            if analysis.average_confidence is not None else "Not available"
        ],
        [
            "Sampling Rate",
            f"{analysis.sampling_rate} Hz"
            if analysis.sampling_rate else "Not available"
        ],
        [
            "Total Samples",
            str(analysis.total_samples)
            if analysis.total_samples is not None else "Not available"
        ]
    ]

    metrics_table = Table(
        metrics,
        colWidths=[235, 235]
    )

    metrics_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#173C65")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#F0F4F9")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("PADDING", (0, 0), (-1, -1), 8)
    ]))

    story.append(metrics_table)

    story.append(
        Paragraph("Heartbeat Class Distribution", heading_style)
    )

    class_counts = analysis.class_counts or {}
    class_percentages = analysis.class_percentages or {}

    class_data = [
        ["AAMI Class", "Beat Count", "Percentage"]
    ]

    for class_name in ["N", "S", "V", "F", "Q"]:

        count = class_counts.get(class_name, 0)
        percentage = class_percentages.get(class_name, 0)

        class_data.append([
            class_name,
            str(count),
            f"{float(percentage):.2f}%"
        ])

    class_table = Table(
        class_data,
        colWidths=[150, 160, 160]
    )

    class_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#087E83")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("PADDING", (0, 0), (-1, -1), 8)
    ]))

    story.append(class_table)

    story.append(
        Paragraph("ECG Waveform Visualization", heading_style)
    )

    if analysis.filtered_signal:
        story.append(
            ECGWaveform(analysis.filtered_signal)
        )
    else:
        story.append(
            Paragraph(
                "Waveform is unavailable for this historical analysis.",
                styles["BodyText"]
            )
        )

    story.append(
        Paragraph("Analysis Summary", heading_style)
    )

    summary = escape(
        analysis.result_summary or
        "No additional analysis summary was stored."
    )

    story.append(
        Paragraph(summary, styles["BodyText"])
    )

    story.append(
        Paragraph("Stored Classification", heading_style)
    )

    story.append(
        Paragraph(
            escape(analysis.risk_level or "Not available"),
            styles["BodyText"]
        )
    )

    story.append(
        Paragraph("Important Medical Disclaimer", heading_style)
    )

    disclaimer = (
        "PulseAI is an experimental academic research prototype. "
        "Its AI predictions and measurements have not been clinically "
        "validated. Model confidence is not a measure of clinical accuracy. "
        "This report is not a medical diagnosis and must not be used for "
        "treatment decisions or emergency assessment. Consult a qualified "
        "healthcare professional for interpretation of ECG results."
    )

    story.append(
        Paragraph(disclaimer, styles["BodyText"])
    )

    story.append(Spacer(1, 20))

    story.append(
        Paragraph(
            "Generated by PulseAI | Academic Research Project",
            subtitle_style
        )
    )

    pdf.build(story)

    buffer.seek(0)

    filename = f"PulseAI_Detailed_ECG_Report_{analysis.id}.pdf"

    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )
