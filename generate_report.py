"""
Academic Report Generator for Machine Learning Assignment 1
===========================================================
Title: Assignment-1: Polynomial Regression and Regularization
Course Name: Machine Learning
Name: Divyanshu Ghosh
Roll Number: IMT2024068
Target Length: Exactly 4 to 5 Pages (Built for 5 Pages)
"""

import os
import json
import pandas as pd
import numpy as np

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable, PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY, TA_RIGHT
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Register TrueType Arial fonts
pdfmetrics.registerFont(TTFont("Arial", "C:/Windows/Fonts/arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Bold", "C:/Windows/Fonts/arialbd.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Italic", "C:/Windows/Fonts/ariali.ttf"))
pdfmetrics.registerFont(TTFont("Arial-BoldItalic", "C:/Windows/Fonts/arialbi.ttf"))
pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic", boldItalic="Arial-BoldItalic")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VAR1_DIR = os.path.join(SCRIPT_DIR, "var1")
VAR2_DIR = os.path.join(SCRIPT_DIR, "var2")
PDF_PATH = os.path.join(SCRIPT_DIR, "report.pdf")

# Load verified experiment metadata
with open(os.path.join(VAR1_DIR, "var1_model_meta.json")) as f:
    v1 = json.load(f)
with open(os.path.join(VAR2_DIR, "var2_model_meta.json")) as f:
    v2 = json.load(f)

# Load sweep and coefficient tables
df_v1_sweep = pd.read_csv(os.path.join(VAR1_DIR, "var1_degree_sweep.csv"))
df_v2_sweep = pd.read_csv(os.path.join(VAR2_DIR, "var2_degree_sweep.csv"))
# df_v1_sub table values are formatted directly in story


class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Arial", 8.5)
        self.setFillColor(colors.HexColor("#4A5568"))

        # Header (Pages 2+) - ONLY "Divyanshu Ghosh"
        if self._pageNumber > 1:
            self.drawString(36, 758, "Divyanshu Ghosh")
            self.setStrokeColor(colors.HexColor("#CBD5E0"))
            self.setLineWidth(0.5)
            self.line(36, 752, 576, 752)

        # Footer (All Pages) - "Machine Learning Assignment 1: Polynomial Regression & Regularization"
        self.setStrokeColor(colors.HexColor("#CBD5E0"))
        self.setLineWidth(0.5)
        self.line(36, 36, 576, 36)
        self.drawString(36, 25, "Machine Learning Assignment 1: Polynomial Regression & Regularization")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(576, 25, page_str)
        self.restoreState()


def build_pdf():
    doc = SimpleDocTemplate(
        PDF_PATH,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=44,
        bottomMargin=44
    )

    styles = getSampleStyleSheet()

    # Color Palette
    PRIMARY_NAVY = colors.HexColor("#1A365D")
    ACCENT_BLUE = colors.HexColor("#2B6CB0")
    TEXT_DARK = colors.HexColor("#1A202C")
    TEXT_BODY = colors.HexColor("#2D3748")
    BG_LIGHT = colors.HexColor("#F7FAFC")
    BG_HEADER = colors.HexColor("#2B6CB0")
    BORDER_COLOR = colors.HexColor("#CBD5E0")

    # Typography styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Arial-Bold",
        fontSize=15,
        leading=18,
        textColor=PRIMARY_NAVY,
        alignment=TA_CENTER
    )
    meta_style = ParagraphStyle(
        "DocMeta",
        parent=styles["Normal"],
        fontName="Arial",
        fontSize=9.5,
        leading=13,
        textColor=TEXT_DARK,
        alignment=TA_CENTER
    )
    h1_style = ParagraphStyle(
        "SecH1",
        parent=styles["Normal"],
        fontName="Arial-Bold",
        fontSize=10.5,
        leading=13.5,
        textColor=PRIMARY_NAVY,
        spaceBefore=6,
        spaceAfter=3,
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        "SecH2",
        parent=styles["Normal"],
        fontName="Arial-Bold",
        fontSize=9.2,
        leading=12,
        textColor=ACCENT_BLUE,
        spaceBefore=4,
        spaceAfter=2,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        "SecBody",
        parent=styles["Normal"],
        fontName="Arial",
        fontSize=8.2,
        leading=11.2,
        textColor=TEXT_BODY,
        alignment=TA_JUSTIFY,
        spaceAfter=3
    )
    eq_style = ParagraphStyle(
        "MathEq",
        parent=styles["Normal"],
        fontName="Arial-Bold",
        fontSize=8.5,
        leading=11.5,
        textColor=PRIMARY_NAVY,
        alignment=TA_CENTER,
        spaceBefore=2,
        spaceAfter=2
    )
    caption_style = ParagraphStyle(
        "FigCap",
        parent=styles["Normal"],
        fontName="Arial-Italic",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#4A5568"),
        alignment=TA_CENTER,
        spaceBefore=1,
        spaceAfter=3
    )
    th_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Arial-Bold",
        fontSize=7.8,
        leading=9.5,
        textColor=colors.white,
        alignment=TA_CENTER
    )
    td_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Arial",
        fontSize=7.5,
        leading=9.2,
        textColor=TEXT_DARK,
        alignment=TA_CENTER
    )
    td_bold = ParagraphStyle(
        "TableCellBold",
        parent=styles["Normal"],
        fontName="Arial-Bold",
        fontSize=7.5,
        leading=9.2,
        textColor=PRIMARY_NAVY,
        alignment=TA_CENTER
    )

    story = []

    # =========================================================================
    # PAGE 1: TITLE, INTRO, FORMULATION, VALIDATION, VAR1 INTRO & SUBSETS
    # =========================================================================
    story.append(Paragraph("Assignment-1: Polynomial Regression and Regularization", title_style))
    story.append(Spacer(1, 3))
    story.append(Paragraph("Course Name: Machine Learning &nbsp;&nbsp;|&nbsp;&nbsp; Name: Divyanshu Ghosh &nbsp;&nbsp;|&nbsp;&nbsp; Roll Number: IMT2024068", meta_style))
    story.append(Spacer(1, 2))
    story.append(Paragraph("GitHub Repository: <font color='#2B6CB0'><u>https://github.com/77-Div-77/ML_Assignment_1</u></font>", meta_style))
    story.append(Spacer(1, 3))
    story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY_NAVY, spaceBefore=2, spaceAfter=5))

    # Section 1
    story.append(Paragraph("1. Introduction and Problem Formulation", h1_style))
    story.append(Paragraph(
        "This project presents a rigorous investigation of polynomial regression, bias-variance tradeoff, and regularization "
        "applied to two continuous regression benchmarks: <b>Dataset Var1</b> (6 input features, 1,000 training instances) and "
        "<b>Dataset Var2</b> (3 input features, 1,000 training instances). Both datasets require predicting an unlabelled test set of 1,000 instances. "
        "When expanding input features into higher-degree polynomial feature spaces, Ordinary Least Squares (OLS) estimation often suffers from "
        "numerical ill-conditioning and severe overfitting due to multicollinearity. Regularized estimation is employed to control model complexity.",
        body_style
    ))

    # Mathematical Equations
    story.append(Paragraph("<b>Mathematical Formulation:</b>", h2_style))
    eq_box_data = [
        [
            Paragraph("<b>Polynomial Model Form:</b>", td_bold),
            Paragraph("y&#770; = w<sub>0</sub> + &sum;<sub>j=1</sub><sup>M</sup> w<sub>j</sub> &phi;<sub>j</sub>(x)", eq_style),
            Paragraph("where &phi;<sub>j</sub>(x) represents monomial basis terms up to degree d.", body_style)
        ],
        [
            Paragraph("<b>Evaluation Metrics:</b>", td_bold),
            Paragraph("MSE = (1/N) &sum;<sub>i=1</sub><sup>N</sup> (y<sub>i</sub> - y&#770;<sub>i</sub>)<sup>2</sup> &nbsp;&nbsp;|&nbsp;&nbsp; RMSE = &radic;MSE &nbsp;&nbsp;|&nbsp;&nbsp; R<sup>2</sup> = 1 - &sum;(y<sub>i</sub> - y&#770;<sub>i</sub>)<sup>2</sup> / &sum;(y<sub>i</sub> - y&#772;)<sup>2</sup>", eq_style),
            Paragraph("Measuring residual dispersion and explained target variance.", body_style)
        ],
        [
            Paragraph("<b>Regularized Objective:</b>", td_bold),
            Paragraph("J(w) = MSE + &lambda;<sub>1</sub> ||w||<sub>1</sub> + &lambda;<sub>2</sub> ||w||<sub>2</sub><sup>2</sup>", eq_style),
            Paragraph("where &lambda;<sub>1</sub> controls L1 (LASSO) sparsity and &lambda;<sub>2</sub> controls L2 (Ridge) shrinkage.", body_style)
        ]
    ]
    t_eq = Table(eq_box_data, colWidths=[105, 230, 205])
    t_eq.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BG_LIGHT),
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t_eq)
    story.append(Spacer(1, 4))

    story.append(Paragraph("<b>Cross-Validation Protocol:</b>", h2_style))
    story.append(Paragraph(
        "To guarantee unbiased evaluation and prevent data leakage, 10-fold cross-validation is used throughout all experiments. "
        "All feature scaling (z-score standardization) is fit strictly on training folds and applied to validation folds. "
        "Optimal polynomial degrees and regularization hyperparameters (&lambda;<sub>1</sub>, &lambda;<sub>2</sub>) are selected "
        "based strictly on minimizing cross-validation Root Mean Squared Error (CV RMSE).",
        body_style
    ))

    # Section 2: Var1 Analysis
    story.append(Paragraph("2. Dataset Var1 Analysis & Feature Subsets", h1_style))
    story.append(Paragraph(
        "Dataset Var1 contains 6 features (x1 to x6). To isolate feature relevance before sweeping degrees, subsets were screened at Degree 3: "
        "unlike linear models, Degree 3 captures non-linear curvatures and 3-way interactions, while having only 83 terms (well below N=1,000) "
        "so OLS remains numerically stable across all folds without confounding feature importance with ill-conditioning. "
        "All 2<sup>6</sup> - 1 = 63 combinations were evaluated under 10-fold CV. As shown in Table 1, retaining all 6 features achieves the "
        "lowest error (CV RMSE = 0.9892), whereas removing features increases error substantially.",
        body_style
    ))

    # Subsets table
    sub_table_data = [
        [Paragraph("Feature Subset", th_style), Paragraph("Num Features", th_style), Paragraph("Degree 3 CV RMSE", th_style), Paragraph("Rank / Assessment", th_style)],
        [Paragraph("x1 + x2 + x3 + x4 + x5 + x6", td_bold), Paragraph("6", td_style), Paragraph("0.9892", td_bold), Paragraph("Rank 1 (Optimal Full Set)", td_style)],
        [Paragraph("x1 + x2 + x3 + x5 + x6", td_style), Paragraph("5", td_style), Paragraph("1.3083", td_style), Paragraph("Rank 2 (+32.3% error penalty)", td_style)],
        [Paragraph("x1 + x3 + x4 + x5 + x6", td_style), Paragraph("5", td_style), Paragraph("1.5906", td_style), Paragraph("Rank 3 (+60.8% error penalty)", td_style)],
        [Paragraph("x2 + x3 + x4 + x5 + x6", td_style), Paragraph("5", td_style), Paragraph("1.7614", td_style), Paragraph("Rank 4 (+78.1% error penalty)", td_style)],
        [Paragraph("x1 + x2 + x3 (First 3 features)", td_style), Paragraph("3", td_style), Paragraph("2.0224", td_style), Paragraph("Severe Underfitting (+104.5%)", td_style)],
    ]
    t_sub = Table(sub_table_data, colWidths=[180, 80, 110, 170])
    t_sub.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BG_HEADER),
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BG_LIGHT]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    story.append(t_sub)
    story.append(Paragraph("Table 1: Top feature subsets for Dataset Var1 evaluated at degree 3 under 10-fold cross-validation.", caption_style))

    # Page Break to Page 2
    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: VAR1 DEGREE SWEEP, REGULARIZATION, AND FIGURES
    # =========================================================================
    story.append(Paragraph("3. Dataset Var1 Model Selection & Evaluation", h1_style))
    story.append(Paragraph(
        "Using all 6 features, a joint degree and regularization sweep was executed across degrees 1 to 8. "
        "Ordinary Least Squares (OLS), Ridge regression, and LASSO were evaluated across all degrees. "
        "As summarized in Table 2, OLS produces valid numerical solutions up to degree 5 (CV RMSE = 1.1019 at degree 5), "
        "before suffering from ill-conditioning and severe collinearity at degrees 6 to 8. "
        "Ridge regression steadily reduces validation error to 0.7066 at degree 5. LASSO regression achieves superior performance, "
        "reaching the global minimum CV RMSE of <b>0.5581</b> (out-of-fold RMSE = 0.5773) at <b>Degree 5</b> with &lambda;<sub>1</sub> = 0.032.",
        body_style
    ))

    # Var1 Sweep Table
    v1_rows = [
        [Paragraph("Degree", th_style), Paragraph("Non-bias Terms", th_style), Paragraph("OLS CV RMSE", th_style),
         Paragraph("Ridge &lambda;<sub>2</sub>", th_style), Paragraph("Ridge CV RMSE", th_style),
         Paragraph("LASSO &lambda;<sub>1</sub>", th_style), Paragraph("LASSO CV RMSE", th_style)]
    ]
    for _, r in df_v1_sweep.iterrows():
        deg = int(r["degree"])
        is_best = (deg == 5)
        cell_style = td_bold if is_best else td_style
        ols_txt = f"{r['ols_cv_rmse']:.4f}" if r['ols_cv_rmse'] < 50.0 else "Ill-conditioned"
        v1_rows.append([
            Paragraph(f"Degree {deg}" + (" (Selected)" if is_best else ""), cell_style),
            Paragraph(str(int(r["non_bias_terms"])), cell_style),
            Paragraph(ols_txt, cell_style),
            Paragraph(f"{r['best_ridge_lambda2']}", cell_style),
            Paragraph(f"{r['ridge_cv_rmse']:.4f}", cell_style),
            Paragraph(f"{r['best_lasso_lambda1']}", cell_style),
            Paragraph(f"{r['lasso_cv_rmse']:.4f}", cell_style),
        ])

    t_v1 = Table(v1_rows, colWidths=[90, 75, 75, 70, 75, 75, 80])
    t_v1.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BG_HEADER),
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BG_LIGHT]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(t_v1)
    story.append(Paragraph("Table 2: Dataset Var1 10-fold cross-validation RMSE across polynomial degrees 1 to 8.", caption_style))
    story.append(Spacer(1, 2))

    # Var1 Figures: Row 1 (RMSE vs Degree & R2 Score vs Degree)
    p_v1_deg = os.path.join(VAR1_DIR, "degree_error_plot.png")
    p_v1_r2 = os.path.join(VAR1_DIR, "r2_score_plot.png")
    p_v1_heat = os.path.join(VAR1_DIR, "regularization_heatmap.png")
    p_v1_curve = os.path.join(VAR1_DIR, "train_vs_cv_error_plot.png")

    fig_row1 = [
        [Image(p_v1_deg, width=265, height=115), Image(p_v1_r2, width=265, height=115)],
        [Paragraph("Figure 1: Var1 CV RMSE vs polynomial degree (OLS deg 1-5, Ridge, LASSO).", caption_style),
         Paragraph("Figure 2: Var1 CV R<sup>2</sup> score vs polynomial degree (optimal at deg 5).", caption_style)]
    ]
    t_fig1 = Table(fig_row1, colWidths=[270, 270])
    t_fig1.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(t_fig1)
    story.append(Spacer(1, 2))

    # Var1 Figures: Row 2 (Regularization Heatmap & Training vs CV curve)
    fig_row2 = [
        [Image(p_v1_heat, width=265, height=115), Image(p_v1_curve, width=265, height=115)],
        [Paragraph("Figure 3: Var1 (Degree 5) validation error across (&lambda;<sub>1</sub>, &lambda;<sub>2</sub>).", caption_style),
         Paragraph("Figure 4: Var1 (Degree 5) Training vs CV RMSE across &lambda;<sub>1</sub>.", caption_style)]
    ]
    t_fig2 = Table(fig_row2, colWidths=[270, 270])
    t_fig2.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(t_fig2)
    story.append(Spacer(1, 2))

    story.append(Paragraph(
        "<b>Var1 Model Selection & Sparsity Summary:</b> Degree 5 with LASSO (&lambda;<sub>1</sub> = 0.032) achieves the optimal "
        "generalization balance. Out of 461 expanded polynomial features, LASSO sets 363 coefficients to exactly zero, retaining "
        "<b>98 active terms</b> (78.7% sparsity). This eliminates variance inflation while preserving non-linear fidelity.",
        body_style
    ))

    # Page Break to Page 3
    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: VAR2 DEGREE SWEEP, OLS ILL-CONDITIONING, OVERFITTING CHECK
    # =========================================================================
    story.append(Paragraph("4. Dataset Var2 Analysis & Degree Sweep (Degrees 1 to 15)", h1_style))
    story.append(Paragraph(
        "Dataset Var2 contains 3 input features (x1, x2, x3). Because the input dimension is moderate, high-degree expansions "
        "can be explored without catastrophic dimensional explosion. A complete sweep from <b>Degree 1 to Degree 15</b> was conducted "
        "to evaluate Ordinary Least Squares, Ridge, and LASSO across 10-fold cross-validation.",
        body_style
    ))

    # Var2 Table
    v2_rows = [
        [Paragraph("Degree", th_style), Paragraph("Terms", th_style), Paragraph("OLS CV RMSE", th_style),
         Paragraph("Cond. No.", th_style), Paragraph("Ridge &lambda;<sub>2</sub>", th_style),
         Paragraph("Ridge RMSE", th_style), Paragraph("LASSO &lambda;<sub>1</sub>", th_style),
         Paragraph("LASSO RMSE", th_style)]
    ]
    for _, r in df_v2_sweep.iterrows():
        deg = int(r["degree"])
        is_best = (deg == 14)
        cell_style = td_bold if is_best else td_style
        ols_val = r["ols_cv_rmse"]
        ols_txt = f"{ols_val:.4f}" if ols_val < 100.0 else f"{ols_val:.1f}"
        v2_rows.append([
            Paragraph(f"Deg {deg}" + ("*" if is_best else ""), cell_style),
            Paragraph(str(int(r["non_bias_terms"])), cell_style),
            Paragraph(ols_txt, cell_style),
            Paragraph(str(r["ols_cond_number"]), cell_style),
            Paragraph(f"{r['best_ridge_lambda2']}", cell_style),
            Paragraph(f"{r['ridge_cv_rmse']:.4f}", cell_style),
            Paragraph(f"{r['best_lasso_lambda1']}", cell_style),
            Paragraph(f"{r['lasso_cv_rmse']:.4f}", cell_style),
        ])

    t_v2 = Table(v2_rows, colWidths=[55, 45, 75, 75, 70, 70, 75, 75])
    t_v2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BG_HEADER),
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BG_LIGHT]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
    ]))
    story.append(t_v2)
    story.append(Paragraph("Table 3: Dataset Var2 cross-validation performance, condition number, and regularization metrics across degrees 1 to 15 (* indicates final selected model).", caption_style))
    story.append(Spacer(1, 3))

    # Overfitting analysis and OLS Ill-conditioning
    story.append(Paragraph("<b>Investigation of OLS Ill-Conditioning & High-Degree Overfitting:</b>", h2_style))
    story.append(Paragraph(
        "<b>1. OLS Numerical Behavior:</b> Valid numerical CV RMSE values were computed for OLS across all 15 degrees. "
        "As seen in Table 3, OLS improves from degree 1 (5.6387) to degree 8 (0.5081). Beyond degree 8, the condition number of the feature matrix "
        "grows exponentially from 1.31&times;10<sup>3</sup> to 9.15&times;10<sup>6</sup>. Unregularized OLS begins overfitting severely: "
        "CV RMSE degrades to 0.6279 at degree 11, 4.5226 at degree 14, and explodes to <b>34.5389</b> at degree 15. "
        "This empirical divergence proves that high-degree polynomial regression cannot be solved using unregularized least squares.<br/>"
        "<b>2. Why Degree 14 LASSO Does Not Overfit:</b> Among all tested configurations, <b>Degree 14 with LASSO</b> achieves the global minimum "
        "cross-validation RMSE of <b>0.4776</b> (&lambda;<sub>1</sub> = 0.002). We specifically verified whether Degree 14 overfits: "
        "The training RMSE is 0.4010, resulting in a minimal generalization gap of only 0.0766 between train and CV error. "
        "L1 regularization provides selective sparsity, shrinking 381 redundant higher-order monomial terms to exactly zero and retaining "
        "only 298 active features. At Degree 15, validation RMSE begins to increase (0.4780), confirming that Degree 14 represents the optimal "
        "bias-variance sweet spot. Selecting Degree 14 based on minimum CV RMSE is completely consistent with course methodology.",
        body_style
    ))

    # Page Break to Page 4
    story.append(PageBreak())

    # =========================================================================
    # PAGE 4: VAR2 VISUALIZATIONS & REGULARIZATION LANDSCAPE
    # =========================================================================
    story.append(Paragraph("5. Dataset Var2 Visualizations & Regularization Tuning", h1_style))
    story.append(Paragraph(
        "To illustrate the bias-variance transition and regularization dynamics for Dataset Var2, Figure 5 displays cross-validation error "
        "across all 15 polynomial degrees, while Figure 6 displays the corresponding cross-validation R<sup>2</sup> trajectory. "
        "Figure 7 illustrates the 2D regularization surface across (&lambda;<sub>1</sub>, &lambda;<sub>2</sub>) at Degree 14, and "
        "Figure 8 depicts the training vs. validation error curve across &lambda;<sub>1</sub>.",
        body_style
    ))

    p_v2_deg = os.path.join(VAR2_DIR, "degree_error_plot.png")
    p_v2_r2 = os.path.join(VAR2_DIR, "r2_score_plot.png")
    p_v2_heat = os.path.join(VAR2_DIR, "regularization_heatmap.png")
    p_v2_curve = os.path.join(VAR2_DIR, "train_vs_cv_error_plot.png")

    fig_row3 = [
        [Image(p_v2_deg, width=265, height=115), Image(p_v2_r2, width=265, height=115)],
        [Paragraph("Figure 5: Var2 CV RMSE vs polynomial degree showing OLS (deg &le; 14), Ridge, and LASSO.", caption_style),
         Paragraph("Figure 6: Var2 CV R<sup>2</sup> score vs polynomial degree (global maximum at deg 14).", caption_style)]
    ]
    t_fig3 = Table(fig_row3, colWidths=[270, 270])
    t_fig3.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(t_fig3)
    story.append(Spacer(1, 2))

    fig_row4 = [
        [Image(p_v2_heat, width=265, height=115), Image(p_v2_curve, width=265, height=115)],
        [Paragraph("Figure 7: Var2 (Degree 14) validation error heatmap across (&lambda;<sub>1</sub>, &lambda;<sub>2</sub>).", caption_style),
         Paragraph("Figure 8: Var2 (Degree 14) Training vs Validation RMSE across &lambda;<sub>1</sub>.", caption_style)]
    ]
    t_fig4 = Table(fig_row4, colWidths=[270, 270])
    t_fig4.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(t_fig4)
    story.append(Spacer(1, 2))

    story.append(Paragraph(
        "<b>Var2 Regularization Analysis & 5-Fold Sanity Check:</b><br/>"
        "&bull; <b>Under-regularization (&lambda;<sub>1</sub> &le; 10<sup>-4</sup>):</b> CV RMSE rises toward 0.54 due to excess variance and high condition numbers.<br/>"
        "&bull; <b>Optimal Regularization (&lambda;<sub>1</sub> = 0.002):</b> CV RMSE achieves its global minimum of <b>0.4776</b> (CV R<sup>2</sup> = 0.9949).<br/>"
        "&bull; <b>Over-regularization (&lambda;<sub>1</sub> &ge; 0.01):</b> CV RMSE rises sharply above 0.70+ as essential non-linear terms are eliminated.<br/>"
        "An independent 5-fold cross-validation check produced an RMSE of <b>0.4834</b>, verifying that the selected model's predictive performance is stable.",
        body_style
    ))

    # Page Break to Page 5
    story.append(PageBreak())

    # =========================================================================
    # PAGE 5: FINAL COMPARISON, TEST PREDICTIONS, VERIFICATION & CONCLUSION
    # =========================================================================
    story.append(Paragraph("6. Final Models Comparison and Summary Metrics", h1_style))
    story.append(Paragraph(
        "Table 4 provides a comprehensive comparison of the final selected models for Dataset Var1 and Dataset Var2. "
        "All reported metrics are exact values obtained from full training fits and 10-fold cross-validation evaluations.",
        body_style
    ))

    comp_rows = [
        [Paragraph("Metric / Property", th_style), Paragraph("Dataset Var1 (Final Model)", th_style), Paragraph("Dataset Var2 (Final Model)", th_style)],
        [Paragraph("Input Features", td_bold), Paragraph("x1, x2, x3, x4, x5, x6 (All 6)", td_style), Paragraph("x1, x2, x3 (All 3)", td_style)],
        [Paragraph("Selected Polynomial Degree", td_bold), Paragraph("<b>Degree 5</b>", td_style), Paragraph("<b>Degree 14</b>", td_style)],
        [Paragraph("Model Family", td_bold), Paragraph("LASSO (L1 Regularized)", td_style), Paragraph("LASSO (L1 Regularized)", td_style)],
        [Paragraph("Regularization Parameters", td_bold), Paragraph("&lambda;<sub>1</sub> = 0.032, &lambda;<sub>2</sub> = 0.0", td_style), Paragraph("&lambda;<sub>1</sub> = 0.002, &lambda;<sub>2</sub> = 0.0", td_style)],
        [Paragraph("scikit-learn alpha", td_bold), Paragraph("0.016", td_style), Paragraph("0.001", td_style)],
        [Paragraph("Active Terms / Total Non-bias", td_bold), Paragraph(f"{v1['active_terms']} / {v1['non_bias_terms']} (78.7% sparse)", td_style), Paragraph(f"{v2['active_terms']} / {v2['non_bias_terms']} (56.1% sparse)", td_style)],
        [Paragraph("Training MSE", td_bold), Paragraph(f"{v1['train_mse']:.4f}", td_style), Paragraph(f"{v2['train_mse']:.4f}", td_style)],
        [Paragraph("Training RMSE", td_bold), Paragraph(f"{v1['train_rmse']:.4f}", td_style), Paragraph(f"{v2['train_rmse']:.4f}", td_style)],
        [Paragraph("Training R<sup>2</sup>", td_bold), Paragraph(f"{v1['train_r2']:.4f}", td_style), Paragraph(f"{v2['train_r2']:.4f}", td_style)],
        [Paragraph("10-Fold CV MSE", td_bold), Paragraph(f"{v1['oof_mse']:.4f}", td_style), Paragraph(f"{v2['oof_mse']:.4f}", td_style)],
        [Paragraph("10-Fold CV RMSE", td_bold), Paragraph(f"<b>{v1['oof_rmse']:.4f} &plusmn; {v1['oof_se']:.4f}</b>", td_bold), Paragraph(f"<b>{v2['oof_rmse']:.4f} &plusmn; {v2['oof_se']:.4f}</b>", td_bold)],
        [Paragraph("10-Fold CV R<sup>2</sup>", td_bold), Paragraph(f"{v1['oof_r2']:.4f}", td_style), Paragraph(f"{v2['oof_r2']:.4f}", td_style)],
        [Paragraph("5-Fold CV Check RMSE", td_bold), Paragraph(f"{v1['cv5_rmse']:.4f}", td_style), Paragraph(f"{v2['cv5_rmse']:.4f}", td_style)],
    ]

    t_comp = Table(comp_rows, colWidths=[160, 190, 190])
    t_comp.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BG_HEADER),
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BG_LIGHT]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(t_comp)
    story.append(Paragraph("Table 4: Comprehensive comparison of final selected models for Dataset Var1 and Dataset Var2.", caption_style))
    story.append(Spacer(1, 4))

    # Section 7: Test Prediction Details
    story.append(Paragraph("7. Test Prediction Details", h1_style))
    story.append(Paragraph(
        "Predictions on the unlabelled test sets were generated by retraining the final selected pipelines on the entire 1,000 training instances "
        "and evaluating on the test feature matrices. Both prediction files conform strictly to the required specifications:",
        body_style
    ))

    verif_rows = [
        [Paragraph("Property / Verification Item", th_style), Paragraph("Dataset Var1 File", th_style), Paragraph("Dataset Var2 File", th_style)],
        [Paragraph("File Name", td_bold), Paragraph("IMT2024068_pred_var1.csv", td_style), Paragraph("IMT2024068_pred_var2.csv", td_style)],
        [Paragraph("Row Count", td_bold), Paragraph("1,000 rows (exact test size)", td_style), Paragraph("1,000 rows (exact test size)", td_style)],
        [Paragraph("Header Specification", td_bold), Paragraph("Single column 'y'", td_style), Paragraph("Single column 'y'", td_style)],
        [Paragraph("Missing / NaN Values", td_bold), Paragraph("0 NaNs (100% complete)", td_style), Paragraph("0 NaNs (100% complete)", td_style)],
        [Paragraph("Prediction Range", td_bold), Paragraph("[-10.03, 14.88]", td_style), Paragraph("[-29.89, 39.18]", td_style)],
        [Paragraph("Mean &plusmn; Std Prediction", td_bold), Paragraph("0.9824 &plusmn; 4.1472", td_style), Paragraph("2.0706 &plusmn; 6.3833", td_style)],
    ]
    t_ver = Table(verif_rows, colWidths=[170, 185, 185])
    t_ver.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BG_HEADER),
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BG_LIGHT]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(t_ver)
    story.append(Paragraph("Table 5: Details of submission prediction files for Dataset Var1 and Dataset Var2.", caption_style))
    story.append(Spacer(1, 4))

    # Section 8: Conclusion
    story.append(Paragraph("8. Conclusion", h1_style))
    story.append(Paragraph(
        "This assignment demonstrated the fundamental importance of regularization and systematic cross-validation in high-dimensional polynomial regression. "
        "For Dataset Var1, retaining all 6 input features and expanding to Degree 5 with LASSO (&lambda;<sub>1</sub> = 0.032) yielded an optimal CV RMSE of 0.5581, "
        "eliminating 78.7% of redundant polynomial cross-terms. For Dataset Var2, expanding to Degree 14 with LASSO (&lambda;<sub>1</sub> = 0.002) achieved the global "
        "minimum CV RMSE of 0.4776, successfully taming severe matrix ill-conditioning (condition number 1.29&times;10<sup>6</sup>) and preventing overfitting.",
        body_style
    ))

    # Build PDF
    doc.build(story, canvasmaker=NumberedCanvas)
    print("Report built successfully at:", PDF_PATH)


if __name__ == "__main__":
    build_pdf()
