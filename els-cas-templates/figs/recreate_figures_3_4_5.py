#!/usr/bin/env python3
"""Recreate manuscript Figures 3, 4, and 5 as editable vector PDFs.

Install the only plotting dependency with: python -m pip install reportlab
Run from any directory: python recreate_figures_3_4_5.py

The adjacent reconstructed_figure_data.json contains the drawing data. Figure 3
uses the existing SVG's exact cell colors and geometry, not recovered numerical
measurements. Figure 4's values were read from the PNGs and rounded to 0.001;
they are reconstruction estimates, not newly measured experimental results.
Figure 5 reproduces the original illustrative topology. No PNG is embedded in
the exported PDFs, and neither the original images nor the LaTeX are changed.
The default paper style enlarges text and lines for the manuscript's actual
figure widths. Use --style original to reproduce the initial pixel layout.
"""

import argparse
import json
import math
from pathlib import Path

from reportlab import rl_config
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


HERE = Path(__file__).resolve().parent
FONT_NAME = "RecreatedFigureSans"


def register_font(font_path=None):
    """Use the original Arial face when available, with portable fallbacks."""
    candidates = [Path(font_path)] if font_path else [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    candidates.extend(Path(directory) / "Vera.ttf" for directory in rl_config.TTFSearchPath)
    for candidate in candidates:
        if candidate.is_file():
            pdfmetrics.registerFont(TTFont(FONT_NAME, str(candidate)))
            return
    raise FileNotFoundError("No usable TrueType font found. Supply --font /path/to/font.ttf.")


class VectorFigure:
    """Draw in the original image's pixel coordinate system, with a top origin."""

    def __init__(self, path, width, height, page_width, title, subject):
        self.width = width
        self.height = height
        scale = page_width / width
        self.canvas = canvas.Canvas(
            str(path), pagesize=(page_width, height * scale), pageCompression=1,
        )
        self.canvas.setTitle(title)
        self.canvas.setSubject(subject)
        self.canvas.setAuthor("RPCD-GNN figure reconstruction")
        self.canvas.scale(scale, scale)
        self.rect(0, 0, width, height, "#ffffff")

    def rect(self, x, y, width, height, fill, stroke=None, line_width=1):
        c = self.canvas
        c.saveState()
        c.setFillColor(HexColor(fill))
        if stroke:
            c.setStrokeColor(HexColor(stroke))
            c.setLineWidth(line_width)
        c.rect(x, self.height - y - height, width, height, fill=1, stroke=int(bool(stroke)))
        c.restoreState()

    def line(self, points, color, width=2, dash=None):
        c = self.canvas
        c.saveState()
        c.setStrokeColor(HexColor(color))
        c.setLineWidth(width)
        c.setLineJoin(1)
        if dash:
            c.setDash(dash)
        path = c.beginPath()
        path.moveTo(points[0][0], self.height - points[0][1])
        for x, y in points[1:]:
            path.lineTo(x, self.height - y)
        c.drawPath(path)
        c.restoreState()

    def polygon(self, points, fill, stroke=None, width=1):
        c = self.canvas
        c.saveState()
        c.setFillColor(HexColor(fill))
        if stroke:
            c.setStrokeColor(HexColor(stroke))
            c.setLineWidth(width)
        path = c.beginPath()
        path.moveTo(points[0][0], self.height - points[0][1])
        for x, y in points[1:]:
            path.lineTo(x, self.height - y)
        path.close()
        c.drawPath(path, fill=1, stroke=int(bool(stroke)))
        c.restoreState()

    def ellipse(self, x, y, radius_x, radius_y, fill, stroke, width=1, dash=None):
        c = self.canvas
        c.saveState()
        c.setFillColor(HexColor(fill))
        c.setStrokeColor(HexColor(stroke))
        c.setLineWidth(width)
        if dash:
            c.setDash(dash)
        c.ellipse(
            x - radius_x, self.height - y - radius_y,
            x + radius_x, self.height - y + radius_y, fill=1, stroke=1,
        )
        c.restoreState()

    def text(self, x, y, value, size, color="#222222", anchor="middle", rotation=0):
        """Place text at a baseline, following SVG text-anchor conventions."""
        c = self.canvas
        c.saveState()
        c.translate(x, self.height - y)
        c.rotate(-rotation)
        c.setFont(FONT_NAME, size)
        c.setFillColor(HexColor(color))
        value = str(value)
        if anchor == "end":
            c.drawRightString(0, 0, value)
        elif anchor == "middle":
            c.drawCentredString(0, 0, value)
        else:
            c.drawString(0, 0, value)
        c.restoreState()

    def centered_text(self, x, y, lines, size, color):
        leading = size * 1.1
        ascent = pdfmetrics.getAscent(FONT_NAME) * size / 1000
        descent = pdfmetrics.getDescent(FONT_NAME) * size / 1000
        first_baseline = y - (len(lines) - 1) * leading / 2 + (ascent + descent) / 2
        for index, value in enumerate(lines):
            self.text(x, first_baseline + index * leading, value, size, color)

    def save(self):
        self.canvas.showPage()
        self.canvas.save()


def draw_heatmap(data, output_dir, page_width, style="paper"):
    """Rebuild all four panels from the source SVG's displayed colors."""
    destination = output_dir / "relation-propagation-heatmap.pdf"
    paper_style = style == "paper"
    width, height = (1510, 1260) if paper_style else (data["width"], data["height"])
    tick_size, bar_size, caption_size = (21, 20, 26) if paper_style else (11, 10, 15)
    fig = VectorFigure(
        destination, width, height, page_width,
        "Figure 3: relation-propagation heatmaps",
        "Vector reconstruction of the existing SVG colors; layout style: " + style + ".",
    )
    cell_width, cell_height = data["cell_size"]
    for panel in data["panels"]:
        left, top = panel["origin"]
        if paper_style:
            # Remove unused horizontal space while allowing larger relation labels.
            left += 65 if left == 165 else -60
        for row_index, colors in enumerate(panel["colors"]):
            for column_index, color in enumerate(colors):
                fig.rect(
                    left + column_index * cell_width,
                    top + row_index * cell_height,
                    cell_width, cell_height, color, "#ffffff", 1,
                )
        for index, label in enumerate(panel["labels"]):
            row_baseline = 24 if paper_style else 21
            fig.text(left - 9, top + index * cell_height + row_baseline, label, tick_size, "#344650", "end")
            fig.text(
                left + (index + 0.5) * cell_width, top + 359,
                label, tick_size, "#344650", "end", -55,
            )
        for stripe in data["colorbar"]:
            fig.rect(
                left + stripe["x"], top + stripe["y"],
                stripe["width"], stripe["height"], stripe["fill"],
            )
        fig.text(left + 470, top + (17 if paper_style else 10), "0.69", bar_size, "#344650", "start")
        fig.text(left + 470, top + 340, "0", bar_size, "#344650", "start")
        caption_offset = (545 if top == 70 else 570) if paper_style else 495
        fig.text(left + 210, top + caption_offset, panel["caption"], caption_size, "#344650")
    fig.save()
    return destination


def draw_marker(fig, x, y, marker, color, radius=10, line_width=4.8):
    if marker == "circle":
        fig.ellipse(x, y, radius, radius, "#ffffff", color, line_width)
    elif marker == "triangle":
        extent = radius * 1.2
        fig.polygon([(x, y - extent), (x - extent, y + extent), (x + extent, y + extent)], "#ffffff", color, line_width)
    else:
        raise ValueError("Unknown marker: " + marker)


def draw_curve(data, output_dir, page_width, style="paper"):
    destination = output_dir / (data["name"] + ".pdf")
    fig = VectorFigure(
        destination, 1700, 1120, page_width,
        "Figure 4: " + data["name"],
        "Vector reconstruction; plotted values read from the original PNG to 0.001 precision.",
    )
    paper_style = style == "paper"
    left, top, right, bottom = (260 if paper_style else 200), 164, 1560, 910
    tick_size, axis_size, legend_size = (50, 58, 52) if paper_style else (22, 26, 24)
    line_width = 9 if paper_style else 5.8
    marker_radius, marker_width = (16, 6.5) if paper_style else (10, 4.8)
    low, high = data["y_limits"]

    def y_position(value):
        return bottom - (value - low) / (high - low) * (bottom - top)

    # The source eta panels use equally spaced categories. Preserve that layout.
    x_positions = [left + index * (right - left) / (len(data["x_labels"]) - 1)
                   for index in range(len(data["x_labels"]))]
    for tick in data["y_ticks"]:
        y = y_position(tick)
        fig.line([(left, y), (right, y)], "#c7c7c7" if paper_style else "#d9d9d9", 2.5 if paper_style else 2, [8, 8])
        fig.text(left - (32 if paper_style else 24), y + tick_size * 0.34, data["y_format"].format(tick), tick_size, anchor="end")
    fig.line([(left, top), (left, bottom), (right, bottom)], "#222222", 3.5 if paper_style else 2.4)
    for x, label in zip(x_positions, data["x_labels"]):
        fig.line([(x, bottom), (x, bottom + 12)], "#222222", 2)
        fig.text(x, bottom + (62 if paper_style else 48), label, tick_size)
    fig.text((left + right) / 2, bottom + (130 if paper_style else 84), data["x_axis"], axis_size)
    fig.text(54 if paper_style else 76, (top + bottom) / 2 + 8, data["y_axis"], axis_size, rotation=-90)
    for series_index, series in enumerate(data["series"]):
        points = [(x, y_position(value)) for x, value in zip(x_positions, series["values"])]
        fig.line(points, series["color"], line_width)
        for x, y in points:
            draw_marker(fig, x, y, series["marker"], series["color"], marker_radius, marker_width)
        legend_left = [600, 990][series_index]
        fig.line([(legend_left, 68), (legend_left + 52, 68)], series["color"], line_width)
        draw_marker(fig, legend_left + 26, 68, series["marker"], series["color"], marker_radius, marker_width)
        fig.text(legend_left + 70, 84 if paper_style else 76, series["label"], legend_size, anchor="start")
    fig.save()
    return destination


def ellipse_boundary(node, target):
    dx, dy = target[0] - node["center"][0], target[1] - node["center"][1]
    radius_x, radius_y = node["radii"]
    ratio = 1 / math.sqrt((dx / radius_x) ** 2 + (dy / radius_y) ** 2)
    return node["center"][0] + ratio * dx, node["center"][1] + ratio * dy


def draw_arrow(fig, start, end, color, width, head_length, head_width, dash=None):
    dx, dy = end[0] - start[0], end[1] - start[1]
    distance = math.hypot(dx, dy)
    ux, uy = dx / distance, dy / distance
    base = (end[0] - head_length * ux, end[1] - head_length * uy)
    fig.line([start, base], color, width, dash)
    fig.polygon([
        end,
        (base[0] - uy * head_width / 2, base[1] + ux * head_width / 2),
        (base[0] + uy * head_width / 2, base[1] - ux * head_width / 2),
    ], color)


def draw_case_study(data, output_dir, page_width, style="paper"):
    destination = output_dir / "reasoning-path-case-study.pdf"
    fig = VectorFigure(
        destination, data["width"], data["height"], page_width,
        "Figure 5: illustrative UMLS reasoning paths",
        "Vector reconstruction of the original schematic; no learned attention weights.",
    )
    paper_style = style == "paper"
    nodes = {node["id"]: node.copy() for node in data["nodes"]}
    styles = {key: value.copy() for key, value in data["edge_styles"].items()}
    edges = [edge.copy() for edge in data["edges"]]
    node_size, edge_size = (64, 58) if paper_style else (36, 30)
    if paper_style:
        # Fit enlarged labels inside the ellipses, keeping the original topology.
        radii = {
            "query": [270, 110], "receptor": [210, 80], "cell": [225, 145],
            "biologic": [220, 112], "target": [260, 110],
            "diagnostic": [225, 112], "laboratory": [260, 112],
            "tissue": [130, 76], "chemical": [175, 80],
            "pathologic": [230, 112], "experimental": [300, 112],
            "clinical": [180, 112],
        }
        for identifier, node in nodes.items():
            node["radii"] = radii[identifier]
        nodes["query"]["center"] = [300, 1436]
        nodes["experimental"]["center"] = [3949.5, 586]
        nodes["clinical"]["center"] = [2611.5, 2390]
        for key, stroke_width in [("main", 14), ("auxiliary", 12), ("context", 5)]:
            styles[key]["width"] = stroke_width
            styles[key]["head_length"] *= 1.25
            styles[key]["head_width"] *= 1.25
        for edge in edges:
            if edge["label"] == "Ingredient of":
                edge["label_position"] = [635, 1580]
    for edge in edges:
        source, target = nodes[edge["source"]], nodes[edge["target"]]
        start = ellipse_boundary(source, target["center"])
        end = ellipse_boundary(target, source["center"])
        style = styles[edge["style"]]
        draw_arrow(
            fig, start, end, style["color"], style["width"],
            style["head_length"], style["head_width"], style.get("dash"),
        )
    for node in nodes.values():
        fig.ellipse(
            *node["center"], *node["radii"], node["fill"], node["stroke"],
            node["stroke_width"], node.get("dash"),
        )
        fig.centered_text(*node["center"], node["label"], node_size, node["text_color"])
    for edge in edges:
        x, y = edge["label_position"]
        style = styles[edge["style"]]
        # A small white halo makes labels readable where the original line crosses.
        text_width = pdfmetrics.stringWidth(edge["label"], FONT_NAME, edge_size)
        halo_top, halo_height = (edge_size * 0.9, edge_size * 1.15) if paper_style else (28, 34)
        fig.rect(x - text_width / 2 - 6, y - halo_top, text_width + 12, halo_height, "#ffffff")
        fig.text(x, y, edge["label"], edge_size, style["label_color"])
    fig.save()
    return destination


def validate_data(data):
    for panel in data["heatmap"]["panels"]:
        if len(panel["colors"]) != 10 or any(len(row) != 10 for row in panel["colors"]):
            raise ValueError("Every heatmap panel must contain a 10 x 10 color matrix.")
    for curve in data["curves"]:
        for series in curve["series"]:
            if len(series["values"]) != len(curve["x_labels"]):
                raise ValueError("Curve values and x labels must have the same length.")
    node_ids = {node["id"] for node in data["case_study"]["nodes"]}
    for edge in data["case_study"]["edges"]:
        if edge["source"] not in node_ids or edge["target"] not in node_ids:
            raise ValueError("Case-study edge references an unknown node.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE / "reconstructed_figure_data.json")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--figures", nargs="+", type=int, choices=[3, 4, 5], default=[3, 4, 5])
    parser.add_argument("--page-width", type=float, default=720, help="PDF page width in points.")
    parser.add_argument("--style", choices=["paper", "original"], default="paper", help="Readable manuscript style, or initial pixel-layout reconstruction.")
    parser.add_argument("--font", help="Optional TrueType font path; Arial is preferred.")
    args = parser.parse_args()
    if args.page_width <= 0:
        parser.error("--page-width must be positive")
    data = json.loads(args.data.read_text(encoding="utf-8"))
    validate_data(data)
    register_font(args.font)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    if 3 in args.figures:
        outputs.append(draw_heatmap(data["heatmap"], args.output_dir, args.page_width, args.style))
    if 4 in args.figures:
        outputs.extend(draw_curve(curve, args.output_dir, args.page_width, args.style) for curve in data["curves"])
    if 5 in args.figures:
        outputs.append(draw_case_study(data["case_study"], args.output_dir, args.page_width, args.style))
    for output in outputs:
        print(output.resolve())


if __name__ == "__main__":
    main()
