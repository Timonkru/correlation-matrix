#!/usr/bin/env python3
"""
Correlation Matrix - Animated Video Generator
KruegerAlgorithms | https://kruegeralgorithms.com

Generates a cinematic Instagram Reel / YouTube Short from
live correlation data using Manim (the 3Blue1Brown engine).

Features:
  - 2D animated heatmap with diagonal cell fill
  - 3D correlation landscape (hills & valleys)
  - Rolling correlation line chart (2D)
  - 3D rolling correlation (lines in 3D space with camera rotation)
  - Branded intro/outro

Usage:
    python generate_video.py                          # Full video (all scenes)
    python generate_video.py --vertical               # 9:16 portrait (Instagram/TikTok)
    python generate_video.py --quality high            # 1080p 60fps
    python generate_video.py --scene 3d               # Only 3D landscape
    python generate_video.py --scene rolling3d         # Only 3D rolling correlation
    python generate_video.py --scene 2d               # Only 2D heatmap
    python generate_video.py --scene full             # Everything (default)
    python generate_video.py --symbols ^GDAXI ^GSPC GC=F EURUSD=X
"""

import argparse
import sys
import os

# ─── Dependency checks ────────────────────────────────────────
try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install yfinance pandas numpy")
    sys.exit(1)

try:
    from manim import *
except ImportError:
    print("Error: manim required. Install with: pip install manim")
    sys.exit(1)


# ─── DATA ─────────────────────────────────────────────────────

DEFAULT_SYMBOLS = {
    '^GDAXI':   'DAX',
    '^GSPC':    'S&P 500',
    '^IXIC':    'NASDAQ',
    'EURUSD=X': 'EUR/USD',
    'GC=F':     'Gold',
    'CL=F':     'Oil',
}

def fetch_correlation_data(symbols: list, period: str = '1y'):
    """Fetch prices, compute correlation matrix and rolling correlations."""
    data = {}
    for sym in symbols:
        try:
            hist = yf.Ticker(sym).history(period=period)
            if len(hist) > 0:
                label = DEFAULT_SYMBOLS.get(sym, sym)
                close = hist['Close']
                close.index = close.index.tz_localize(None).normalize()
                data[label] = close
        except Exception:
            pass

    df = pd.DataFrame(data).dropna()
    returns = df.pct_change().dropna()
    corr = returns.corr()

    # Rolling correlations for top 3 most interesting pairs
    pairs = []
    labels = list(corr.columns)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            val = corr.iloc[i, j]
            pairs.append((labels[i], labels[j], val))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    top_pairs = pairs[:3]

    rolling_data = {}
    for s1, s2, _ in top_pairs:
        rolling = returns[s1].rolling(30).corr(returns[s2]).dropna()
        rolling_data[f"{s1} vs {s2}"] = rolling

    return corr, top_pairs, rolling_data, len(df)


# ─── COLOR HELPERS ────────────────────────────────────────────

def corr_to_color(val):
    """Map correlation value (-1 to 1) to color (red -> white -> green)."""
    if val >= 0:
        r = 1.0 - val * 0.7
        g = 1.0 - val * 0.2
        b = 1.0 - val * 0.7
        return rgb_to_color([r, g, b])
    else:
        r = 1.0 + val * 0.2
        g = 1.0 + val * 0.7
        b = 1.0 + val * 0.7
        return rgb_to_color([r, g, b])


def corr_to_rgb(val):
    """Return [r, g, b] for a correlation value."""
    if val >= 0:
        return [1.0 - val * 0.7, 1.0 - val * 0.2, 1.0 - val * 0.7]
    else:
        return [1.0 + val * 0.2, 1.0 + val * 0.7, 1.0 + val * 0.7]


def corr_text_color(val):
    """White text on strong colors, black on light backgrounds."""
    return WHITE if abs(val) > 0.5 else BLACK


# ─── SCENE: 3D CORRELATION LANDSCAPE ─────────────────────────

class CorrelationLandscapeScene(ThreeDScene):
    """3D surface visualization — hills for positive, valleys for negative."""

    def setup(self):
        self.camera.background_color = "#0d1117"

    def construct(self):
        symbols = getattr(self, 'custom_symbols', list(DEFAULT_SYMBOLS.keys()))
        corr, top_pairs, _, n_days = fetch_correlation_data(symbols)
        labels = list(corr.columns)
        n = len(labels)
        corr_values = corr.values

        # ── Title (fixed in frame, doesn't rotate) ──
        title = Text("3D Correlation Landscape", font_size=36, color=WHITE)
        title.to_edge(UP, buff=0.4)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title), run_time=0.8)

        # ── Set up 3D camera ──
        self.set_camera_orientation(phi=65 * DEGREES, theta=-45 * DEGREES, zoom=0.7)

        # ── Create 3D axes ──
        axes = ThreeDAxes(
            x_range=[0, n - 1, 1],
            y_range=[0, n - 1, 1],
            z_range=[-1, 1, 0.5],
            x_length=6,
            y_length=6,
            z_length=4,
            axis_config={"color": GREY_B, "stroke_width": 1},
        )

        # Z-axis labels
        z_labels = VGroup()
        for z_val in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            label = Text(f"{z_val:.1f}", font_size=12, color=GREY_B)
            label.rotate(90 * DEGREES, axis=RIGHT)
            label.next_to(axes.c2p(0, 0, z_val), LEFT + OUT, buff=0.2)
            z_labels.add(label)

        # Asset labels along X and Y axes
        x_labels = VGroup()
        for i, name in enumerate(labels):
            label = Text(name, font_size=max(8, int(48 / n)), color=WHITE)
            label.rotate(90 * DEGREES, axis=RIGHT)
            label.rotate(-45 * DEGREES, axis=OUT)
            label.next_to(axes.c2p(i, -0.5, 0), DOWN + LEFT, buff=0.1)
            x_labels.add(label)

        y_labels_3d = VGroup()
        for j, name in enumerate(labels):
            label = Text(name, font_size=max(8, int(48 / n)), color=WHITE)
            label.rotate(90 * DEGREES, axis=RIGHT)
            label.rotate(45 * DEGREES, axis=OUT)
            label.next_to(axes.c2p(-0.5, j, 0), LEFT + DOWN, buff=0.1)
            y_labels_3d.add(label)

        self.play(Create(axes), run_time=1.0)
        self.play(FadeIn(x_labels), FadeIn(y_labels_3d), FadeIn(z_labels), run_time=0.8)

        # ── Build 3D bars (one per cell) ──
        bars = VGroup()
        for i in range(n):
            for j in range(n):
                val = corr_values[i, j]
                height = abs(val) * 2  # Scale: 1.0 corr = 2 units tall
                if height < 0.05:
                    height = 0.05

                bar = Prism(
                    dimensions=[0.7, 0.7, height],
                )

                # Position: center of cell, z at half-height
                z_pos = val * 2 / 2  # Positive goes up, negative goes down
                bar.move_to(axes.c2p(i, j, 0))
                bar.shift(OUT * z_pos)

                # Color based on correlation
                rgb = corr_to_rgb(val)
                bar.set_color(rgb_to_color(rgb))
                bar.set_opacity(0.85)

                bars.add(bar)

        # ── Animate bars growing from the base plane ──
        # Start flat, then grow to full height
        flat_bars = VGroup()
        for i in range(n):
            for j in range(n):
                flat = Prism(dimensions=[0.7, 0.7, 0.01])
                flat.move_to(axes.c2p(i, j, 0))
                val = corr_values[i, j]
                rgb = corr_to_rgb(val)
                flat.set_color(rgb_to_color(rgb))
                flat.set_opacity(0.85)
                flat_bars.add(flat)

        self.play(
            LaggedStart(*[FadeIn(b) for b in flat_bars], lag_ratio=0.02),
            run_time=1.0
        )

        # Transform flat bars to full-height bars
        self.play(
            *[Transform(flat_bars[k], bars[k]) for k in range(len(bars))],
            run_time=2.5,
            rate_func=smooth,
        )
        self.wait(0.5)

        # ── Rotate camera around the landscape ──
        self.begin_ambient_camera_rotation(rate=0.15)
        self.wait(4)
        self.stop_ambient_camera_rotation()

        # ── Zoom into a peak (strongest positive correlation) ──
        if top_pairs:
            s1, s2, val = top_pairs[0]
            i = labels.index(s1)
            j = labels.index(s2)

            # Info text (fixed in frame)
            info = Text(
                f"Peak: {s1} - {s2} ({val:.2f})",
                font_size=22,
                color="#66bb6a" if val > 0 else "#ef5350"
            )
            info.to_edge(DOWN, buff=0.5)
            self.add_fixed_in_frame_mobjects(info)
            self.play(
                FadeIn(info, shift=UP * 0.2),
                run_time=0.5
            )

            # Camera move toward the peak
            self.move_camera(
                phi=45 * DEGREES,
                theta=-30 * DEGREES,
                zoom=1.0,
                run_time=2.0,
            )
            self.wait(1.5)

            self.play(FadeOut(info), run_time=0.4)

        # ── Final panoramic sweep ──
        self.move_camera(
            phi=55 * DEGREES,
            theta=-135 * DEGREES,
            zoom=0.65,
            run_time=2.5,
        )
        self.wait(1.0)

        # ── Fade out ──
        self.play(
            FadeOut(flat_bars), FadeOut(axes),
            FadeOut(x_labels), FadeOut(y_labels_3d), FadeOut(z_labels),
            FadeOut(title),
            run_time=0.8
        )


# ─── SCENE: 2D HEATMAP ──────────────────────────────────────

class CorrelationMatrixScene(Scene):
    """2D animated heatmap scene."""

    def setup(self):
        self.camera.background_color = "#0d1117"

    def construct(self):
        symbols = getattr(self, 'custom_symbols', list(DEFAULT_SYMBOLS.keys()))
        corr, top_pairs, rolling_data, n_days = fetch_correlation_data(symbols)
        labels = list(corr.columns)
        n = len(labels)

        # ══════════════════════════════════════════════════════
        # ACT 1: TITLE INTRO
        # ══════════════════════════════════════════════════════
        title = Text("Asset Correlation Matrix", font_size=42, color=WHITE)
        subtitle = Text("Live Data Analysis", font_size=24, color=GREY_B)
        subtitle.next_to(title, DOWN, buff=0.3)
        brand = Text("KruegerAlgorithms.com", font_size=18, color="#4fc3f7")
        brand.next_to(subtitle, DOWN, buff=0.5)

        self.play(Write(title), run_time=1.0)
        self.play(FadeIn(subtitle, shift=UP * 0.2), run_time=0.6)
        self.play(FadeIn(brand, shift=UP * 0.2), run_time=0.5)
        self.wait(0.8)
        self.play(FadeOut(title), FadeOut(subtitle), FadeOut(brand), run_time=0.6)

        # ══════════════════════════════════════════════════════
        # ACT 2: BUILD THE GRID
        # ══════════════════════════════════════════════════════

        cell_size = min(4.5 / n, 0.85)
        grid_width = n * cell_size
        grid_start_x = -grid_width / 2 + cell_size / 2
        grid_start_y = grid_width / 2 - cell_size / 2

        col_labels = VGroup()
        for j, name in enumerate(labels):
            x = grid_start_x + j * cell_size
            y = grid_start_y + cell_size * 0.8
            t = Text(name, font_size=max(10, int(60 / n)), color=WHITE)
            t.rotate(45 * DEGREES)
            t.move_to([x, y, 0])
            col_labels.add(t)

        row_labels = VGroup()
        for i, name in enumerate(labels):
            x = grid_start_x - cell_size * 0.9
            y = grid_start_y - i * cell_size
            t = Text(name, font_size=max(10, int(60 / n)), color=WHITE)
            t.move_to([x, y, 0])
            row_labels.add(t)

        self.play(
            LaggedStart(*[FadeIn(l, shift=DOWN * 0.2) for l in col_labels], lag_ratio=0.08),
            LaggedStart(*[FadeIn(l, shift=RIGHT * 0.2) for l in row_labels], lag_ratio=0.08),
            run_time=1.2
        )

        # ══════════════════════════════════════════════════════
        # ACT 3: CELLS FILL IN DIAGONALLY
        # ══════════════════════════════════════════════════════

        cells = VGroup()
        value_texts = VGroup()
        cell_rects = {}

        for i in range(n):
            for j in range(n):
                val = corr.iloc[i, j]
                x = grid_start_x + j * cell_size
                y = grid_start_y - i * cell_size

                rect = Square(side_length=cell_size * 0.92)
                rect.move_to([x, y, 0])
                rect.set_fill(corr_to_color(val), opacity=0.9)
                rect.set_stroke(color="#1a1a2e", width=1)
                cells.add(rect)
                cell_rects[(i, j)] = rect

                txt = Text(f"{val:.2f}", font_size=max(8, int(50 / n)),
                           color=corr_text_color(val), weight=BOLD)
                txt.move_to([x, y, 0])
                value_texts.add(txt)

        diagonal_order = []
        for d in range(2 * n - 1):
            for i in range(n):
                j = d - i
                if 0 <= j < n:
                    diagonal_order.append(i * n + j)

        ordered_cells = [cells[i] for i in diagonal_order]
        ordered_texts = [value_texts[i] for i in diagonal_order]

        self.play(
            LaggedStart(*[GrowFromCenter(c) for c in ordered_cells], lag_ratio=0.03),
            run_time=2.0
        )
        self.play(
            LaggedStart(*[FadeIn(t) for t in ordered_texts], lag_ratio=0.02),
            run_time=1.5
        )
        self.wait(0.5)

        # ══════════════════════════════════════════════════════
        # ACT 4: HIGHLIGHT STRONG CORRELATIONS
        # ══════════════════════════════════════════════════════

        for pair_idx, (s1, s2, val) in enumerate(top_pairs):
            i = labels.index(s1)
            j = labels.index(s2)

            rect1 = cell_rects[(i, j)]
            rect2 = cell_rects[(j, i)]

            highlight_color = "#66bb6a" if val > 0 else "#ef5350"
            glow1 = rect1.copy().set_stroke(color=highlight_color, width=4)
            glow2 = rect2.copy().set_stroke(color=highlight_color, width=4)

            direction = "positiv" if val > 0 else "negativ"
            info = Text(
                f"{s1} <> {s2}: {val:.2f} ({direction})",
                font_size=20, color=highlight_color
            )
            info.to_edge(DOWN, buff=0.4)

            self.play(
                Create(glow1), Create(glow2),
                FadeIn(info, shift=UP * 0.2),
                run_time=0.6
            )
            self.wait(1.0)
            self.play(FadeOut(glow1), FadeOut(glow2), FadeOut(info), run_time=0.4)

        self.wait(0.3)

        # Fade out 2D matrix
        all_matrix = VGroup(cells, value_texts, col_labels, row_labels)
        self.play(FadeOut(all_matrix), run_time=0.8)


# ─── SCENE: ROLLING CHART ───────────────────────────────────

class RollingChartScene(Scene):
    """Animated rolling correlation line chart."""

    def setup(self):
        self.camera.background_color = "#0d1117"

    def construct(self):
        symbols = getattr(self, 'custom_symbols', list(DEFAULT_SYMBOLS.keys()))
        corr, top_pairs, rolling_data, n_days = fetch_correlation_data(symbols)

        chart_title = Text("Rolling Correlation (30-Day)", font_size=32, color=WHITE)
        chart_title.to_edge(UP, buff=0.5)
        self.play(Write(chart_title), run_time=0.6)

        axes = Axes(
            x_range=[0, 1, 0.25],
            y_range=[-1, 1, 0.5],
            x_length=10,
            y_length=4.5,
            axis_config={"color": GREY_B, "include_numbers": False},
            tips=False,
        )
        axes.shift(DOWN * 0.3)

        y_labels = VGroup()
        for val in [-1.0, -0.5, 0, 0.5, 1.0]:
            label = Text(f"{val:.1f}", font_size=14, color=GREY_B)
            label.next_to(axes.c2p(0, val), LEFT, buff=0.15)
            y_labels.add(label)

        zero_line = DashedLine(axes.c2p(0, 0), axes.c2p(1, 0),
                               color=WHITE, stroke_width=0.5, stroke_opacity=0.3)
        pos_line = DashedLine(axes.c2p(0, 0.7), axes.c2p(1, 0.7),
                               color="#66bb6a", stroke_width=0.5, stroke_opacity=0.3)
        neg_line = DashedLine(axes.c2p(0, -0.7), axes.c2p(1, -0.7),
                               color="#ef5350", stroke_width=0.5, stroke_opacity=0.3)

        self.play(Create(axes), FadeIn(y_labels), run_time=0.8)
        self.play(Create(zero_line), Create(pos_line), Create(neg_line), run_time=0.4)

        line_colors = ["#4fc3f7", "#ff9800", "#ab47bc", "#66bb6a", "#ef5350"]
        legend_items = VGroup()

        for idx, (pair_name, series) in enumerate(rolling_data.items()):
            color = line_colors[idx % len(line_colors)]
            values = series.values
            n_points = len(values)

            if n_points < 2:
                continue

            points = []
            for k, v in enumerate(values):
                x = k / (n_points - 1)
                points.append(axes.c2p(x, np.clip(v, -1, 1)))

            line = VMobject()
            line.set_points_smoothly(points)
            line.set_stroke(color=color, width=2, opacity=0.9)

            legend_dot = Dot(radius=0.06, color=color)
            legend_text = Text(pair_name, font_size=12, color=color)
            legend_text.next_to(legend_dot, RIGHT, buff=0.1)
            legend_items.add(VGroup(legend_dot, legend_text))

            self.play(Create(line), run_time=1.2)

        legend_items.arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        legend_items.to_corner(UR, buff=0.5).shift(DOWN * 0.5)
        self.play(FadeIn(legend_items), run_time=0.5)
        self.wait(1.5)

        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.8)


# ─── SCENE: 3D ROLLING CORRELATION ──────────────────────────

class RollingChart3DScene(ThreeDScene):
    """3D rolling correlation — lines floating in depth with camera rotation."""

    def setup(self):
        self.camera.background_color = "#0d1117"

    def construct(self):
        symbols = getattr(self, 'custom_symbols', list(DEFAULT_SYMBOLS.keys()))
        corr, top_pairs, rolling_data, n_days = fetch_correlation_data(symbols)

        if not rolling_data:
            return

        # ── Title (fixed in frame) ──
        title = Text("3D Rolling Correlation (30-Day)", font_size=34, color=WHITE)
        title.to_edge(UP, buff=0.4)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title), run_time=0.8)

        # ── Camera setup ──
        self.set_camera_orientation(phi=70 * DEGREES, theta=-55 * DEGREES, zoom=0.65)

        n_pairs = len(rolling_data)

        # ── 3D Axes ──
        # X = time (normalized 0→1), Y = depth (one lane per pair), Z = correlation
        axes = ThreeDAxes(
            x_range=[0, 1, 0.25],
            y_range=[0, max(n_pairs - 1, 1), 1],
            z_range=[-1, 1, 0.5],
            x_length=10,
            y_length=max(n_pairs * 1.8, 3),
            z_length=4,
            axis_config={"color": GREY_B, "stroke_width": 1},
        )

        # Z-axis labels (correlation values)
        z_labels = VGroup()
        for z_val in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            label = Text(f"{z_val:.1f}", font_size=12, color=GREY_B)
            label.rotate(90 * DEGREES, axis=RIGHT)
            label.next_to(axes.c2p(0, 0, z_val), LEFT + OUT, buff=0.2)
            z_labels.add(label)

        # Reference lines at z=0, z=0.7, z=-0.7 (across all depth lanes)
        ref_lines = VGroup()
        for y_idx in range(n_pairs):
            zero_plane = Line3D(
                start=axes.c2p(0, y_idx, 0),
                end=axes.c2p(1, y_idx, 0),
                color=WHITE,
                stroke_width=0.5,
            ).set_opacity(0.2)
            ref_lines.add(zero_plane)

            pos_ref = Line3D(
                start=axes.c2p(0, y_idx, 0.7),
                end=axes.c2p(1, y_idx, 0.7),
                color="#66bb6a",
                stroke_width=0.5,
            ).set_opacity(0.15)
            ref_lines.add(pos_ref)

            neg_ref = Line3D(
                start=axes.c2p(0, y_idx, -0.7),
                end=axes.c2p(1, y_idx, -0.7),
                color="#ef5350",
                stroke_width=0.5,
            ).set_opacity(0.15)
            ref_lines.add(neg_ref)

        self.play(Create(axes), FadeIn(z_labels), run_time=1.0)
        self.play(FadeIn(ref_lines), run_time=0.5)

        # ── Build 3D lines for each pair ──
        line_colors = ["#4fc3f7", "#ff9800", "#ab47bc", "#66bb6a", "#ef5350"]
        all_lines = VGroup()
        legend_items = VGroup()

        for idx, (pair_name, series) in enumerate(rolling_data.items()):
            color = line_colors[idx % len(line_colors)]
            values = series.values
            n_points = len(values)

            if n_points < 2:
                continue

            # Create 3D line path
            # X = normalized time, Y = pair index (depth lane), Z = correlation
            points_3d = []
            step = max(1, n_points // 200)  # Subsample for performance
            for k in range(0, n_points, step):
                t = k / (n_points - 1)
                z_val = float(np.clip(values[k], -1, 1))
                point = axes.c2p(t, idx, z_val)
                points_3d.append(point)

            # Ensure we include the last point
            if (n_points - 1) % step != 0:
                t = 1.0
                z_val = float(np.clip(values[-1], -1, 1))
                points_3d.append(axes.c2p(t, idx, z_val))

            line = VMobject()
            line.set_points_smoothly(points_3d)
            line.set_stroke(color=color, width=2.5, opacity=0.9)
            all_lines.add(line)

            # Pair label at the start of each line (3D positioned)
            pair_label = Text(pair_name, font_size=11, color=color)
            pair_label.rotate(90 * DEGREES, axis=RIGHT)
            pair_label.rotate(-45 * DEGREES, axis=OUT)
            pair_label.next_to(axes.c2p(0, idx, 0), LEFT + OUT, buff=0.3)

            # Legend for fixed-in-frame display
            legend_dot = Dot(radius=0.06, color=color)
            legend_text = Text(pair_name, font_size=12, color=color)
            legend_text.next_to(legend_dot, RIGHT, buff=0.1)
            legend_items.add(VGroup(legend_dot, legend_text))

            # Animate each line drawing + label
            self.play(
                Create(line),
                FadeIn(pair_label),
                run_time=1.2
            )

        # ── Legend (fixed in frame) ──
        legend_items.arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        legend_items.to_corner(UR, buff=0.5).shift(DOWN * 0.5)
        self.add_fixed_in_frame_mobjects(legend_items)
        self.play(FadeIn(legend_items), run_time=0.5)

        # ── Rotate camera around the 3D lines ──
        self.begin_ambient_camera_rotation(rate=0.12)
        self.wait(4)
        self.stop_ambient_camera_rotation()

        # ── Side view — shows depth separation ──
        self.move_camera(
            phi=75 * DEGREES,
            theta=10 * DEGREES,
            zoom=0.7,
            run_time=2.0,
        )
        self.wait(1.5)

        # ── Top-down view — looks like traditional 2D chart ──
        self.move_camera(
            phi=5 * DEGREES,
            theta=-90 * DEGREES,
            zoom=0.8,
            run_time=2.0,
        )
        self.wait(1.0)

        # ── Final dramatic angle ──
        self.move_camera(
            phi=60 * DEGREES,
            theta=-135 * DEGREES,
            zoom=0.65,
            run_time=2.0,
        )
        self.wait(1.0)

        # ── Highlight strongest pair info ──
        if top_pairs:
            s1, s2, val = top_pairs[0]
            info = Text(
                f"Strongest: {s1} - {s2} ({val:.2f})",
                font_size=22,
                color="#66bb6a" if val > 0 else "#ef5350"
            )
            info.to_edge(DOWN, buff=0.5)
            self.add_fixed_in_frame_mobjects(info)
            self.play(FadeIn(info, shift=UP * 0.2), run_time=0.5)
            self.wait(1.5)
            self.play(FadeOut(info), run_time=0.4)

        # ── Fade out ──
        self.play(
            *[FadeOut(mob) for mob in self.mobjects],
            FadeOut(title), FadeOut(legend_items),
            run_time=0.8
        )


# ─── SCENE: OUTRO ───────────────────────────────────────────

class OutroScene(Scene):
    """Summary stats and branded outro."""

    def setup(self):
        self.camera.background_color = "#0d1117"

    def construct(self):
        symbols = getattr(self, 'custom_symbols', list(DEFAULT_SYMBOLS.keys()))
        corr, top_pairs, _, n_days = fetch_correlation_data(symbols)
        labels = list(corr.columns)

        stats_title = Text("Summary", font_size=36, color=WHITE)
        stats = VGroup()
        stats.add(Text(f"{len(labels)} Assets Analyzed", font_size=22, color=GREY_B))
        stats.add(Text(f"{n_days} Trading Days", font_size=22, color=GREY_B))

        strong_pos = sum(1 for _, _, v in top_pairs if v > 0.7)
        strong_neg = sum(1 for _, _, v in top_pairs if v < -0.7)
        if strong_pos > 0:
            stats.add(Text(f"{strong_pos} Strong Positive Correlation(s)", font_size=22, color="#66bb6a"))
        if strong_neg > 0:
            stats.add(Text(f"{strong_neg} Strong Negative Correlation(s)", font_size=22, color="#ef5350"))

        stats.arrange(DOWN, buff=0.25)
        stats_title.next_to(stats, UP, buff=0.5)

        self.play(Write(stats_title), run_time=0.5)
        self.play(LaggedStart(*[FadeIn(s, shift=UP * 0.2) for s in stats], lag_ratio=0.15), run_time=1.0)
        self.wait(1.0)
        self.play(FadeOut(stats_title), FadeOut(stats), run_time=0.5)

        logo = Text("KruegerAlgorithms", font_size=44, color=WHITE, weight=BOLD)
        tagline = Text("Free Trading Tools", font_size=22, color="#4fc3f7")
        url = Text("kruegeralgorithms.com", font_size=20, color=GREY_B)
        tagline.next_to(logo, DOWN, buff=0.3)
        url.next_to(tagline, DOWN, buff=0.3)

        self.play(Write(logo), run_time=0.8)
        self.play(FadeIn(tagline), run_time=0.5)
        self.play(FadeIn(url), run_time=0.5)
        self.wait(2.0)
        self.play(FadeOut(logo), FadeOut(tagline), FadeOut(url), run_time=0.8)


# ─── CLI ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Generate Correlation Matrix Video')
    parser.add_argument('--symbols', '-s', nargs='+',
                        default=list(DEFAULT_SYMBOLS.keys()),
                        help='Yahoo Finance symbols')
    parser.add_argument('--vertical', '-v', action='store_true',
                        help='9:16 portrait mode (Instagram/TikTok)')
    parser.add_argument('--quality', '-q', default='medium',
                        choices=['low', 'medium', 'high'],
                        help='Video quality')
    parser.add_argument('--scene', default='full',
                        choices=['full', '2d', '3d', 'rolling', 'rolling3d', 'outro'],
                        help='Which scene to render')
    parser.add_argument('--output', '-o', default=None,
                        help='Output filename')
    args = parser.parse_args()

    quality_map = {
        'low':    'l',
        'medium': 'm',
        'high':   'h',
    }
    q_flag = quality_map[args.quality]

    scene_map = {
        'full':      ['CorrelationMatrixScene', 'CorrelationLandscapeScene',
                      'RollingChartScene', 'RollingChart3DScene', 'OutroScene'],
        '2d':        ['CorrelationMatrixScene'],
        '3d':        ['CorrelationLandscapeScene'],
        'rolling':   ['RollingChartScene'],
        'rolling3d': ['RollingChart3DScene'],
        'outro':     ['OutroScene'],
    }
    scenes = scene_map[args.scene]

    os.environ['CORR_SYMBOLS'] = ','.join(args.symbols)
    script_path = os.path.abspath(__file__)

    for scene_name in scenes:
        cmd_parts = [
            sys.executable, '-m', 'manim', 'render',
            f'-q{q_flag}',
            '--disable_caching',
        ]
        if args.vertical:
            cmd_parts.extend(['--resolution', '1080,1920'])

        cmd_parts.extend([script_path, scene_name])

        cmd = ' '.join(f'"{p}"' if ' ' in p else p for p in cmd_parts)
        print(f"\nRendering {scene_name} ({args.quality} quality)...")
        os.system(cmd)

    print("\nDone! Check the media/ folder for the output videos.")
    print("\nTo combine into one video, use:")
    print('  ffmpeg -f concat -i filelist.txt -c copy final_video.mp4')


if __name__ == '__main__':
    main()
