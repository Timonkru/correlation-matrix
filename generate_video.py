#!/usr/bin/env python3
"""
Correlation Matrix - Animated Video Generator
KruegerAlgorithms | https://kruegeralgorithms.com

Generates a cinematic Instagram Reel / YouTube Short from
live correlation data using Manim (the 3Blue1Brown engine).

Usage:
    python generate_video.py                         # 16:9 landscape (YouTube)
    python generate_video.py --vertical              # 9:16 portrait  (Instagram/TikTok)
    python generate_video.py --quality high           # 1080p 60fps
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


def corr_text_color(val):
    """White text on strong colors, black on light backgrounds."""
    return WHITE if abs(val) > 0.5 else BLACK


# ─── SCENE: CORRELATION MATRIX ───────────────────────────────

class CorrelationMatrixScene(Scene):
    """Main animated scene for the correlation matrix video."""

    def setup(self):
        self.camera.background_color = "#0d1117"

    def construct(self):
        # ── Fetch live data ──
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

        # Column labels (top)
        col_labels = VGroup()
        for j, name in enumerate(labels):
            x = grid_start_x + j * cell_size
            y = grid_start_y + cell_size * 0.8
            t = Text(name, font_size=max(10, int(60 / n)), color=WHITE)
            t.rotate(45 * DEGREES)
            t.move_to([x, y, 0])
            col_labels.add(t)

        # Row labels (left)
        row_labels = VGroup()
        for i, name in enumerate(labels):
            x = grid_start_x - cell_size * 0.9
            y = grid_start_y - i * cell_size
            t = Text(name, font_size=max(10, int(60 / n)), color=WHITE)
            t.move_to([x, y, 0])
            row_labels.add(t)

        # Animate labels appearing
        self.play(
            LaggedStart(*[FadeIn(l, shift=DOWN * 0.2) for l in col_labels], lag_ratio=0.08),
            LaggedStart(*[FadeIn(l, shift=RIGHT * 0.2) for l in row_labels], lag_ratio=0.08),
            run_time=1.2
        )

        # ══════════════════════════════════════════════════════
        # ACT 3: CELLS FILL IN WITH ANIMATED COUNTING
        # ══════════════════════════════════════════════════════

        cells = VGroup()
        value_texts = VGroup()
        cell_rects = {}

        for i in range(n):
            for j in range(n):
                val = corr.iloc[i, j]
                x = grid_start_x + j * cell_size
                y = grid_start_y - i * cell_size

                # Background rectangle
                rect = Square(side_length=cell_size * 0.92)
                rect.move_to([x, y, 0])
                rect.set_fill(corr_to_color(val), opacity=0.9)
                rect.set_stroke(color="#1a1a2e", width=1)
                cells.add(rect)
                cell_rects[(i, j)] = rect

                # Value text
                txt = Text(f"{val:.2f}", font_size=max(8, int(50 / n)),
                           color=corr_text_color(val), weight=BOLD)
                txt.move_to([x, y, 0])
                value_texts.add(txt)

        # Animate cells appearing diagonally
        diagonal_order = []
        for d in range(2 * n - 1):
            for i in range(n):
                j = d - i
                if 0 <= j < n:
                    idx = i * n + j
                    diagonal_order.append(idx)

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

            # Pulsing highlight
            highlight_color = "#66bb6a" if val > 0 else "#ef5350"
            glow1 = rect1.copy().set_stroke(color=highlight_color, width=4)
            glow2 = rect2.copy().set_stroke(color=highlight_color, width=4)

            # Info text
            direction = "positiv" if val > 0 else "negativ"
            emoji = "+" if val > 0 else "-"
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

            if pair_idx < len(top_pairs) - 1:
                self.play(FadeOut(glow1), FadeOut(glow2), FadeOut(info), run_time=0.4)
            else:
                self.play(FadeOut(glow1), FadeOut(glow2), FadeOut(info), run_time=0.4)

        self.wait(0.3)

        # ══════════════════════════════════════════════════════
        # ACT 5: TRANSITION TO ROLLING CHART
        # ══════════════════════════════════════════════════════

        # Fade out matrix
        all_matrix = VGroup(cells, value_texts, col_labels, row_labels)
        self.play(FadeOut(all_matrix), run_time=0.8)

        # Rolling correlation chart title
        chart_title = Text("Rolling Correlation (30-Day)", font_size=32, color=WHITE)
        chart_title.to_edge(UP, buff=0.5)
        self.play(Write(chart_title), run_time=0.6)

        # Draw axes
        axes = Axes(
            x_range=[0, 1, 0.25],
            y_range=[-1, 1, 0.5],
            x_length=10,
            y_length=4.5,
            axis_config={"color": GREY_B, "include_numbers": False},
            tips=False,
        )
        axes.shift(DOWN * 0.3)

        # Y-axis labels
        y_labels = VGroup()
        for val in [-1.0, -0.5, 0, 0.5, 1.0]:
            label = Text(f"{val:.1f}", font_size=14, color=GREY_B)
            label.next_to(axes.c2p(0, val), LEFT, buff=0.15)
            y_labels.add(label)

        # Reference lines
        zero_line = DashedLine(
            axes.c2p(0, 0), axes.c2p(1, 0),
            color=WHITE, stroke_width=0.5, stroke_opacity=0.3
        )
        pos_line = DashedLine(
            axes.c2p(0, 0.7), axes.c2p(1, 0.7),
            color="#66bb6a", stroke_width=0.5, stroke_opacity=0.3
        )
        neg_line = DashedLine(
            axes.c2p(0, -0.7), axes.c2p(1, -0.7),
            color="#ef5350", stroke_width=0.5, stroke_opacity=0.3
        )

        self.play(Create(axes), FadeIn(y_labels), run_time=0.8)
        self.play(Create(zero_line), Create(pos_line), Create(neg_line), run_time=0.4)

        # Plot rolling correlation lines
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

            # Legend entry
            legend_dot = Dot(radius=0.06, color=color)
            legend_text = Text(pair_name, font_size=12, color=color)
            legend_text.next_to(legend_dot, RIGHT, buff=0.1)
            legend_entry = VGroup(legend_dot, legend_text)
            legend_items.add(legend_entry)

            # Animate the line drawing
            self.play(Create(line), run_time=1.2)

        # Position legend
        legend_items.arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        legend_items.to_corner(UR, buff=0.5)
        legend_items.shift(DOWN * 0.5)
        self.play(FadeIn(legend_items), run_time=0.5)
        self.wait(1.5)

        # ══════════════════════════════════════════════════════
        # ACT 6: OUTRO
        # ══════════════════════════════════════════════════════

        # Fade everything
        self.play(
            *[FadeOut(mob) for mob in self.mobjects],
            run_time=0.8
        )

        # Stats summary
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

        # Brand outro
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
    parser.add_argument('--output', '-o', default=None,
                        help='Output filename')
    args = parser.parse_args()

    # Quality presets
    quality_map = {
        'low':    'l',    # 480p 15fps
        'medium': 'm',    # 720p 30fps
        'high':   'h',    # 1080p 60fps
    }
    q_flag = quality_map[args.quality]

    # Build manim command
    cmd_parts = [
        sys.executable, '-m', 'manim', 'render',
        f'-q{q_flag}',
        '--disable_caching',
    ]

    if args.vertical:
        # 9:16 portrait for Instagram Reels
        cmd_parts.extend([
            '--resolution', '1080,1920',
        ])

    script_path = os.path.abspath(__file__)
    cmd_parts.extend([script_path, 'CorrelationMatrixScene'])

    # Pass symbols via environment variable
    os.environ['CORR_SYMBOLS'] = ','.join(args.symbols)

    cmd = ' '.join(f'"{p}"' if ' ' in p else p for p in cmd_parts)
    print(f"Rendering video ({args.quality} quality)...")
    print(f"Command: {cmd}")
    os.system(cmd)

    print("\nDone! Check the media/ folder for the output video.")


if __name__ == '__main__':
    main()
