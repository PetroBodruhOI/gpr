import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import ChunkDistributionChart from "./ChunkDistributionChart";
import type { ChunkResult } from "../api/client";

const makeChunk = (idx: number, probs: Record<string, number>): ChunkResult => ({
  chunk_idx: idx,
  time_start: (idx - 1) * 5,
  time_end: idx * 5,
  label: Object.keys(probs)[0],
  confidence: Object.values(probs)[0],
  probs,
});

const CHUNKS: ChunkResult[] = [
  makeChunk(1, { "6a": 0.8, "6b": 0.2 }),
  makeChunk(2, { "6a": 0.5, "6b": 0.5 }),
  makeChunk(3, { "6a": 0.3, "8a": 0.7 }),
];

describe("ChunkDistributionChart", () => {
  it("renders the title 'Розподіл за часом'", () => {
    render(<ChunkDistributionChart chunks={CHUNKS} />);
    expect(screen.getByText("Розподіл за часом")).toBeInTheDocument();
  });

  it("renders the subtitle text", () => {
    render(<ChunkDistributionChart chunks={CHUNKS} />);
    expect(
      screen.getByText(/Ймовірності кожного класу в усіх часових сегментах/),
    ).toBeInTheDocument();
  });

  it("renders legend entries for each unique class", () => {
    render(<ChunkDistributionChart chunks={CHUNKS} />);
    // 6a, 6b, 8a all appear across chunks
    const legends = screen.getAllByText(/^(6a|6b|8a)$/, { selector: "span.font-mono" });
    const labels = legends.map((el) => el.textContent);
    expect(labels).toContain("6a");
    expect(labels).toContain("6b");
    expect(labels).toContain("8a");
  });

  it("shows placeholder text when no segment is hovered", () => {
    render(<ChunkDistributionChart chunks={CHUNKS} />);
    expect(
      screen.getByText(/Наведіть на сегмент/),
    ).toBeInTheDocument();
  });

  it("shows tooltip with class name on mouseEnter", () => {
    const { container } = render(<ChunkDistributionChart chunks={CHUNKS} />);

    // Bars are divs inside the chart that have onMouseEnter
    const bars = container.querySelectorAll(
      ".flex-1.flex.flex-col-reverse [style*='height']",
    );
    expect(bars.length).toBeGreaterThan(0);

    fireEvent.mouseEnter(bars[0]);

    // Placeholder should be gone; tooltip with class label visible
    expect(screen.queryByText(/Наведіть на сегмент/)).not.toBeInTheDocument();
  });

  it("restores placeholder on mouseLeave", () => {
    const { container } = render(<ChunkDistributionChart chunks={CHUNKS} />);

    const bars = container.querySelectorAll(
      ".flex-1.flex.flex-col-reverse [style*='height']",
    );
    fireEvent.mouseEnter(bars[0]);
    fireEvent.mouseLeave(bars[0]);

    expect(screen.getByText(/Наведіть на сегмент/)).toBeInTheDocument();
  });

  it("renders percentage gridlines (0%, 25%, 50%, 75%, 100%)", () => {
    render(<ChunkDistributionChart chunks={CHUNKS} />);
    [0, 25, 50, 75, 100].forEach((v) => {
      expect(screen.getByText(`${v}%`)).toBeInTheDocument();
    });
  });

  it("renders known pattern name in legend for recognised class", () => {
    render(<ChunkDistributionChart chunks={CHUNKS} />);
    // "6a" maps to "Арпеджіо 1" in PATTERNS
    expect(screen.getAllByText("Арпеджіо 1").length).toBeGreaterThan(0);
  });

  it("handles empty chunks array without crashing", () => {
    render(<ChunkDistributionChart chunks={[]} />);
    expect(screen.getByText("Розподіл за часом")).toBeInTheDocument();
  });

  it("tooltip shows segment index and time range after hover", () => {
    const { container } = render(<ChunkDistributionChart chunks={CHUNKS} />);

    const bars = container.querySelectorAll(
      ".flex-1.flex.flex-col-reverse [style*='height']",
    );
    fireEvent.mouseEnter(bars[0]);

    // Chunk 1 → "сегмент #1 · 0.0–5.0с"
    expect(screen.getByText(/сегмент #/)).toBeInTheDocument();
  });
});
