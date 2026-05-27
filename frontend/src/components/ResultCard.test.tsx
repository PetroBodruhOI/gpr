import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import ResultCard, { pickRecommendations } from "./ResultCard";
import type { PredictResult } from "../api/client";

// ── Pure logic: pickRecommendations ────────────────────────────────────────

describe("pickRecommendations", () => {
  it("returns single top class when confidence is high", () => {
    const recs = pickRecommendations({
      "6a": 0.85, "6b": 0.10, "8a": 0.03, "8b": 0.02,
    });
    expect(recs).toHaveLength(1);
    expect(recs[0]).toEqual({ label: "6a", prob: 0.85 });
  });

  it("returns multiple candidates when model is uncertain (below threshold)", () => {
    const recs = pickRecommendations({
      "6a": 0.40, "6b": 0.30, "8a": 0.20, "8b": 0.10,
    });
    expect(recs.length).toBeGreaterThan(1);
    expect(recs[0].label).toBe("6a");
    // Sorted descending by prob
    for (let i = 1; i < recs.length; i++) {
      expect(recs[i - 1].prob).toBeGreaterThanOrEqual(recs[i].prob);
    }
  });

  it("filters out classes below chance level (1/n)", () => {
    const recs = pickRecommendations({
      "6a": 0.40, "6b": 0.30, "8a": 0.20, "8b": 0.10,
    });
    // Chance = 1/4 = 0.25, so 8a (0.20) and 8b (0.10) excluded
    expect(recs.every((r) => r.prob > 0.25)).toBe(true);
  });

  it("caps results at 3 candidates", () => {
    const recs = pickRecommendations({
      "a": 0.20, "b": 0.19, "c": 0.18, "d": 0.17, "e": 0.16, "f": 0.10,
    });
    expect(recs.length).toBeLessThanOrEqual(3);
  });

  it("handles empty probs gracefully", () => {
    expect(pickRecommendations({})).toEqual([]);
  });

  it("uses 0.75 threshold for high-confidence shortcut", () => {
    // 0.74 → below threshold → multiple candidates
    const low = pickRecommendations({ "6a": 0.74, "6b": 0.26 });
    // 0.75 → at threshold → single candidate
    const high = pickRecommendations({ "6a": 0.75, "6b": 0.25 });
    expect(low.length).toBeGreaterThanOrEqual(1);
    expect(high).toHaveLength(1);
  });
});

// ── Component: ResultCard + TimelineTable + FeedbackPanel ──────────────────

const buildResult = (overrides: Partial<PredictResult> = {}): PredictResult => ({
  final_label: "6a",
  final_conf: 0.88,
  mean_probs: { "6a": 0.88, "6b": 0.08, "8a": 0.04 },
  n_chunks: 5,
  chunks: Array.from({ length: 5 }, (_, i) => ({
    chunk_idx: i + 1,
    time_start: i * 6,
    time_end: (i + 1) * 6,
    label: "6a",
    confidence: 0.9 - i * 0.05,
    probs: { "6a": 0.9 - i * 0.05, "6b": 0.05, "8a": 0.05 },
  })),
  ...overrides,
});

vi.mock("../api/client", async () => {
  const actual = await vi.importActual<typeof import("../api/client")>("../api/client");
  return { ...actual, submitFeedback: vi.fn().mockResolvedValue(undefined) };
});

describe("ResultCard", () => {
  it("renders 'Рекомендований патерн' headline for high-confidence result", () => {
    render(<ResultCard result={buildResult({ final_conf: 0.92 })} taskId="t-1" />);
    expect(screen.getByText("Рекомендований патерн")).toBeInTheDocument();
  });

  it("renders 'Можливі патерни' headline for low-confidence result", () => {
    render(<ResultCard result={buildResult({ final_conf: 0.40 })} taskId="t-2" />);
    expect(screen.getByText("Можливі патерни")).toBeInTheDocument();
  });

  it("shows total segment count in timeline header", () => {
    render(<ResultCard result={buildResult({ n_chunks: 5 })} taskId="t-3" />);
    expect(screen.getByText(/5 сегментів/)).toBeInTheDocument();
  });
});

// ── TimelineTable expand/collapse ──────────────────────────────────────────

describe("TimelineTable", () => {
  it("shows only 2 rows initially when more chunks exist", () => {
    render(<ResultCard result={buildResult()} taskId="t-4" />);
    // 5 chunks total, preview = 2 → "Показати більше (3)"
    expect(screen.getByRole("button", { name: /Показати більше \(3\)/ })).toBeInTheDocument();
  });

  it("expands and collapses when toggle button is clicked", async () => {
    const user = userEvent.setup();
    render(<ResultCard result={buildResult()} taskId="t-5" />);

    const expandBtn = screen.getByRole("button", { name: /Показати більше/ });
    await user.click(expandBtn);

    expect(screen.getByRole("button", { name: /Показати менше/ })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Показати менше/ }));
    expect(screen.getByRole("button", { name: /Показати більше/ })).toBeInTheDocument();
  });

  it("does not render toggle button when chunks fit in preview", () => {
    render(<ResultCard result={buildResult({ n_chunks: 2, chunks: buildResult().chunks.slice(0, 2) })} taskId="t-6" />);
    expect(screen.queryByRole("button", { name: /Показати більше/ })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Показати менше/ })).not.toBeInTheDocument();
  });
});

// ── FeedbackPanel ──────────────────────────────────────────────────────────

describe("FeedbackPanel", () => {
  it("renders the CTA prompt", () => {
    render(<ResultCard result={buildResult()} taskId="t-7" />);
    expect(screen.getByText("Допоможіть нам покращити модель")).toBeInTheDocument();
    expect(screen.getByText(/Чи вважаєте, що рекомендація була правильною/)).toBeInTheDocument();
  });

  it("submits 'good' feedback and shows thank-you message", async () => {
    const user = userEvent.setup();
    const { submitFeedback } = await import("../api/client");
    render(<ResultCard result={buildResult()} taskId="task-good" />);

    await user.click(screen.getByRole("button", { name: /Так, підходить/ }));

    expect(submitFeedback).toHaveBeenCalledWith("task-good", "good");
    expect(await screen.findByText("Дякуємо за відгук!")).toBeInTheDocument();
  });

  it("submits 'bad' feedback", async () => {
    const user = userEvent.setup();
    const { submitFeedback } = await import("../api/client");
    render(<ResultCard result={buildResult()} taskId="task-bad" />);

    await user.click(screen.getByRole("button", { name: /Ні, не те/ }));

    expect(submitFeedback).toHaveBeenCalledWith("task-bad", "bad");
    expect(await screen.findByText("Дякуємо за відгук!")).toBeInTheDocument();
  });
});
