import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import PatternsGallery from "./PatternsGallery";
import { PATTERN_ORDER, PATTERNS } from "../data/patterns";

describe("PatternsGallery", () => {
  it("renders both section headers (Бої + Арпеджіо)", () => {
    render(<PatternsGallery />);
    // h2 (section headers) — pattern names are h3, narrow by level
    expect(screen.getByRole("heading", { level: 2, name: /Бої/ })).toBeInTheDocument();
    expect(screen.getByRole("heading", { level: 2, name: /Арпеджіо/ })).toBeInTheDocument();
  });

  it("renders every pattern from PATTERN_ORDER with its label", () => {
    render(<PatternsGallery />);
    for (const key of PATTERN_ORDER) {
      const p = PATTERNS[key];
      if (!p) continue;
      // label appears in pattern card (e.g. "6a", "8b", "wa")
      expect(screen.getAllByText(p.label).length).toBeGreaterThan(0);
    }
  });

  it("renders Ukrainian legend tokens for both kinds", () => {
    render(<PatternsGallery />);
    // strum legend
    expect(screen.getAllByText(/вниз/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/вгору/).length).toBeGreaterThan(0);
    // arp legend
    expect(screen.getAllByText(/басова нота/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/номер струни/).length).toBeGreaterThan(0);
  });
});
