import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import PatternDisplay, { PatternLegend } from "./PatternDisplay";
import type { Pattern } from "../data/patterns";

// ─── Fixtures ────────────────────────────────────────────────────────────────

const strumPattern: Pattern = {
  kind: "strum",
  label: "test-s",
  name: "Test Strum",
  description: "D U X -",
  color: "#fff",
  steps: ["D", "U", "X", "-", "u"],
};

const arpPattern: Pattern = {
  kind: "arp",
  label: "test-a",
  name: "Test Arp",
  description: "b 1 2",
  color: "#fff",
  steps: [
    { isBass: true, strings: [6] },
    { strings: [1] },
    { strings: [2, 3] },
    { strings: [] },
  ],
};

// ─── PatternDisplay – strum ───────────────────────────────────────────────────

describe("PatternDisplay (strum)", () => {
  it("renders down-stroke arrow ↓", () => {
    render(<PatternDisplay pattern={strumPattern} />);
    expect(screen.getByText("↓")).toBeInTheDocument();
  });

  it("renders up-stroke arrow ↑ (both U and u)", () => {
    render(<PatternDisplay pattern={strumPattern} />);
    const arrows = screen.getAllByText("↑");
    expect(arrows.length).toBeGreaterThanOrEqual(2);
  });

  it("renders mute × symbol", () => {
    render(<PatternDisplay pattern={strumPattern} />);
    expect(screen.getByText("×")).toBeInTheDocument();
  });

  it("renders pause · symbol for '-' step", () => {
    render(<PatternDisplay pattern={strumPattern} />);
    expect(screen.getByText("·")).toBeInTheDocument();
  });

  it("renders beat labels 1–4 and &", () => {
    render(<PatternDisplay pattern={strumPattern} />);
    expect(screen.getByText("1")).toBeInTheDocument();
    expect(screen.getAllByText("&").length).toBeGreaterThan(0);
  });

  it("renders with 'sm' size without crashing", () => {
    render(<PatternDisplay pattern={strumPattern} size="sm" />);
    expect(screen.getByText("↓")).toBeInTheDocument();
  });

  it("renders with 'lg' size without crashing", () => {
    render(<PatternDisplay pattern={strumPattern} size="lg" />);
    expect(screen.getByText("↓")).toBeInTheDocument();
  });
});

// ─── PatternDisplay – arp ────────────────────────────────────────────────────

describe("PatternDisplay (arp)", () => {
  it("renders bass note 'B'", () => {
    render(<PatternDisplay pattern={arpPattern} />);
    expect(screen.getByText("B")).toBeInTheDocument();
  });

  it("renders single string number", () => {
    render(<PatternDisplay pattern={arpPattern} />);
    expect(screen.getByText("1")).toBeInTheDocument();
  });

  it("renders chord strings (multiple numbers)", () => {
    render(<PatternDisplay pattern={arpPattern} />);
    expect(screen.getByText("2")).toBeInTheDocument();
    expect(screen.getByText("3")).toBeInTheDocument();
  });

  it("renders rest step as '·' for empty strings array", () => {
    render(<PatternDisplay pattern={arpPattern} />);
    expect(screen.getByText("·")).toBeInTheDocument();
  });

  it("renders with 'sm' size without crashing", () => {
    render(<PatternDisplay pattern={arpPattern} size="sm" />);
    expect(screen.getByText("B")).toBeInTheDocument();
  });
});

// ─── PatternLegend ────────────────────────────────────────────────────────────

describe("PatternLegend (strum)", () => {
  it("renders ↓ вниз", () => {
    render(<PatternLegend kind="strum" />);
    expect(screen.getByText("↓")).toBeInTheDocument();
    expect(screen.getByText(/вниз/)).toBeInTheDocument();
  });

  it("renders ↑ вгору", () => {
    render(<PatternLegend kind="strum" />);
    expect(screen.getAllByText("↑").length).toBeGreaterThan(0);
    expect(screen.getByText(/вгору/)).toBeInTheDocument();
  });

  it("renders × приглушення", () => {
    render(<PatternLegend kind="strum" />);
    expect(screen.getByText("×")).toBeInTheDocument();
    expect(screen.getByText(/приглушення/)).toBeInTheDocument();
  });

  it("renders pause label", () => {
    render(<PatternLegend kind="strum" />);
    expect(screen.getByText(/пауза/)).toBeInTheDocument();
  });
});

describe("PatternLegend (arp)", () => {
  it("renders bass note label", () => {
    render(<PatternLegend kind="arp" />);
    expect(screen.getByText("B")).toBeInTheDocument();
    expect(screen.getByText(/басова нота/)).toBeInTheDocument();
  });

  it("renders string number range label", () => {
    render(<PatternLegend kind="arp" />);
    expect(screen.getByText(/1–6/)).toBeInTheDocument();
    expect(screen.getByText(/номер струни/)).toBeInTheDocument();
  });

  it("renders chord column label", () => {
    render(<PatternLegend kind="arp" />);
    expect(screen.getByText(/стовпчик = акорд/)).toBeInTheDocument();
  });
});
