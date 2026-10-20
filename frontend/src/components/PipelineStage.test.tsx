import { describe, expect, it } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import PipelineStage from "./PipelineStage";

const nextLoop = (container: HTMLElement) => {
  fireEvent.animationIteration(container.querySelector(".stage-playhead")!);
};

describe("PipelineStage", () => {
  it("explains the three stages in reading order", () => {
    const { container } = render(<PipelineStage />);
    const text = container.textContent ?? "";
    expect(text.indexOf("1 · Пісня")).toBeLessThan(text.indexOf("2 · Тільки гітара"));
    expect(text.indexOf("2 · Тільки гітара")).toBeLessThan(text.indexOf("3 · Як грати"));
  });

  it("draws one guitar hit and drop line per played step, none on rests", () => {
    const { container } = render(<PipelineStage />);
    // 6b = D - D U - U D U → 6 strokes, each with a hit marker and a line
    expect(container.querySelectorAll(".stage-transient")).toHaveLength(12);
    expect(container.querySelectorAll(".stage-arrow")).toHaveLength(6);
  });

  it("shows the next strum pattern on every loop, then wraps around", () => {
    const { container } = render(<PipelineStage />);
    expect(screen.getByText("Класичний бій")).toBeInTheDocument();
    nextLoop(container);
    expect(screen.getByText("Розгорнутий бій")).toBeInTheDocument();
    nextLoop(container);
    expect(screen.getByText("Бій з заглушуванням")).toBeInTheDocument();
    expect(container.querySelectorAll(".stage-arrow")).toHaveLength(6); // D - X U - U X U
    nextLoop(container);
    expect(screen.getByText("Класичний бій")).toBeInTheDocument();
  });
});
