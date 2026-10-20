import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import ProgressBar from "./ProgressBar";

describe("ProgressBar", () => {
  it("renders the 'Обробка' heading", () => {
    render(<ProgressBar progress={50} message="Завантаження" />);
    expect(screen.getByText("Обробка")).toBeInTheDocument();
  });

  it("displays the progress percentage", () => {
    render(<ProgressBar progress={42} message="" />);
    expect(screen.getByText("42%")).toBeInTheDocument();
  });

  it("displays 0% when progress is 0", () => {
    render(<ProgressBar progress={0} message="Старт" />);
    expect(screen.getByText("0%")).toBeInTheDocument();
  });

  it("displays 100% when complete", () => {
    render(<ProgressBar progress={100} message="Готово" />);
    expect(screen.getByText("100%")).toBeInTheDocument();
  });

  it("renders the message text", () => {
    render(<ProgressBar progress={70} message="Обробка аудіо" />);
    expect(screen.getByText("Обробка аудіо")).toBeInTheDocument();
  });

  it("renders empty message without error", () => {
    render(<ProgressBar progress={50} message="" />);
    expect(screen.getByText("50%")).toBeInTheDocument();
  });

  it("exposes progress to assistive tech and scales the bar", () => {
    const { container } = render(<ProgressBar progress={65} message="" />);
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "65");
    const fill = container.querySelector("[style]") as HTMLElement;
    expect(fill.style.transform).toBe("scaleX(0.65)");
  });

  it("marks finished stages and highlights the current one", () => {
    render(<ProgressBar progress={60} message="" />);
    const items = screen.getAllByRole("listitem");
    expect(items.map((li) => li.textContent)).toEqual([
      "Виділення гітари", "Аналіз ритму", "Підбір патерну",
    ]);
    expect(items[0].querySelector("svg")).toBeTruthy();   // done → check
    expect(items[1].querySelector("svg")).toBeNull();     // active → dot
  });
});
