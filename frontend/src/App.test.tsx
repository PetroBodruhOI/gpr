import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, render, screen } from "@testing-library/react";
import App from "./App";

const go = (hash: string) =>
  act(() => {
    window.location.hash = hash;
    window.dispatchEvent(new HashChangeEvent("hashchange"));
  });

beforeEach(() => {
  window.location.hash = "";
  window.scrollTo = vi.fn() as never;
});
afterEach(() => {
  window.location.hash = "";
});

describe("App routing", () => {
  it("opens on the home page with the explainer and an upload form", () => {
    render(<App />);
    expect(screen.getByRole("heading", { level: 1, name: /Почуй пісню/ })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /Анімація/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Підібрати патерн/ })).toBeInTheDocument();
  });

  it("has links to all three pages and marks the current one", () => {
    render(<App />);
    const nav = screen.getByRole("navigation");
    expect(nav.querySelector('a[href="#/"]')).toHaveAttribute("aria-current", "page");
    expect(nav.querySelector('a[href="#/patterns"]')).toBeInTheDocument();
    expect(nav.querySelector('a[href="#/detect"]')).toBeInTheDocument();
  });

  it("shows the patterns page on #/patterns", () => {
    render(<App />);
    go("#/patterns");
    expect(screen.getByRole("heading", { level: 2, name: /Бої/ })).toBeInTheDocument();
    expect(screen.getByRole("heading", { level: 2, name: /Арпеджіо/ })).toBeInTheDocument();
  });

  it("shows the detect page on #/detect", () => {
    render(<App />);
    go("#/detect");
    expect(screen.getByRole("heading", { level: 1, name: "Визначити патерн" })).toBeInTheDocument();
    expect(screen.queryByRole("img", { name: /Анімація/ })).not.toBeInTheDocument();
  });

  it("falls back to home for an unknown hash", () => {
    render(<App />);
    go("#/nope");
    expect(screen.getByRole("heading", { level: 1, name: /Почуй пісню/ })).toBeInTheDocument();
  });
});
