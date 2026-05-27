import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "./App";

// scrollIntoView is not implemented in jsdom — stub it.
// Restore between tests so spies are fresh.
const originalScrollIntoView = Element.prototype.scrollIntoView;
beforeEach(() => {
  Element.prototype.scrollIntoView = vi.fn();
});
afterEach(() => {
  Element.prototype.scrollIntoView = originalScrollIntoView;
});

describe("App", () => {
  it("opens on 'Аналіз' view by default (URL form visible)", () => {
    render(<App />);
    // The mode toggle for URL/file is unique to the analyze view
    expect(screen.getByRole("button", { name: /Посилання на YouTube/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Завантажити файл/ })).toBeInTheDocument();
  });

  it("renders sticky header with both nav items", () => {
    render(<App />);
    // CTA button "Аналіз" in header (there are two: header CTA + body controls)
    expect(screen.getAllByRole("button", { name: /Аналіз/ }).length).toBeGreaterThan(0);
    expect(screen.getByRole("button", { name: /Список патернів/ })).toBeInTheDocument();
  });

  it("switches to patterns view when 'Список патернів' clicked", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: /Список патернів/ }));

    // Patterns view shows section headings (h2). Pattern names are h3 so narrow by level.
    expect(screen.getByRole("heading", { level: 2, name: /Бої/ })).toBeInTheDocument();
    expect(screen.getByRole("heading", { level: 2, name: /Арпеджіо/ })).toBeInTheDocument();
    // Analyze form no longer mounted
    expect(screen.queryByRole("button", { name: /Посилання на YouTube/ })).not.toBeInTheDocument();
  });

  it("returns to analyze view + scrolls to form when header CTA clicked", async () => {
    const user = userEvent.setup();
    render(<App />);

    // First switch away from analyze
    await user.click(screen.getByRole("button", { name: /Список патернів/ }));
    expect(screen.queryByRole("button", { name: /Посилання на YouTube/ })).not.toBeInTheDocument();

    // Click the header CTA "Аналіз" — there's exactly one button named "Аналіз" while on patterns view
    const ctas = screen.getAllByRole("button", { name: /Аналіз/ });
    await user.click(ctas[0]);

    expect(screen.getByRole("button", { name: /Посилання на YouTube/ })).toBeInTheDocument();
    await waitFor(() => {
      expect(Element.prototype.scrollIntoView).toHaveBeenCalled();
    });
  });
});
