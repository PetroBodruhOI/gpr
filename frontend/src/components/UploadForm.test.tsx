import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import UploadForm from "./UploadForm";

const mockPredictFile = vi.fn();

vi.mock("../api/client", () => ({
  predictFile: (...args: unknown[]) => mockPredictFile(...args),
}));

const makeAudioFile = (name = "test.mp3", type = "audio/mpeg") =>
  new File(["data"], name, { type });

describe("UploadForm", () => {
  beforeEach(() => {
    mockPredictFile.mockReset();
  });

  it("renders upload area with placeholder text", () => {
    render(<UploadForm onStart={vi.fn()} />);
    expect(
      screen.getByText(/Перетягніть аудіо або відео або натисніть/),
    ).toBeInTheDocument();
  });

  it("renders supported format labels", () => {
    render(<UploadForm onStart={vi.fn()} />);
    expect(screen.getByText(/MP3 · WAV · OGG · FLAC/)).toBeInTheDocument();
  });

  it("submit button is disabled when no file selected", () => {
    render(<UploadForm onStart={vi.fn()} />);
    expect(screen.getByRole("button", { name: /Підібрати патерн/ })).toBeDisabled();
  });

  it("shows 'Файл готовий' and enables submit after valid file selected", async () => {
    const user = userEvent.setup();
    render(<UploadForm onStart={vi.fn()} />);

    const input = document.querySelector("input[type=file]") as HTMLInputElement;
    await user.upload(input, makeAudioFile("song.mp3", "audio/mpeg"));

    expect(screen.getByText("Файл готовий")).toBeInTheDocument();
    // File name appears in the info card (truncated paragraph)
    expect(screen.getAllByText("song.mp3").length).toBeGreaterThan(0);
    expect(screen.getByRole("button", { name: /Підібрати патерн/ })).not.toBeDisabled();
  });

  it("shows error message for unsupported file type", async () => {
    const user = userEvent.setup();
    render(<UploadForm onStart={vi.fn()} />);

    const input = document.querySelector("input[type=file]") as HTMLInputElement;
    // Extension .mp3 passes the accept filter; MIME "application/pdf" is not in allowed[]
    await user.upload(input, new File(["x"], "fake.mp3", { type: "application/pdf" }));

    expect(
      screen.getByText(/Підтримуються аудіофайли/),
    ).toBeInTheDocument();
    expect(screen.queryByText("Файл готовий")).not.toBeInTheDocument();
  });

  it("removes file when remove button clicked", async () => {
    const user = userEvent.setup();
    render(<UploadForm onStart={vi.fn()} />);

    const input = document.querySelector("input[type=file]") as HTMLInputElement;
    await user.upload(input, makeAudioFile());

    const fileCard = screen.getByText("Файл готовий").closest("div[style]")!;
    const removeBtn = within(fileCard as HTMLElement).getByRole("button");
    await user.click(removeBtn);

    expect(screen.queryByText("Файл готовий")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Підібрати патерн/ })).toBeDisabled();
  });

  it("calls predictFile and onStart when form submitted", async () => {
    const user = userEvent.setup();
    const onStart = vi.fn();
    mockPredictFile.mockResolvedValue("task-123");

    render(<UploadForm onStart={onStart} />);

    const input = document.querySelector("input[type=file]") as HTMLInputElement;
    await user.upload(input, makeAudioFile());

    await user.click(screen.getByRole("button", { name: /Підібрати патерн/ }));

    expect(mockPredictFile).toHaveBeenCalledTimes(1);
    await vi.waitFor(() => expect(onStart).toHaveBeenCalledWith("task-123"));
  });

  it("shows 'Обробка' label while submitting", async () => {
    const user = userEvent.setup();
    let resolve!: (v: string) => void;
    mockPredictFile.mockReturnValue(
      new Promise<string>((r) => { resolve = r; }),
    );

    render(<UploadForm onStart={vi.fn()} />);

    const input = document.querySelector("input[type=file]") as HTMLInputElement;
    await user.upload(input, makeAudioFile());
    await user.click(screen.getByRole("button", { name: /Підібрати патерн/ }));

    expect(screen.getByText(/Обробка/)).toBeInTheDocument();

    resolve("done");
  });

  it("shows 'Відпустіть, щоб завантажити' on dragOver", () => {
    render(<UploadForm onStart={vi.fn()} />);

    const dropLabel = screen
      .getByText(/Перетягніть аудіо або відео/)
      .closest("label")!;

    fireEvent.dragOver(dropLabel);
    expect(screen.getByText("Відпустіть, щоб завантажити")).toBeInTheDocument();
  });

  it("resets dragOver state on dragLeave", () => {
    render(<UploadForm onStart={vi.fn()} />);

    const dropLabel = screen
      .getByText(/Перетягніть аудіо або відео/)
      .closest("label")!;

    fireEvent.dragOver(dropLabel);
    fireEvent.dragLeave(dropLabel);

    expect(screen.queryByText("Відпустіть, щоб завантажити")).not.toBeInTheDocument();
  });

  it("accepts valid audio file on drop", () => {
    render(<UploadForm onStart={vi.fn()} />);

    const dropLabel = screen
      .getByText(/Перетягніть аудіо або відео/)
      .closest("label")!;
    const file = makeAudioFile("dropped.wav", "audio/wav");

    fireEvent.drop(dropLabel, {
      dataTransfer: { files: [file] },
    });

    expect(screen.getByText("Файл готовий")).toBeInTheDocument();
  });

  it("clears error when a valid file is selected after an invalid one", async () => {
    const user = userEvent.setup();
    render(<UploadForm onStart={vi.fn()} />);

    const input = document.querySelector("input[type=file]") as HTMLInputElement;
    await user.upload(input, new File(["x"], "fake.mp3", { type: "application/pdf" }));
    expect(screen.getByText(/Підтримуються аудіофайли/)).toBeInTheDocument();

    await user.upload(input, makeAudioFile("good.mp3", "audio/mpeg"));
    expect(screen.queryByText(/Підтримуються аудіофайли/)).not.toBeInTheDocument();
    expect(screen.getByText("Файл готовий")).toBeInTheDocument();
  });
});
