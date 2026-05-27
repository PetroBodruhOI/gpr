import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// Mock axios.create to return a stub instance whose `get` we can drive.
// pollTask uses `api.get` internally (via the module-local `api` instance),
// so we have to intercept at the axios layer rather than mocking client.ts itself.
const mockGet  = vi.fn();
const mockPost = vi.fn();

vi.mock("axios", () => ({
  default: {
    create: () => ({ get: mockGet, post: mockPost }),
  },
}));

// Import AFTER the mock so client picks up the mocked axios
const { pollTask } = await import("./client");
import type { TaskStatus } from "./client";

const status = (overrides: Partial<TaskStatus> = {}): TaskStatus => ({
  task_id: "t-1",
  status: "processing",
  progress: 50,
  message: "working",
  ...overrides,
});

describe("pollTask", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    mockGet.mockReset();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it("resolves immediately on terminal 'done' status", async () => {
    mockGet.mockResolvedValueOnce({
      data: status({ status: "done", progress: 100 }),
    });
    const onUpdate = vi.fn();
    const final = await pollTask("t-1", onUpdate, 100);

    expect(final.status).toBe("done");
    expect(onUpdate).toHaveBeenCalledTimes(1);
    expect(mockGet).toHaveBeenCalledTimes(1);
    expect(mockGet).toHaveBeenCalledWith("/tasks/t-1");
  });

  it("resolves immediately on terminal 'error' status", async () => {
    mockGet.mockResolvedValueOnce({
      data: status({ status: "error", message: "boom" }),
    });

    const final = await pollTask("t-1", vi.fn(), 100);
    expect(final.status).toBe("error");
    expect(final.message).toBe("boom");
  });

  it("polls multiple times until status is terminal", async () => {
    mockGet
      .mockResolvedValueOnce({ data: status({ status: "pending",    progress: 0   }) })
      .mockResolvedValueOnce({ data: status({ status: "processing", progress: 50  }) })
      .mockResolvedValueOnce({ data: status({ status: "done",       progress: 100 }) });
    const onUpdate = vi.fn();

    const promise = pollTask("t-1", onUpdate, 100);
    await vi.runAllTimersAsync();
    const final = await promise;

    expect(final.status).toBe("done");
    expect(mockGet).toHaveBeenCalledTimes(3);
    expect(onUpdate).toHaveBeenCalledTimes(3);
  });
});
