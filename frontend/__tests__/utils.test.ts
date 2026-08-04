import { cn, formatConfidence, formatLatency } from "@/lib/utils";

describe("cn (class merging)", () => {
  it("merges class names and resolves Tailwind conflicts", () => {
    expect(cn("p-2", "p-4")).toBe("p-4"); // tailwind-merge should keep only the last padding
  });

  it("handles conditional classes", () => {
    expect(cn("base", false && "hidden", "visible")).toBe("base visible");
  });

  it("handles no arguments", () => {
    expect(cn()).toBe("");
  });
});

describe("formatConfidence", () => {
  it("formats a 0-1 confidence value as a percentage with one decimal", () => {
    expect(formatConfidence(0.9873)).toBe("98.7%");
  });

  it("handles 0 and 1 correctly", () => {
    expect(formatConfidence(0)).toBe("0.0%");
    expect(formatConfidence(1)).toBe("100.0%");
  });
});

describe("formatLatency", () => {
  it("formats sub-second latency in milliseconds", () => {
    expect(formatLatency(45.2)).toBe("45ms");
    expect(formatLatency(999)).toBe("999ms");
  });

  it("switches to seconds at 1000ms and above", () => {
    expect(formatLatency(1000)).toBe("1.00s");
    expect(formatLatency(2500)).toBe("2.50s");
  });
});
