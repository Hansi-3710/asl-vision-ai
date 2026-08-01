import { render, screen } from "@testing-library/react";
import { BoundingBoxOverlay } from "@/components/camera/bounding-box-overlay";
import type { Prediction } from "@/types/prediction";

function makePrediction(overrides: Partial<Prediction> = {}): Prediction {
  return {
    id: "test-id",
    predicted_class: "A",
    confidence: 0.95,
    top_k: [{ class: "A", confidence: 0.95 }],
    source: "webcam",
    latency_ms: 20,
    image_path: null,
    bounding_box: { x_min: 0.25, y_min: 0.25, x_max: 0.75, y_max: 0.75, confidence: 0.9 },
    hand_detected: true,
    created_at: null,
    ...overrides,
  };
}

describe("BoundingBoxOverlay", () => {
  it("renders nothing box-related when there is no prediction yet", () => {
    const { container } = render(<BoundingBoxOverlay prediction={null} />);
    expect(container.querySelector(".border-signal")).not.toBeInTheDocument();
  });

  it("positions the box using percentage values derived from normalized coordinates", () => {
    const prediction = makePrediction();
    render(<BoundingBoxOverlay prediction={prediction} />);

    const box = document.querySelector(".border-signal.rounded-lg") as HTMLElement;
    expect(box).toBeInTheDocument();
    expect(box.style.left).toBe("25%");
    expect(box.style.top).toBe("25%");
    expect(box.style.width).toBe("50%"); // (0.75 - 0.25) * 100
    expect(box.style.height).toBe("50%");
  });

  it("shows the predicted letter and confidence label on the box", () => {
    const prediction = makePrediction({ predicted_class: "b", confidence: 0.876 });
    render(<BoundingBoxOverlay prediction={prediction} />);
    expect(screen.getByText("B")).toBeInTheDocument();
    expect(screen.getByText("87.6%")).toBeInTheDocument();
  });

  it("shows a 'no hand detected' hint when hand_detected is explicitly false", () => {
    const prediction = makePrediction({ bounding_box: null, hand_detected: false });
    render(<BoundingBoxOverlay prediction={prediction} />);
    expect(screen.getByText(/no hand detected/i)).toBeInTheDocument();
  });

  it("does not show the 'no hand detected' hint when hand detection is simply unavailable (null)", () => {
    const prediction = makePrediction({ bounding_box: null, hand_detected: null });
    render(<BoundingBoxOverlay prediction={prediction} />);
    expect(screen.queryByText(/no hand detected/i)).not.toBeInTheDocument();
  });
});
