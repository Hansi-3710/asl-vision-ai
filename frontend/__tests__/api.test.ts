import axios from "axios";
import { describeApiError } from "@/lib/api";

jest.mock("axios", () => {
  const actual = jest.requireActual("axios");
  return {
    ...actual,
    isAxiosError: jest.fn(),
  };
});

const mockedIsAxiosError = axios.isAxiosError as unknown as jest.Mock;

describe("describeApiError", () => {
  afterEach(() => jest.clearAllMocks());

  it("returns a specific message for a 503 (model not loaded)", () => {
    mockedIsAxiosError.mockReturnValue(true);
    const error = { response: { status: 503, data: {} } };
    expect(describeApiError(error)).toMatch(/model isn't loaded/i);
  });

  it("surfaces the backend's detail message when present", () => {
    mockedIsAxiosError.mockReturnValue(true);
    const error = { response: { status: 400, data: { detail: "image_base64 is not valid base64." } } };
    expect(describeApiError(error)).toBe("image_base64 is not valid base64.");
  });

  it("reports a timeout distinctly", () => {
    mockedIsAxiosError.mockReturnValue(true);
    const error = { code: "ECONNABORTED", response: undefined };
    expect(describeApiError(error)).toMatch(/timed out/i);
  });

  it("reports an unreachable backend distinctly from other errors", () => {
    mockedIsAxiosError.mockReturnValue(true);
    const error = { response: undefined, code: undefined };
    expect(describeApiError(error)).toMatch(/couldn't reach the backend/i);
  });

  it("falls back to a generic message for non-axios errors", () => {
    mockedIsAxiosError.mockReturnValue(false);
    expect(describeApiError(new Error("some other error"))).toBe("Something went wrong. Please try again.");
  });
});
