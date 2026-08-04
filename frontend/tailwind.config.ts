import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: ["class"],
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Base surfaces -- deep indigo-black, not flat #000.
        background: {
          DEFAULT: "#0A0D16",
          elevated: "#0F1420",
        },
        // Glassmorphic panel surfaces (used with backdrop-blur + border).
        glass: {
          DEFAULT: "rgba(255,255,255,0.04)",
          border: "rgba(255,255,255,0.08)",
          hover: "rgba(255,255,255,0.07)",
        },
        // Signal Cyan -- the primary accent. Evokes AR/CV viewfinder
        // overlays (bounding boxes, scan lines) rather than a generic
        // SaaS purple/blue.
        signal: {
          DEFAULT: "#4CE0D2",
          soft: "#8FEFE4",
          dim: "#2A7A72",
        },
        // Warm Amber -- secondary accent, used for confidence/attention
        // states (a detected-and-confirmed sign, a highlighted stat).
        amber: {
          DEFAULT: "#F5A623",
          soft: "#FBC873",
        },
        ink: {
          DEFAULT: "#F3F5F8",
          muted: "#8B93A8",
          faint: "#565D70",
        },
        danger: "#EF6461",
      },
      fontFamily: {
        display: ["var(--font-space-grotesk)", "sans-serif"],
        body: ["var(--font-inter)", "sans-serif"],
        mono: ["var(--font-jetbrains-mono)", "monospace"],
      },
      borderRadius: {
        xl: "1rem",
        "2xl": "1.5rem",
      },
      backdropBlur: {
        xs: "2px",
      },
      keyframes: {
        "gradient-shift": {
          "0%, 100%": { backgroundPosition: "0% 50%" },
          "50%": { backgroundPosition: "100% 50%" },
        },
        "scan-line": {
          "0%": { transform: "translateY(-100%)" },
          "100%": { transform: "translateY(100%)" },
        },
        "fade-up": {
          "0%": { opacity: "0", transform: "translateY(12px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        "pulse-ring": {
          "0%": { boxShadow: "0 0 0 0 rgba(76,224,210,0.35)" },
          "100%": { boxShadow: "0 0 0 16px rgba(76,224,210,0)" },
        },
      },
      animation: {
        "gradient-shift": "gradient-shift 8s ease infinite",
        "scan-line": "scan-line 2.4s linear infinite",
        "fade-up": "fade-up 0.6s cubic-bezier(0.16, 1, 0.3, 1) both",
        "pulse-ring": "pulse-ring 1.6s cubic-bezier(0.4, 0, 0.6, 1) infinite",
      },
      backgroundSize: {
        "gradient-xl": "200% 200%",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
};

export default config;
