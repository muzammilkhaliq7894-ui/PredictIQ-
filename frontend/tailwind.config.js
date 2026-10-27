/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#07111f",
        panel: "#0d1a2b",
        line: "#1d3148",
        cyan: "#51d5e5",
        lime: "#b8e986",
        amber: "#f7ba61",
        danger: "#ff7d72"
      },
      boxShadow: {
        glow: "0 0 32px rgba(81, 213, 229, 0.12)"
      }
    }
  },
  plugins: []
};
