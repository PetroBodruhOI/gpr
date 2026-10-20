import type { Config } from 'tailwindcss'

export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // "Ватра": a campfire stage at night.
        white: '#f4ead9',     // smoke — all text is warm, never pure white
        ink: '#171310',       // night ground
        surface: '#2a221c',   // coal — cards
        raised: '#3a2f26',    // cells and controls on cards
        accent: {
          DEFAULT: '#ff7a1a', // ember — primary action, downstrokes
          hover: '#ff9142',
        },
        glow: '#ffb347',      // amber — upstrokes, highlights, success
      },
      fontFamily: {
        display: ['Rubik', 'system-ui', 'sans-serif'],
        sans: ['"Golos Text"', 'system-ui', 'sans-serif'],
      },
      transitionTimingFunction: {
        out: 'cubic-bezier(0.23, 1, 0.32, 1)',
      },
    },
  },
  plugins: [],
} satisfies Config
