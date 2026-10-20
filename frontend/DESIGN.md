---
name: GPR — Guitar Pattern Recommender
description: A campfire stage at night where a song's waveform settles, beat by beat, into the strum pattern you play.
colors:
  night: "#171310"
  coal: "#2a221c"
  ember-ash: "#3a2f26"
  ember: "#ff7a1a"
  ember-hover: "#ff9142"
  amber-glow: "#ffb347"
  smoke: "#f4ead9"
  error: "#fca5a5"
  pattern-6b: "#facc15"
  pattern-6x: "#f97316"
  pattern-8b: "#ec4899"
  pattern-6a: "#10b981"
  pattern-8a: "#06b6d4"
  pattern-wa: "#3b82f6"
typography:
  display:
    fontFamily: "Rubik, system-ui, sans-serif"
    fontSize: "4.5rem"
    fontWeight: 800
    lineHeight: 1.02
    letterSpacing: "-0.025em"
  headline:
    fontFamily: "Rubik, system-ui, sans-serif"
    fontSize: "2.25rem"
    fontWeight: 800
    lineHeight: 1.1
    letterSpacing: "-0.025em"
  title:
    fontFamily: "Rubik, system-ui, sans-serif"
    fontSize: "1.5rem"
    fontWeight: 700
    lineHeight: 1.25
  body:
    fontFamily: "Golos Text, system-ui, sans-serif"
    fontSize: "1rem"
    fontWeight: 400
    lineHeight: 1.6
  lead:
    fontFamily: "Golos Text, system-ui, sans-serif"
    fontSize: "1.25rem"
    fontWeight: 400
    lineHeight: 1.6
  label:
    fontFamily: "Golos Text, system-ui, sans-serif"
    fontSize: "0.6875rem"
    fontWeight: 600
    letterSpacing: "0.14em"
rounded:
  md: "8px"
  cell: "12px"
  card: "16px"
  panel: "24px"
  pill: "9999px"
spacing:
  card: "24px"
  panel: "32px"
  section: "128px"
components:
  button-primary:
    backgroundColor: "{colors.ember}"
    textColor: "{colors.night}"
    rounded: "{rounded.pill}"
    padding: "12px 24px"
  button-primary-hover:
    backgroundColor: "{colors.ember-hover}"
    textColor: "{colors.night}"
  button-primary-disabled:
    backgroundColor: "{colors.ember-ash}"
    textColor: "{colors.smoke}"
  button-secondary:
    backgroundColor: "{colors.night}"
    textColor: "{colors.smoke}"
    rounded: "{rounded.pill}"
    padding: "10px 20px"
  card:
    backgroundColor: "{colors.coal}"
    textColor: "{colors.smoke}"
    rounded: "{rounded.card}"
    padding: "{spacing.card}"
  step-cell:
    backgroundColor: "{colors.ember-ash}"
    rounded: "{rounded.md}"
  cta-band:
    backgroundColor: "{colors.ember}"
    textColor: "{colors.night}"
    rounded: "{rounded.panel}"
    padding: "64px 48px"
---

# Design System: GPR — Ватра

## Overview

**Creative North Star: Ватра (the campfire).** GPR is where a beginner guitarist sits down with a song at night and learns how to strum it. The whole site is that warm, low-lit stage: a night ground, coal-coloured surfaces, and ember light that appears only where sound is being turned into playing.

The world was chosen by the user in a direction round ("warm, musical", beginner guitarists, phone and laptop). It replaces an earlier cool-navy SaaS look. The signature is the home-page **pipeline stage**: a song waveform, the isolated guitar, and the eight strum cells, all on one 4-second clock. Guitar transients ignite and arrows strike exactly when the playhead reaches them.

Pages: **Головна** (persuade: explain and invite an upload), **Патерни** (reference: all six patterns), **Визначити** (operate: upload → stages → result). The home page and Detect page both carry the working upload form; the primary action is never only a link.

## Colors

One hue family carries meaning; everything else is warm neutral.

- **Night `#171310`**: the page ground. Never pure black, never blue-black.
- **Coal `#2a221c`**: cards and panels. **Ember-ash `#3a2f26`**: cells, disabled buttons, tracks on top of coal.
- **Ember `#ff7a1a`**: the primary action, downstrokes, the current stage, and the one full-bleed closing band. Text on ember is always Night.
- **Amber glow `#ffb347`**: upstrokes, completed stages, the playhead, lane labels, the logo, focus rings.
- **Smoke `#f4ead9`**: all text. It's mapped to Tailwind's `white`, so `text-white/70` is warm smoke at 70%. Body copy stays at 70% or higher on coal; hints at 50–55%.
- **Pattern series** (`pattern-*`) are *data colours* for the six classes in charts, timelines and labels. They exist for distinguishability, not atmosphere, and appear only where a class is identified.

**The Ember Rule.** Ember marks what acts or what sounds: the primary button, downstrokes, the active stage, the CTA band. It never decorates a heading, border or background just to add warmth.

## Typography

- **Rubik** (500/700/800) for every heading and the logo; heavy and rounded, with full Cyrillic. Display 3–4.5rem, line-height ~1.02, tracking −0.025em, `text-balance`.
- **Golos Text** (400–700) for body, labels and controls; a Cyrillic-first UI face. Lead copy 1.125–1.25rem.
- Numerals that change or compare (percentages, times, the progress figure) use `tabular-nums`.
- The only uppercase tracked text is the **lane labels** inside the pipeline stage (they name data lanes, not headings).
- Pattern notation strings (`D - D U - U D U`) are set in the monospace stack because they are notation data.

## Layout

- Container `max-w-7xl` with 16–24px gutters; the Detect page narrows to `max-w-3xl` because it is one column of task.
- Home sections are separated by 96–128px; inside a section, 32–40px between heading and content.
- Hero: two columns from `lg` (copy+upload ≈45%, stage ≈55%); stacked on phones with the upload form above the stage.
- "How it works" is **one connected band** of three stages divided by hairlines, with ember connector badges between stages. They point right on desktop and down on phones.
- Sticky header: logo left; Головна, Патерни and the ember «Визначити патерн» pill right (shortened to «Визначити» on phones).

## Elevation & Depth

Flat. Depth is tonal (night → coal → ember-ash) plus 6–8% smoke hairline borders. There are no drop shadows on cards. The only light effects are firelight: a soft ember radial glow behind the guitar lane of the stage, and the brief glow on a guitar transient as it ignites.

## Shapes

Soft and round, like worn wood and picks: pills for buttons and nav CTA, 16px cards, 24px panels and the CTA band, 12px beat cells in the stage, 8px pattern cells. Bars in waveforms are fully rounded.

## Components

- **Buttons**: pill-shaped. Press feedback is `scale(0.97)` over 160ms with a strong ease-out. Hover only under `(hover: hover)`. Primary = ember on night text; secondary = translucent smoke outline.
- **Upload form**: dashed drop zone on night inside a coal card, keyboard-focusable (the file input is `sr-only`; the zone shows the focus ring). Errors are `role="alert"` with an icon.
- **Progress**: percentage, a bar that animates `transform: scaleX`, and three named stages (Виділення гітари → Аналіз ритму → Підбір патерну) as done / active / to-do.
- **Pattern display**: equal-width step cells; strums use ↓ ember, ↑ amber, × red, · rest; arpeggios use string numbers with an ember **B** bass cell. Meaning never relies on colour alone.
- **Result**: the top recommendation in a coal card with an ember border, plus alternatives, a stacked per-segment chart, a timeline table and feedback buttons. The result enters once with a 60ms-staggered 8px fade-up.
- **Pipeline stage** (home only): see Overview. Respects `prefers-reduced-motion` by showing the finished state, static.
- **Icons**: one inline SVG family (`components/icons.tsx`), 24px grid, 2px round stroke.

## Do's and Don'ts

- **Do** keep ember for sound-and-action moments; let night and coal carry the rest.
- **Do** label demonstration content as an example («Приклад…»).
- **Do** keep every animation tied to what GPR actually does (separate, find the beat, strike the pattern). Motion stays at or under ~600ms except the stage loop.
- **Don't** add eyebrows or kickers above headings; number sequential steps inline in the heading instead.
- **Don't** build sections as rows of same-size cards; connect sequential content into one band.
- **Don't** use gradient text, glass blur, or decorative glow outside the stage.
- **Don't** use pure white or blue-black; text is smoke, ground is night.
