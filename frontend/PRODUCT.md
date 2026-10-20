# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Beginner guitarists: people who know a few chords and want to play a song they like, but don't know how to strum or pick it. They open GPR as often on a phone (guitar in hand, during practice) as on a laptop at a desk.

Secondary audience: the diploma committee reviewing GPR as a thesis project (an ML pipeline behind a web app).

## Product Purpose

The user uploads an audio or video recording of a song; GPR tells them which accompaniment pattern fits it and shows how to play that pattern. There is no single correct way to play a song, so GPR recommends the best fit, or a few likely options when the model is unsure.

Success: a beginner leaves knowing which pattern to try and can read it well enough to start playing.

## Positioning

GPR listens to the actual recording: it separates the guitar stem (HTDemucs), finds the beat grid (BeatThis) and classifies 6-second segments with a LightGBM model, then shows how the pattern changes over the track. It recommends a playing pattern, not chords or tabs.

## Operating Context

- Input: one file, up to 100 MB: MP3, WAV, OGG, FLAC, M4A, MP4, MOV (phone recordings included).
- Processing takes from tens of seconds to a few minutes on a free CPU server, with stages: guitar separation → rhythm analysis → pattern classification.
- Output: the recommended pattern with confidence, alternatives when unsure, a per-segment timeline and chart, and a 👍/👎 rating that helps improve the model.
- A reference page lists all six patterns.

## Capabilities and Constraints

- Six classes. Strumming (бої): 6b Класичний бій, 6x Бій з заглушуванням, 8b Розгорнутий бій. Arpeggio: 6a Шестидольне, 8a Восьмидольне, wa Вальсове арпеджіо.
- Pattern notation: strum steps ↓ down, ↑ up, × muted, · rest over a 1 & 2 & 3 & 4 & count; arpeggio steps are string numbers 1–6 with B for the bass note, a stacked column for a chord.
- One confidence above 75% shows a single recommendation; otherwise up to three options at 25% or more.
- UI language: Ukrainian.
- No URL input, accounts or history.

## Brand Commitments

- Name: GPR — Guitar Pattern Recommender.
- Copy voice: plain Ukrainian, addresses the user as «ви».

## Evidence on Hand

- No testimonials, user counts or accuracy claims are approved for the UI. The research figure (94.7% leave-one-out) is not to be shown as a product claim without the user's approval.
- No photography or illustration assets exist.

## Product Principles

1. Teach, don't just label: every recommendation shows how to play it.
2. Honest uncertainty: when the model isn't sure, say so and show the options.
3. Works with a guitar in hand: readable at a glance on a phone.
4. Encouraging tone for beginners; no jargon without explanation.

## Accessibility & Inclusion

Keyboard-operable upload and controls; pattern notation must not rely on colour alone (arrows and symbols carry meaning).
