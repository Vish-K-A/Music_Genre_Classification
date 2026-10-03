from __future__ import annotations

import html
import logging
import math
from pathlib import Path
from typing import Any

import gradio as gr

from .baseline import BASELINE_MODEL_NAMES, GENRE_ORDER, predict_baselines_from_audio
from .predict import predict_genre as predict_songnet_genre


LOGGER = logging.getLogger(__name__)
GENRES = tuple(GENRE_ORDER)

GENRE_VISUALS = {
    "Rock": {
        "accent": "#783d4c",
        "secondary": "#bd858b",
        "mood": "GUITAR / LIVE ROOM",
        "art": '<path d="M23 78 76 25l19 19-53 53-19 3z"/><path d="m68 33 19 19M58 43l19 19M48 53l19 19M38 63l19 19"/><path d="m78 23 8-8 19 19-8 8M20 85l15 15"/><path d="M88 68v25M99 58v35M110 75v18"/>',
    },
    "Instrumental": {
        "accent": "#8f654e",
        "secondary": "#d19b91",
        "mood": "PIANO / CHAMBER",
        "art": '<path d="M20 28h80v64H20z"/><path d="M20 43h80M31 28v15M43 28v15M55 28v15M67 28v15M79 28v15M91 28v15"/><path d="M36 43v25M60 43v25M84 43v25"/><path d="M30 81h60"/>',
    },
    "Electronic": {
        "accent": "#66527e",
        "secondary": "#b58ba9",
        "mood": "SYNTH / WAVE",
        "art": '<path d="M14 61h15l8-24 13 49 13-66 13 58 9-29 8 12h13"/><path d="M15 94h90M15 25h90" stroke-dasharray="3 8"/>',
    },
    "Hip-Hop": {
        "accent": "#754b60",
        "secondary": "#bf827f",
        "mood": "MIC / RHYTHM",
        "art": '<rect x="45" y="16" width="30" height="54" rx="15"/><path d="M35 54v8a25 25 0 0 0 50 0v-8M60 87v17M43 104h34M52 32h16M52 43h16M52 54h16"/>',
    },
    "Folk": {
        "accent": "#a45f49",
        "secondary": "#d49b7c",
        "mood": "STRINGS / ROOTS",
        "art": '<path d="M56 55 34 33l10-10 23 22M50 60l18 18M39 71l-9 9M58 48l10-10M68 78l9-9"/><path d="M40 55c-9-3-19 4-20 14-1 8 5 16 13 17 8 1 13-5 15-11 3 7 11 11 18 8 8-3 12-13 8-21-3-7-11-11-19-8"/><circle cx="55" cy="68" r="5"/><path d="m75 22 24 24M80 17l5-5M94 31l5-5"/>',
    },
    "International": {
        "accent": "#53677c",
        "secondary": "#bb8594",
        "mood": "WORLD / ENSEMBLE",
        "art": '<circle cx="60" cy="60" r="43"/><path d="M17 60h86M60 17c13 12 20 27 20 43s-7 31-20 43c-13-12-20-27-20-43s7-31 20-43Z"/><path d="M25 39h70M25 81h70"/>',
    },
    "Pop": {
        "accent": "#a35d79",
        "secondary": "#dc9992",
        "mood": "HEADPHONES / STUDIO",
        "art": '<path d="M22 63v-8a38 38 0 0 1 76 0v8"/><rect x="17" y="57" width="18" height="31" rx="8"/><rect x="85" y="57" width="18" height="31" rx="8"/><path d="M35 84c9 12 24 16 38 13"/><path d="M73 97h12"/>',
    },
    "Experimental": {
        "accent": "#725981",
        "secondary": "#bd899f",
        "mood": "FORM / SOUND",
        "art": '<path d="m60 15 14 31 31 14-31 14-14 31-14-31-31-14 31-14z"/><circle cx="60" cy="60" r="12"/><path d="M17 17h18v18H17zM85 85h18v18H85z"/><path d="M87 22h12M93 16v12M22 91h12"/>',
    },
}

DEFAULT_WASH = ("#f7e2e7", "#d9afb2", "#a37f70")
GENRE_WASHES = {
    "Rock": ("#e4cbd0", "#bd9298", "#74524e"),
    "Instrumental": ("#f4e4d7", "#e7c5bf", "#b58e6d"),
    "Electronic": ("#e8dfed", "#c9a9c0", "#78627f"),
    "Hip-Hop": ("#e3c4ca", "#b88d82", "#675060"),
    "Folk": ("#f2d9c9", "#d39e83", "#8c6451"),
    "International": ("#edd9dc", "#b8b8c5", "#8e756a"),
    "Pop": ("#f4dce4", "#efc4b6", "#a797b4"),
    "Experimental": ("#ead3df", "#c2acc9", "#806d83"),
}

MODEL_ICONS = {
    "K-Nearest Neighbors": '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="6" cy="6" r="2"/><circle cx="18" cy="7" r="2"/><circle cx="8" cy="18" r="2"/><circle cx="18" cy="17" r="2"/><path d="m8 7 8 0M7 8l1 8m2 1 6-1m2-7v6"/></svg>',
    "Logistic Regression": '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 19V5m0 14h16M6 16l4-5 3 2 5-7"/><circle cx="18" cy="6" r="1.5"/></svg>',
    "Linear Support Vector Machine": '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 19 20 5M5 7l2 2m-3 3 2 2m11-2 2 2m-3 3 2 2"/><circle cx="7" cy="6" r="1.5"/><circle cx="17" cy="18" r="1.5"/></svg>',
    "Multi-Layer Perceptron": '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="5" cy="6" r="1.5"/><circle cx="5" cy="12" r="1.5"/><circle cx="5" cy="18" r="1.5"/><circle cx="12" cy="8" r="1.5"/><circle cx="12" cy="16" r="1.5"/><circle cx="19" cy="12" r="1.5"/><path d="m6.5 6.5 4 1m-4 4 4-3m-4 4 4 3m-4 4 4-3m3-7 4 3m-4 5 4-3"/></svg>',
}


APP_CSS = r"""
:root {
  color-scheme: light;
  --plum: #3e2d49;
  --plum-soft: #62475f;
  --blue-ink: #39465d;
  --brown-ink: #71574d;
  --muted-ink: #796b70;
  --hairline: rgba(75, 48, 66, .15);
  --paper: rgba(255, 250, 248, .72);
  --paper-strong: rgba(255, 250, 248, .88);
}

html, body { min-height: 100%; background: #ead0d2 !important; }
body, .gradio-container {
  color: var(--plum) !important;
  font-family: 'Aptos', 'Segoe UI', sans-serif !important;
}
body { background-image: linear-gradient(145deg, #f8e8eb, #d9b3b4 58%, #a88776) !important; }
.gradio-container {
  position: relative;
  isolation: isolate;
  min-height: 100vh;
  max-width: none !important;
  padding: 0 !important;
  background: transparent !important;
}

#theme-atmosphere {
  position: fixed;
  inset: 0;
  z-index: 0 !important;
  width: 100vw !important;
  height: 100vh !important;
  margin: 0 !important;
  padding: 0 !important;
  overflow: hidden !important;
  pointer-events: none !important;
}

.theme-layer {
  position: absolute;
  inset: 0;
  opacity: 0;
  pointer-events: none;
  background-image:
    url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 360 240'%3E%3Cg fill='none' stroke='%23563e59' stroke-opacity='.12' stroke-width='1.4'%3E%3Cpath d='M-20 64c46-25 91-25 137 0s91 25 137 0 80-24 126 0'/%3E%3Cpath d='M-20 72c46-25 91-25 137 0s91 25 137 0 80-24 126 0'/%3E%3Cpath d='M-20 80c46-25 91-25 137 0s91 25 137 0 80-24 126 0'/%3E%3Cpath d='M188 176c36-21 69-21 105 0s69 21 105 0'/%3E%3C/g%3E%3Cg fill='%23563e59' fill-opacity='.12'%3E%3Cellipse cx='44' cy='60' rx='6' ry='4' transform='rotate(-24 44 60)'/%3E%3Cpath d='M49 59V24l27-6v34h-3V22l-21 5v31z'/%3E%3Cellipse cx='294' cy='173' rx='6' ry='4' transform='rotate(-24 294 173)'/%3E%3Cpath d='M299 172v-32l27-6v31h-3v-27l-21 5v29z'/%3E%3C/g%3E%3C/svg%3E"),
    radial-gradient(ellipse at 12% 4%, rgba(255,255,255,.78), transparent 34%),
    radial-gradient(ellipse at 88% 82%, rgba(103,70,65,.14), transparent 43%),
    linear-gradient(128deg, var(--wash-a) 0%, var(--wash-b) 54%, var(--wash-c) 100%);
  background-size: 420px 280px, auto, auto, auto;
  background-repeat: repeat, no-repeat, no-repeat, no-repeat;
  transition: opacity 1550ms cubic-bezier(.22,.7,.2,1);
  will-change: opacity;
}
.theme-layer[data-theme="default"] { opacity: 1; }
.theme-watermark {
  position: absolute;
  width: min(48vw, 540px);
  height: min(48vw, 540px);
  right: 5vw;
  top: 16vh;
  color: var(--theme-ink);
  fill: none;
  stroke: currentColor;
  stroke-width: 1.25;
  stroke-linecap: round;
  stroke-linejoin: round;
  opacity: .055;
  transform: rotate(7deg);
}
.gradio-container:has(.report[data-genre]) #theme-atmosphere .theme-layer[data-theme="default"] { opacity: 0; }
.gradio-container:has(.report[data-genre="Rock"]) #theme-atmosphere .theme-layer[data-theme="Rock"],
.gradio-container:has(.report[data-genre="Instrumental"]) #theme-atmosphere .theme-layer[data-theme="Instrumental"],
.gradio-container:has(.report[data-genre="Electronic"]) #theme-atmosphere .theme-layer[data-theme="Electronic"],
.gradio-container:has(.report[data-genre="Hip-Hop"]) #theme-atmosphere .theme-layer[data-theme="Hip-Hop"],
.gradio-container:has(.report[data-genre="Folk"]) #theme-atmosphere .theme-layer[data-theme="Folk"],
.gradio-container:has(.report[data-genre="International"]) #theme-atmosphere .theme-layer[data-theme="International"],
.gradio-container:has(.report[data-genre="Pop"]) #theme-atmosphere .theme-layer[data-theme="Pop"],
.gradio-container:has(.report[data-genre="Experimental"]) #theme-atmosphere .theme-layer[data-theme="Experimental"] { opacity: 1; }

.gradio-container > .main { position: relative; z-index: 1; }
main { width: 100% !important; max-width: none !important; margin: 0 !important; }
footer { display: none !important; }
#app-shell { position: relative; z-index: 2; width: min(1180px, calc(100% - 52px)); margin: 0 auto; padding: 38px 0 64px; gap: 0; }

#masthead { display:flex; align-items:center; justify-content:space-between; gap:24px; padding:0 0 25px; border-bottom:1px solid var(--hairline); animation: soft-rise .7s ease both; }
.brand-lockup { display:flex; align-items:center; gap:15px; }
.brand-mark { width:44px; height:44px; display:grid; place-items:center; color:var(--plum); background:rgba(255,250,248,.55); border:1px solid rgba(92,61,83,.22); border-radius:50%; box-shadow:0 6px 18px rgba(92,54,69,.08); }
.brand-mark svg { width:26px; height:26px; }
.wordmark { color:var(--plum); font:600 11px 'Aptos', 'Segoe UI', sans-serif; letter-spacing:.2em; }
.title-block h1 { margin:5px 0 4px; color:var(--plum); font:500 34px/1.08 Georgia, 'Times New Roman', serif; }
.title-block p { margin:0; color:var(--blue-ink); font-size:14px; }
.header-aside { color:var(--brown-ink); text-align:right; font:11px Georgia, 'Times New Roman', serif; line-height:1.7; letter-spacing:.06em; }
.header-aside strong { color:var(--plum); font-weight:600; }

#studio-row { margin:22px 0 28px; gap:17px; align-items:stretch; }
#upload-column, #action-column { min-width:0; }
#track-upload { overflow:hidden; border:1px solid rgba(90,61,72,.18); border-radius:8px; background:rgba(255,250,248,.68) !important; box-shadow:0 12px 30px rgba(96,62,66,.08); }
#track-upload label { padding:5px 9px; border-radius:6px; color:var(--plum) !important; background:rgba(255,250,248,.94) !important; font:600 10px 'Aptos', 'Segoe UI', sans-serif !important; letter-spacing:.12em; }
#track-upload [data-testid="audio"] { border-radius:7px; background:rgba(255,255,255,.28) !important; }
#track-upload button, #track-upload [role="button"] { color:var(--plum) !important; }
#action-column { justify-content:space-between; padding:4px 0; }
.input-note { min-height:70px; display:flex; align-items:center; gap:13px; padding:4px 0 4px 17px; border-left:1px solid var(--hairline); }
.input-note svg { width:29px; height:29px; flex:none; color:var(--plum-soft); }
.input-note b { display:block; color:var(--plum); font:600 10px 'Aptos', 'Segoe UI', sans-serif; letter-spacing:.12em; }
.input-note span { display:block; margin-top:4px; color:var(--brown-ink); font-size:12px; }
#analyze-button { width:100%; min-height:52px; border:0 !important; border-radius:8px !important; color:#fff8f5 !important; background:linear-gradient(120deg,#98677b,#704e69) !important; box-shadow:0 8px 20px rgba(89,52,73,.18); font:600 11px 'Aptos', 'Segoe UI', sans-serif !important; letter-spacing:.15em !important; transition:transform .2s ease, box-shadow .3s ease, background .35s ease !important; }
#analyze-button:hover { background:linear-gradient(120deg,#a87988,#795873) !important; box-shadow:0 12px 24px rgba(89,52,73,.2); }
#analyze-button:active { transform:translateY(0); }

#analysis-output { margin-top:2px; }
.empty-state, .error-state, .hero-result, .detail-panel, .comparison { border:1px solid rgba(91,61,73,.13); border-radius:8px; background:var(--paper); box-shadow:0 14px 38px rgba(93,60,62,.09); backdrop-filter:blur(7px); }
.empty-state, .error-state { min-height:235px; display:grid; grid-template-columns:1fr auto; align-items:center; gap:28px; padding:30px 36px; }
.empty-copy .eyebrow, .error-copy .eyebrow, .section-eyebrow { color:var(--plum-soft); font:600 10px 'Aptos', 'Segoe UI', sans-serif; letter-spacing:.17em; text-transform:uppercase; }
.empty-copy h2, .error-copy h2 { margin:11px 0 7px; color:var(--plum); font:500 30px/1.12 Georgia, 'Times New Roman', serif; }
.empty-copy p, .error-copy p { max-width:530px; margin:0; color:var(--blue-ink); font-size:14px; line-height:1.6; }
.idle-visual { width:220px; height:120px; display:flex; align-items:center; justify-content:center; gap:6px; border-left:1px solid var(--hairline); }
.idle-visual i { width:4px; height:var(--h); display:block; border-radius:4px; background:linear-gradient(0deg,#8f6679,#c58d91); opacity:.62; animation: idle-breathe 1.8s ease-in-out infinite alternate; animation-delay:var(--d); }
.error-state { border-color:rgba(128,70,78,.24); }
.error-copy .eyebrow { color:#8e4d5e; }
.error-detail { margin-top:10px !important; color:#74565c !important; font:12px Consolas, monospace; overflow-wrap:anywhere; }

.report { --genre-accent:#76536e; --genre-secondary:#bd858b; position:relative; display:grid; gap:15px; animation:soft-rise .55s ease both; }
.report-head { display:flex; align-items:center; justify-content:space-between; gap:15px; padding:0 2px 2px; color:var(--brown-ink); }
.track-name { max-width:62%; overflow:hidden; color:var(--plum-soft); font:12px Georgia, 'Times New Roman', serif; text-overflow:ellipsis; white-space:nowrap; }
.live-tag { display:inline-flex; align-items:center; gap:7px; color:var(--brown-ink); font:600 9px 'Aptos', 'Segoe UI', sans-serif; letter-spacing:.1em; }
.live-dot { width:7px; height:7px; border-radius:50%; background:var(--genre-accent); box-shadow:0 0 0 4px color-mix(in srgb,var(--genre-accent) 12%,transparent); }
.hero-result { position:relative; overflow:hidden; display:grid; grid-template-columns:minmax(0,1.15fr) minmax(235px,.85fr); align-items:center; gap:26px; min-height:238px; padding:25px 30px; }
.result-label { color:var(--plum-soft); font:600 10px 'Aptos', 'Segoe UI', sans-serif; letter-spacing:.15em; }
.primary-genre { margin:10px 0 16px; color:var(--plum); font:500 62px/.98 Georgia, 'Times New Roman', serif; letter-spacing:0; text-wrap:balance; }
.primary-meta { display:flex; flex-wrap:wrap; gap:8px; color:var(--plum); }
.model-chip { display:inline-flex; align-items:center; gap:7px; padding:7px 10px; border:1px solid rgba(88,58,72,.17); border-radius:6px; background:rgba(255,255,255,.28); color:var(--plum); font:600 9px 'Aptos', 'Segoe UI', sans-serif; letter-spacing:.05em; }
.model-chip em { width:7px; height:7px; border-radius:50%; background:var(--genre-accent); }
.consensus-chip { border-color:color-mix(in srgb,var(--genre-secondary) 48%,transparent); }
.consensus-chip em { background:var(--genre-secondary); }
.genre-scene { position:relative; min-height:180px; overflow:hidden; display:flex; align-items:center; justify-content:space-between; padding:17px 19px; border:1px solid color-mix(in srgb,var(--genre-accent) 23%,transparent); border-radius:8px; background:radial-gradient(ellipse at 80% 20%,color-mix(in srgb,var(--genre-secondary) 23%,transparent),transparent 48%),linear-gradient(135deg,rgba(255,252,249,.48),color-mix(in srgb,var(--genre-accent) 10%,#f5e8e4)); }
.genre-scene::after { content:''; position:absolute; inset:auto 0 0; height:1px; background:linear-gradient(90deg,transparent,var(--genre-accent),transparent); opacity:.45; }
.scene-caption { position:relative; z-index:1; align-self:flex-start; color:var(--plum-soft); font:600 9px 'Aptos', 'Segoe UI', sans-serif; letter-spacing:.11em; }
.genre-glyph { width:122px; height:122px; display:grid; place-items:center; color:var(--genre-accent); }
.genre-glyph svg { width:108px; height:108px; fill:none; stroke:currentColor; stroke-width:2; stroke-linecap:round; stroke-linejoin:round; opacity:.9; animation:instrument-arrive .8s cubic-bezier(.2,.75,.2,1) both; }
.scene-index { position:absolute; bottom:13px; left:17px; color:var(--brown-ink); font:9px Consolas,monospace; letter-spacing:.06em; }

.comparison { padding:18px 20px 20px; }
.comparison-head { display:flex; align-items:baseline; justify-content:space-between; gap:14px; margin-bottom:12px; }
.comparison-head h3, .detail-heading { margin:0; color:var(--plum); font:500 17px Georgia, 'Times New Roman', serif; }
.comparison-head span { color:var(--brown-ink); font:600 9px 'Aptos', 'Segoe UI', sans-serif; letter-spacing:.08em; }
.model-grid { display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:8px; }
.model-cell { min-width:0; min-height:72px; padding:11px 12px; border:1px solid rgba(99,67,77,.11); border-radius:7px; background:rgba(255,255,255,.32); animation:soft-rise .5s ease both; animation-delay:var(--delay); }
.model-cell:first-child { background:color-mix(in srgb,var(--genre-secondary) 10%,rgba(255,255,255,.35)); }
.model-name { min-height:24px; color:var(--muted-ink); font:600 9px/1.4 'Aptos', 'Segoe UI', sans-serif; letter-spacing:.035em; text-transform:uppercase; }
.model-prediction { margin-top:7px; color:var(--plum); font:500 13px Georgia, 'Times New Roman', serif; line-height:1.25; overflow-wrap:anywhere; }

.detail-grid { display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:15px; }
.detail-panel { min-width:0; padding:21px 22px; }
.detail-top { display:flex; align-items:baseline; justify-content:space-between; gap:11px; margin-bottom:17px; }
.detail-kicker { color:var(--plum-soft); font:600 9px 'Aptos', 'Segoe UI', sans-serif; letter-spacing:.13em; text-transform:uppercase; }
.detail-heading { margin-top:5px; font-size:20px; }
.detail-side-note { color:var(--brown-ink); font:600 9px 'Aptos', 'Segoe UI', sans-serif; letter-spacing:.08em; text-align:right; }
.score-list { display:grid; gap:11px; }
.score-row { animation:soft-rise .45s ease both; animation-delay:var(--delay); }
.score-head { display:flex; align-items:baseline; justify-content:space-between; gap:10px; margin-bottom:5px; color:#000 !important; font-size:11px; }
.score-head > span { color:#000 !important; }
.vote-label span { color:#000 !important; }
.score-number { color:var(--plum); font:600 10px Consolas,monospace; font-variant-numeric:tabular-nums; }
.score-track, .vote-track { height:5px; overflow:hidden; border-radius:6px; background:rgba(92,67,75,.10); }
.score-fill { display:block; width:var(--score-width); height:100%; border-radius:inherit; background:linear-gradient(90deg,var(--genre-accent),var(--genre-secondary)); transform-origin:left; animation:bar-arrive .9s cubic-bezier(.2,.8,.2,1) both; animation-delay:var(--delay); }
.baseline-list { display:grid; }
.baseline-row { display:flex; align-items:center; gap:10px; padding:9px 0; border-bottom:1px solid rgba(88,61,72,.10); }
.baseline-row:last-child { border-bottom:0; }
.model-icon { width:24px; height:24px; display:grid; place-items:center; flex:none; color:var(--genre-accent); }
.model-icon svg { width:20px; height:20px; fill:none; stroke:currentColor; stroke-width:1.7; stroke-linecap:round; stroke-linejoin:round; }
.baseline-name { flex:1; color:var(--blue-ink); font-size:11px; line-height:1.35; }
.baseline-prediction { color:var(--plum); font:600 12px Georgia, 'Times New Roman', serif; text-align:right; }
.ensemble { margin-top:13px; padding:13px 14px; border:1px solid color-mix(in srgb,var(--genre-secondary) 38%,transparent); border-radius:7px; background:color-mix(in srgb,var(--genre-secondary) 9%,rgba(255,250,248,.6)); }
.ensemble-head { display:flex; align-items:baseline; justify-content:space-between; gap:9px; margin-bottom:10px; }
.ensemble-head span:first-child { color:var(--plum-soft); font:600 9px 'Aptos', 'Segoe UI', sans-serif; letter-spacing:.1em; }
.ensemble-head strong { color:var(--plum); font:600 15px Georgia, 'Times New Roman', serif; }
.vote-list { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px 13px; }
.vote-item { min-width:0; }
.vote-label { display:flex; justify-content:space-between; gap:6px; margin-bottom:4px; color:var(--blue-ink); font-size:9px; }
.vote-label b { color:var(--plum); font:600 9px Consolas,monospace; }
.vote-track { height:3px; }
.vote-fill { display:block; width:var(--vote-width); height:100%; border-radius:inherit; background:var(--genre-secondary); }
.report-foot { display:flex; justify-content:space-between; gap:12px; padding:0 2px; color:var(--brown-ink); font:600 9px 'Aptos', 'Segoe UI', sans-serif; letter-spacing:.08em; text-transform:uppercase; }

@keyframes soft-rise { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }
@keyframes bar-arrive { from { transform:scaleX(0); } to { transform:scaleX(1); } }
@keyframes instrument-arrive { from { opacity:0; transform:translateY(7px) rotate(-4deg); } to { opacity:.9; transform:translateY(0) rotate(0); } }
@keyframes idle-breathe { from { transform:scaleY(.48); opacity:.35; } to { transform:scaleY(1); opacity:.7; } }

@media (max-width: 850px) {
  #app-shell { width:min(100% - 34px,700px); padding-top:25px; }
  #masthead { align-items:flex-start; }
  .header-aside { display:none; }
  #studio-row { flex-direction:column; }
  #action-column { gap:13px; }
  .input-note { min-height:48px; }
  .empty-state, .error-state { min-height:205px; grid-template-columns:1fr; padding:25px; }
  .idle-visual { display:none; }
  .hero-result { grid-template-columns:1fr; gap:18px; }
  .genre-scene { min-height:145px; }
  .model-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
  .model-cell:last-child { grid-column:1 / -1; }
}
@media (max-width: 560px) {
  #app-shell { width:calc(100% - 24px); padding:18px 0 34px; }
  #masthead { padding-bottom:17px; }
  .brand-mark { width:38px; height:38px; }
  .title-block h1 { max-width:300px; font-size:27px; line-height:1.08; }
  .title-block p { font-size:12px; }
  .hero-result { min-height:0; padding:21px 18px; }
  .primary-genre { font-size:45px; }
  .genre-scene { min-height:126px; padding:13px; }
  .genre-glyph { width:94px; height:94px; }
  .genre-glyph svg { width:86px; height:86px; }
  .comparison { padding:15px; }
  .comparison-head { align-items:flex-start; flex-direction:column; gap:5px; }
  .model-cell { min-height:72px; padding:9px; }
  .model-name { font-size:8px; }
  .model-prediction { font-size:12px; }
  .detail-grid { grid-template-columns:1fr; }
  .detail-panel { padding:18px; }
  .report-foot { flex-direction:column; gap:5px; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation-duration:.01ms !important; animation-iteration-count:1 !important; scroll-behavior:auto !important; }
}
"""


def _svg_icon() -> str:
    return (
        '<svg viewBox="0 0 48 48" aria-hidden="true">'
    '<path d="M11 31V13l20-4v19" fill="none" stroke="currentColor" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round"/>'
    '<ellipse cx="8" cy="32" rx="5" ry="3.5" transform="rotate(-18 8 32)" fill="currentColor"/>'
    '<ellipse cx="27" cy="29" rx="5" ry="3.5" transform="rotate(-18 27 29)" fill="currentColor"/>'
    '<path d="M35 17c3 2 4 5 4 8m-7-5c2 1 2 3 2 5" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/>'
        "</svg>"
    )


def _empty_state() -> str:
    bars = "".join(
        f'<i style="--h:{height}px;--d:{index * 75}ms"></i>'
        for index, height in enumerate((19, 34, 57, 42, 77, 48, 28, 62, 38, 70, 31, 53, 23))
    )
    return f"""
    <section class="empty-state" aria-live="polite">
      <div class="empty-copy">
        <div class="eyebrow">SIGNAL CHAIN / READY</div>
        <h2>Bring a track into focus.</h2>
        <p>SONGNET will read the same audio through two independent model pipelines.</p>
      </div>
      <div class="idle-visual" aria-hidden="true">{bars}</div>
    </section>
    """


def _error_state(message: str) -> str:
    safe_message = html.escape(message[:260])
    return f"""
    <section class="error-state" role="alert" aria-live="assertive">
      <div class="error-copy">
        <div class="eyebrow">ANALYSIS INTERRUPTED</div>
        <h2>This track could not be analyzed.</h2>
        <p>Check the audio file and try again.</p>
        <p class="error-detail">{safe_message}</p>
      </div>
    </section>
    """


def _render_results(
    audio_path: str,
    songnet_genre: str,
    songnet_scores: dict[str, float],
    baseline_result: dict[str, Any],
) -> str:
    if songnet_genre not in GENRES:
        raise ValueError("SongNet returned an unsupported genre.")
    if set(songnet_scores) != set(GENRES):
        raise ValueError("SongNet did not return scores for all eight genres.")

    scores = {genre: float(songnet_scores[genre]) for genre in GENRES}
    if any(not math.isfinite(score) or score < 0 for score in scores.values()):
        raise ValueError("SongNet returned invalid genre scores.")

    predictions = baseline_result.get("model_predictions", {})
    if set(predictions) != set(BASELINE_MODEL_NAMES):
        raise ValueError("Classical ML did not return predictions from all four models.")
    if any(predictions[name] not in GENRES for name in BASELINE_MODEL_NAMES):
        raise ValueError("Classical ML returned an unsupported genre.")

    consensus = baseline_result.get("consensus_prediction")
    if consensus not in GENRES:
        raise ValueError("Classical ML returned an unsupported consensus genre.")

    vote_counts = baseline_result.get("vote_counts", {})
    if set(vote_counts) != set(GENRES):
        raise ValueError("Classical ML did not return vote counts for all genres.")

    visual = GENRE_VISUALS[songnet_genre]
    safe_track_name = html.escape(Path(audio_path).name)
    svg = f'<svg viewBox="0 0 120 120" role="img" aria-label="{songnet_genre} visual">{visual["art"]}</svg>'

    score_rows = []
    for index, genre in enumerate(GENRES):
        score = scores[genre]
        score_rows.append(
            f"""
            <div class="score-row" style="--delay:{index * 55}ms">
              <div class="score-head"><span>{html.escape(genre)} — <span class="score-number">{score * 100:.2f}%</span></span></div>
              <div class="score-track" role="meter" aria-label="{html.escape(genre)} score" aria-valuemin="0" aria-valuemax="1" aria-valuenow="{score:.8f}">
                <span class="score-fill" style="--score-width:{score * 100:.6f}%;--delay:{index * 55}ms"></span>
              </div>
            </div>
            """
        )

    comparison_cells = [
        f"""
        <div class="model-cell" style="--delay:{index * 45}ms">
          <div class="model-name">{html.escape(name)}</div>
          <div class="model-prediction">{html.escape(predictions[name])}</div>
        </div>
        """
        for index, name in enumerate(BASELINE_MODEL_NAMES, start=1)
    ]
    comparison_cells.insert(
        0,
        f"""
        <div class="model-cell" style="--delay:0ms">
          <div class="model-name">SongNet / Deep Learning</div>
          <div class="model-prediction">{html.escape(songnet_genre)}</div>
        </div>
        """,
    )

    baseline_rows = "".join(
        f"""
        <div class="baseline-row">
          <span class="model-icon">{MODEL_ICONS[name]}</span>
          <span class="baseline-name">{html.escape(name)}</span>
          <strong class="baseline-prediction">{html.escape(predictions[name])}</strong>
        </div>
        """
        for name in BASELINE_MODEL_NAMES
    )

    vote_rows = []
    for genre in GENRES:
        count = int(vote_counts[genre])
        if count < 0 or count > len(BASELINE_MODEL_NAMES):
            raise ValueError("Classical ML returned invalid vote counts.")
        vote_rows.append(
            f"""
            <div class="vote-item">
              <div class="vote-label"><span>{html.escape(genre)}</span><b>{count}</b></div>
              <div class="vote-track"><span class="vote-fill" style="--vote-width:{count / len(BASELINE_MODEL_NAMES) * 100:.2f}%"></span></div>
            </div>
            """
        )

    return f"""
    <section class="report" data-genre="{html.escape(songnet_genre)}"
      style="--genre-accent:{visual['accent']};--genre-secondary:{visual['secondary']}" aria-live="polite">
      <div class="report-head">
        <span class="track-name">{safe_track_name}</span>
        <span class="live-tag"><i class="live-dot"></i> CURRENT TRACK / ANALYZED</span>
      </div>

      <div class="hero-result">
        <div>
          <div class="result-label">YOUR RESULT / SONGNET</div>
          <h2 class="primary-genre">{html.escape(songnet_genre)}</h2>
          <div class="primary-meta">
            <span class="model-chip"><em></em> SONGNET: {html.escape(songnet_genre)}</span>
            <span class="model-chip consensus-chip"><em></em> CLASSICAL ENSEMBLE: {html.escape(consensus)}</span>
          </div>
        </div>
        <div class="genre-scene">
          <span class="scene-caption">{html.escape(visual['mood'])}</span>
          <span class="genre-glyph">{svg}</span>
          <span class="scene-index">GENRE SIGNATURE / {GENRES.index(songnet_genre) + 1:02d}</span>
        </div>
      </div>

      <div class="detail-grid">
        <section class="detail-panel" aria-label="SongNet genre scores">
          <div class="detail-top">
            <div><div class="detail-kicker">SONGNET / DEEP LEARNING</div><h3 class="detail-heading">SongNet scores</h3></div>
            <span class="detail-side-note">8 GENRES</span>
          </div>
          <div class="score-list">{''.join(score_rows)}</div>
        </section>

        <section class="detail-panel" aria-label="Classical ML ensemble">
          <div class="detail-top">
            <div><div class="detail-kicker">CLASSICAL MACHINE LEARNING</div><h3 class="detail-heading">Classical ML Ensemble</h3></div>
            <span class="detail-side-note">HARD-LABEL VOTES</span>
          </div>
          <div class="baseline-list">{baseline_rows}</div>
          <div class="ensemble">
            <div class="ensemble-head"><span>ENSEMBLE CONSENSUS</span><strong>{html.escape(consensus)}</strong></div>
            <div class="vote-list">{''.join(vote_rows)}</div>
          </div>
        </section>
      </div>

      <section class="comparison" aria-label="Current model comparison">
        <div class="comparison-head">
          <h3>Model comparison</h3>
          <span>THIS TRACK ONLY</span>
        </div>
        <div class="model-grid">{''.join(comparison_cells)}</div>
      </section>

      <div class="report-foot"><span>FMA-SMALL / 8-GENRE CLASSIFICATION</span><span>INDEPENDENT MODEL OUTPUTS</span></div>
    </section>
    """

def analyze_track(audio_path: str | None) -> str:
    if not audio_path:
        return _error_state("Select an audio file before analysis.")

    try:
        songnet_genre, songnet_scores = predict_songnet_genre(audio_path)
        baseline_result = predict_baselines_from_audio(audio_path)
        return _render_results(audio_path, songnet_genre, songnet_scores, baseline_result)
    except Exception as exc:
        LOGGER.exception("Track analysis failed.")
        return _error_state(f"{type(exc).__name__}: {exc}")


def _theme_layers_html() -> str:
    default_style = ";".join(
        f"--wash-{name}:{color}" for name, color in zip(("a", "b", "c"), DEFAULT_WASH)
    )
    layers = [f'<div class="theme-layer" data-theme="default" style="{default_style}"></div>']
    for genre in GENRES:
        wash_style = ";".join(
            f"--wash-{name}:{color}"
            for name, color in zip(("a", "b", "c"), GENRE_WASHES[genre])
        )
        visual = GENRE_VISUALS[genre]
        watermark = (
            f'<svg class="theme-watermark" viewBox="0 0 120 120" aria-hidden="true" '
            f'style="--theme-ink:{visual["accent"]}">{visual["art"]}</svg>'
        )
        layers.append(
            f'<div class="theme-layer" data-theme="{html.escape(genre)}" '
            f'style="{wash_style}">{watermark}</div>'
        )
    return "".join(layers)


with gr.Blocks(
    analytics_enabled=False,
    title="SONGNET | Real-Time Music Genre Classification",
    fill_width=True,
) as demo:
    gr.HTML(value=_theme_layers_html(), elem_id="theme-atmosphere")

    with gr.Column(elem_id="app-shell"):
        gr.HTML(
            """
            <header id="masthead">
              <div class="brand-lockup">
                <span class="brand-mark">"""
            + _svg_icon()
            + """</span>
                <div class="title-block">
                  <div class="wordmark">SONGNET</div>
                  <h1>Real-Time Music Genre Classification</h1>
                  <p>Deep Learning + Classical ML</p>
                </div>
              </div>
              <div class="header-aside"><strong>ELEGANT AI MUSIC STUDIO</strong><br>FMA-SMALL / 8 GENRES</div>
            </header>
            """,
        )

        with gr.Row(elem_id="studio-row"):
            with gr.Column(scale=7, elem_id="upload-column"):
                audio_input = gr.Audio(
                    sources=["upload"],
                    type="filepath",
                    label="AUDIO INPUT / DROP A TRACK OR BROWSE",
                    elem_id="track-upload",
                )
            with gr.Column(scale=3, elem_id="action-column"):
                gr.HTML(
                    """
                    <div class="input-note">
                      <svg viewBox="0 0 32 32" aria-hidden="true"><rect x="11" y="3" width="10" height="18" rx="5" fill="none" stroke="currentColor" stroke-width="1.7"/><path d="M7 15v2a9 9 0 0 0 18 0v-2M16 26v4m-5 0h10" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/></svg>
                      <div><b>ONE TRACK / TWO LISTENERS</b><span>SongNet + classical ensemble</span></div>
                    </div>
                    """
                )
                analyze_button = gr.Button(
                    "ANALYZE TRACK",
                    elem_id="analyze-button",
                    variant="primary",
                    size="lg",
                )

        results = gr.HTML(value=_empty_state(), elem_id="analysis-output")
        analyze_button.click(
            fn=analyze_track,
            inputs=[audio_input],
            outputs=[results],
            show_progress="minimal",
            concurrency_limit=1,
        )


if __name__ == "__main__":
  demo.launch(
    server_name="127.0.0.1",
    server_port=7860,
    share=False,
    show_error=False,
    theme=gr.themes.Base(),
    css=APP_CSS,
  )