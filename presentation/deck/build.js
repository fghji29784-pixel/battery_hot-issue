// SDM 5분 선별 가능성 검토 — 발표본 생성
const pptx = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

const DIR = "/tmp/claude-0/-home-user-battery-hot-issue/840c3325-79ae-5d86-bfaa-b155630daf7e/scratchpad/deck";
const PNG = path.join(DIR, "png");

// ── 팔레트 — 차트와 같은 색을 쓴다. 안 그러면 슬라이드와 그림이 따로 논다 ──
const C = {
  navy:   "12305B",   // 표지·간지 바탕
  navy2:  "1B4079",
  blue:   "2A78D6",   // 강조 (차트의 --accent 와 동일)
  orange: "EB6834",   // 대비 (차트의 --legacy 와 동일)
  crit:   "D03B3B",
  ink:    "0E1417",
  ink2:   "3D4A52",
  muted:  "6B767E",
  rule:   "DDE4E9",
  bg:     "F4F6F8",
  white:  "FFFFFF",
};
const F = { head: "맑은 고딕", body: "맑은 고딕", mono: "Consolas" };

const P = new pptx();
P.layout = "LAYOUT_WIDE";              // 13.333 x 7.5 인치
P.author = "SDM 분석";
P.title = "SDM 5분 선별 가능성 검토";
const W = 13.333, H = 7.5, M = 0.62;   // 여백

// 그림 크기를 슬라이드 폭에 맞춰 계산한다
const SZ = {};
["sdmconcept","mech","rowprofile","scatter","units","cell140","ranks",
 "growth","snr4","heat","cliff","plan","rulebar"].forEach(n => {
  const b = fs.readFileSync(path.join(PNG, n + ".png"));
  // PNG IHDR: 16바이트 뒤에 폭·높이가 빅엔디안 4바이트씩
  SZ[n] = { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
});
function img(slide, name, opt) {
  const a = SZ[name].h / SZ[name].w;
  const w = opt.w, h = w * a;
  slide.addImage({ path: path.join(PNG, name + ".png"), x: opt.x, y: opt.y, w, h });
  return h;
}

// ── 공통 조각 ──────────────────────────────────────────────
function titleSlide(s, eyebrow, title, sub) {
  s.background = { color: C.navy };
  if (eyebrow) s.addText(eyebrow, { x: M, y: 2.28, w: W - 2 * M, h: 0.34,
    fontFace: F.mono, fontSize: 13, color: C.blue, bold: true, charSpacing: 2, isTextBox: true });
  s.addText(title, { x: M, y: 2.68, w: W - 2 * M, h: 1.5,
    fontFace: F.head, fontSize: 44, bold: true, color: C.white, isTextBox: true });
  if (sub) s.addText(sub, { x: M, y: 4.22, w: W - 2 * M - 1.4, h: 1.2,
    fontFace: F.body, fontSize: 16, color: "AFC4DB", lineSpacing: 26, isTextBox: true });
}

function head(s, tag, title) {
  s.background = { color: C.white };
  s.addText(tag, { x: M, y: 0.42, w: 3.4, h: 0.3, fontFace: F.mono, fontSize: 12,
    color: C.blue, bold: true, charSpacing: 1.4, isTextBox: true, margin: 0 });
  s.addText(title, { x: M, y: 0.76, w: W - 2 * M, h: 0.78, fontFace: F.head,
    fontSize: 27, bold: true, color: C.ink, isTextBox: true, margin: 0 });
}

function note(s, t) { s.addNotes(t); }

// 큰 수치 타일
function stat(s, x, y, w, val, unit, label, color) {
  s.addShape(P.ShapeType.roundRect, { x, y, w, h: 1.5, rectRadius: 0.06,
    fill: { color: C.bg }, line: { color: C.rule, width: 1 } });
  s.addText(label, { x: x + 0.22, y: y + 0.16, w: w - 0.44, h: 0.3,
    fontFace: F.mono, fontSize: 11, color: C.muted, isTextBox: true, margin: 0 });
  s.addText([{ text: val, options: { fontSize: 34, bold: true, color } },
             { text: unit ? " " + unit : "", options: { fontSize: 14, color } }],
    { x: x + 0.22, y: y + 0.52, w: w - 0.44, h: 0.6, fontFace: F.head, isTextBox: true, margin: 0 });
}

// 설명 카드
function card(s, x, y, w, h, title, lines, accent) {
  s.addShape(P.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.05,
    fill: { color: accent ? "EAF2FC" : C.bg }, line: { color: accent ? C.blue : C.rule, width: accent ? 1.5 : 1 } });
  s.addText(title, { x: x + 0.24, y: y + 0.16, w: w - 0.48, h: 0.3, fontFace: F.mono,
    fontSize: 11, bold: true, color: accent ? C.blue : C.muted, isTextBox: true, margin: 0 });
  s.addText(lines.map((t, i) => ({ text: t,
    options: { bullet: lines.length > 1, breakLine: i < lines.length - 1 } })),
    { x: x + 0.24, y: y + 0.52, w: w - 0.48, h: h - 0.7, fontFace: F.body, fontSize: 13,
      color: C.ink2, lineSpacing: 20, paraSpaceAfter: 6, isTextBox: true, margin: 0, valign: "top" });
}

// ═══════════════ 1. 표지 ═══════════════
{
  const s = P.addSlide();
  titleSlide(s, "SDM 데이터 분석 · 중간 보고",
    "SDM 5분 선별 가능성 검토",
    "SDM 전류에 섞인 교란의 정체를 밝히고, 측정 시간을 5분까지 줄여도\n불량을 전부 골라낼 수 있는지 확인했다.");
  s.addText("Keysight BT2152B   ·   9트레이 1,294셀   ·   불량(E) 4개", {
    x: M, y: 6.3, w: W - 2 * M, h: 0.34, fontFace: F.mono, fontSize: 12,
    color: "7E97B5", isTextBox: true });
  note(s, "SDM 전류를 그대로 쓰면 안 된다는 것을 1부에서 보이고, 그 상태에서 5분 측정이 가능한지를 2부에서 판단합니다. 3부는 앞으로 할 일입니다.");
}

// ═══════════════ 2. 요약 ═══════════════
{
  const s = P.addSlide();
  head(s, "요약", "먼저 결론부터");
  stat(s, M, 1.72, 2.9, "20", "위", "5분 측정 · 마지막 불량 순위", C.blue);
  stat(s, M + 3.06, 1.72, 2.9, "1,232", "위", "보정 없이 전류만 쓸 때", C.orange);
  stat(s, M + 6.12, 1.72, 2.9, "1.24", "%", "같이 걸리는 양품 (수율)", C.ink);
  stat(s, M + 9.18, 1.72, 2.9, "3분", "", "여기서는 안 된다", C.crit);

  s.addText([
    { text: "1부  ", options: { bold: true, color: C.blue } },
    { text: "SDM 전류에는 ‘식는 속도’가 섞여 있다. 그래서 음수 전류가 나오고, 불량인데 양품보다 낮게 읽히는 셀이 생긴다.", options: {} },
  ], { x: M, y: 3.62, w: W - 2 * M, h: 0.46, fontFace: F.body, fontSize: 15, color: C.ink2, isTextBox: true, margin: 0 });
  s.addText([
    { text: "2부  ", options: { bold: true, color: C.blue } },
    { text: "같은 행 12셀과 비교해 기울기로 읽으면, 5분 측정으로 1,294셀 중 상위 20위 안에 불량 4개가 전부 들어온다.", options: {} },
  ], { x: M, y: 4.22, w: W - 2 * M, h: 0.46, fontFace: F.body, fontSize: 15, color: C.ink2, isTextBox: true, margin: 0 });
  s.addText([
    { text: "3부  ", options: { bold: true, color: C.blue } },
    { text: "공정 데이터를 결합하고 다른 로트에서 재현해, ‘가능해 보인다’를 ‘가능하다’로 바꾼다.", options: {} },
  ], { x: M, y: 4.82, w: W - 2 * M, h: 0.46, fontFace: F.body, fontSize: 15, color: C.ink2, isTextBox: true, margin: 0 });

  s.addShape(P.ShapeType.roundRect, { x: M, y: 5.62, w: W - 2 * M, h: 1.02, rectRadius: 0.05,
    fill: { color: "FDF3EF" }, line: { color: C.orange, width: 1.2 } });
  s.addText("이 검토는 SDM 전류 곡선과 셀 번호만 썼다. 공정 데이터를 함께 쓰는 기존 15분 판정 로직과 경쟁하는 것이 아니라, 더 짧은 구간에서 SDM 단독으로 어디까지 되는지를 본 것이다.",
    { x: M + 0.26, y: 5.78, w: W - 2 * M - 0.52, h: 0.7, fontFace: F.body, fontSize: 13,
      color: C.ink2, lineSpacing: 20, isTextBox: true, margin: 0 });
  note(s, "숫자 네 개만 기억하시면 됩니다. 5분에 20위, 보정 없이는 1,232위, 수율 손실 1.24%, 그리고 3분은 안 된다. 아래 상자가 중요합니다 — 기존 15분 로직과 경쟁하는 자료가 아닙니다.");
}

// ═══════════════ 3. 배경 ═══════════════
{
  const s = P.addSlide();
  head(s, "배경", "SDM은 전압을 붙들어 놓고, 그것을 유지하는 데 드는 전류를 잰다");
  img(s, "sdmconcept", { x: 1.45, y: 1.76, w: 10.4 });
  card(s, M, 5.10, 6.0, 1.86, "현행 판정",
    ["3일 보관 후 ΔOCV를 재서 ML로 판정한다.",
     "정확하지만 리드타임 3일이 그대로 재고에 얹힌다."], false);
  card(s, M + 6.28, 5.10, 6.0, 1.86, "15분 SDM 판정",
    ["SDM 15분 측정에 공정 데이터까지 결합해 판정하는 로직이 이미 확립돼 있다.",
     "이 발표는 그 위에서 시작한다."], true);
  note(s, "자가방전이 크면 전압이 더 빨리 내려가려 하고, 장비는 그만큼 더 많은 전류를 흘려 넣습니다. 그 전류가 곧 자가방전 전류입니다. 3일이 15분이 됐으니, 이 발표는 그 다음 질문입니다.");
}

// ═══════════════ 4. 1부 간지 ═══════════════
{
  const s = P.addSlide();
  titleSlide(s, "1부", "지금 SDM 데이터에\n무슨 문제가 있나", "음수 전류의 정체를 찾는다");
  note(s, "1부는 세 단계입니다. 증상을 보이고, 원인을 밝히고, 그 원인이 실제로 어떤 사고를 일으키는지 보여드립니다.");
}

// ═══════════════ 5. 1-1 증상 ═══════════════
{
  const s = P.addSlide();
  head(s, "1-1  증상", "자가방전 전류가 음수로 나오는 셀이 있다");
  s.addShape(P.ShapeType.roundRect, { x: M, y: 1.86, w: 5.9, h: 2.5, rectRadius: 0.06,
    fill: { color: "FBECEC" }, line: { color: C.crit, width: 1.5 } });
  s.addText("전류는 반드시\n양수여야 한다", { x: M + 0.34, y: 2.12, w: 5.2, h: 0.9,
    fontFace: F.head, fontSize: 22, bold: true, color: C.crit, lineSpacing: 30, isTextBox: true, margin: 0 });
  s.addText("자가방전은 셀이 스스로 전기를 잃는 현상이다. 방향이 정해져 있으므로 전류의 부호도 정해져 있다. 그런데 음수가 나오는 셀이 있다.",
    { x: M + 0.34, y: 3.12, w: 5.2, h: 1.0, fontFace: F.body, fontSize: 13.5,
      color: C.ink2, lineSpacing: 21, isTextBox: true, margin: 0 });

  s.addShape(P.ShapeType.roundRect, { x: M + 6.28, y: 1.86, w: 5.9, h: 2.5, rectRadius: 0.06,
    fill: { color: C.bg }, line: { color: C.rule, width: 1 } });
  s.addText("트레이를 바꾸면\n값이 달라진다", { x: M + 6.62, y: 2.12, w: 5.2, h: 0.9,
    fontFace: F.head, fontSize: 22, bold: true, color: C.ink, lineSpacing: 30, isTextBox: true, margin: 0 });
  s.addText("15분 전류와 3일 ΔOCV의 상관도 트레이마다 방향이 뒤집힌다. 셀은 그대로인데 결과가 달라진다.",
    { x: M + 6.62, y: 3.12, w: 5.2, h: 1.0, fontFace: F.body, fontSize: 13.5,
      color: C.ink2, lineSpacing: 21, isTextBox: true, margin: 0 });

  s.addShape(P.ShapeType.roundRect, { x: M, y: 4.72, w: W - 2 * M, h: 1.5, rectRadius: 0.06,
    fill: { color: C.navy } });
  s.addText("측정값 안에 자가방전이 아닌 무언가가 섞여 있다는 신호다.",
    { x: M + 0.4, y: 5.02, w: W - 2 * M - 0.8, h: 0.5, fontFace: F.head, fontSize: 21,
      bold: true, color: C.white, isTextBox: true, margin: 0 });
  s.addText("장비가 고장 난 것이 아니다. 더해진 성분이 자가방전보다 클 때 부호가 뒤집힌다.",
    { x: M + 0.4, y: 5.56, w: W - 2 * M - 0.8, h: 0.4, fontFace: F.body, fontSize: 13.5,
      color: "AFC4DB", isTextBox: true, margin: 0 });
  note(s, "여기서 청중이 ‘장비 문제 아니냐’고 생각하기 쉬운데, 그게 아니라는 걸 먼저 못 박습니다. 전류에 다른 성분이 더해져 있고 그게 자가방전보다 클 때 부호가 뒤집히는 겁니다.");
}

// ═══════════════ 6. 1-2 원인 ═══════════════
{
  const s = P.addSlide();
  head(s, "1-2  원인", "섞여 있던 것은 ‘식는 속도’다 — 열드리프트");
  s.addText([
    { text: "I_meas", options: { bold: true, color: C.ink } },
    { text: "  =  ", options: { color: C.muted } },
    { text: "I_sd", options: { bold: true, color: C.blue } },
    { text: "  +  ", options: { color: C.muted } },
    { text: "C · (dU/dT) · (dT/dt)", options: { bold: true, color: C.orange } },
  ], { x: M, y: 1.66, w: W - 2 * M, h: 0.44, fontFace: F.mono, fontSize: 19, isTextBox: true, margin: 0 });
  s.addText([
    { text: "자가방전 — 재고 싶은 것", options: { color: C.blue, bold: true } },
    { text: "          식는 속도에 비례 — 부호가 자유롭다", options: { color: C.orange, bold: true } },
  ], { x: M, y: 2.12, w: W - 2 * M, h: 0.32, fontFace: F.body, fontSize: 12, isTextBox: true, margin: 0 });
  img(s, "mech", { x: 2.52, y: 2.46, w: 8.3 });
  note(s, "온도가 얼마인지가 아니라 얼마나 빨리 식고 있는지가 전류로 읽힙니다. 그래서 투입 직후일수록 심하고, 하필 5분 창이 교란이 가장 큰 자리에 걸립니다. 3단 그림에서 두 선 사이의 칠한 면적이 열드리프트입니다.");
}

// ═══════════════ 7. 1-3a 증거 — 행 프로파일 ═══════════════
{
  const s = P.addSlide();
  head(s, "1-3  증거 ①", "트레이 안의 ‘자리’가 전류를 결정한다");
  img(s, "rowprofile", { x: 1.6, y: 1.72, w: 10.1 });
  s.addText("셀은 12×12 격자로 트레이에 실린다. 트레이 중앙값을 뺀 뒤 행별로 모으면 가장자리 행 A·L만 −1.3 µA 수준으로 내려간다 — 양 끝이 파인 U자다.",
    { x: M, y: 5.94, w: W - 2 * M, h: 0.5, fontFace: F.body, fontSize: 14,
      color: C.ink2, lineSpacing: 22, isTextBox: true, margin: 0 });
  s.addText("가장자리는 공기와 닿는 면이 많아 더 빨리 식는다.",
    { x: M, y: 6.50, w: W - 2 * M, h: 0.4, fontFace: F.body, fontSize: 14,
      bold: true, color: C.ink, isTextBox: true, margin: 0 });
  note(s, "단위는 마이크로암페어입니다. 트레이 중앙값을 뺀 값이라 0이 그 트레이의 보통 셀입니다. A행과 L행만 뚝 떨어져 있습니다.");
}

// ═══════════════ 8. 1-3b 증거 — 산점도 ═══════════════
{
  const s = P.addSlide();
  head(s, "1-3  증거 ②", "많이 식은 행일수록 전류를 낮게 읽는다");
  img(s, "scatter", { x: 2.5, y: 1.62, w: 8.3 });
  s.addText("점 하나가 행 하나다. 가로는 그 행이 얼마나 식었는지, 세로는 그 행의 전류다. 순위상관 ρ = −0.846.",
    { x: M, y: 6.0, w: 8.4, h: 0.5, fontFace: F.body, fontSize: 13.5,
      color: C.ink2, lineSpacing: 21, isTextBox: true, margin: 0 });
  s.addShape(P.ShapeType.roundRect, { x: 9.3, y: 5.88, w: 3.4, h: 1.0, rectRadius: 0.05,
    fill: { color: "FDF3EF" }, line: { color: C.orange, width: 1.2 } });
  s.addText("행 사이 냉각량 차이는 0.02 °C 남짓으로 작다. 순서는 뚜렷하지만 크기는 작다.",
    { x: 9.5, y: 6.0, w: 3.0, h: 0.76, fontFace: F.body, fontSize: 11,
      color: C.ink2, lineSpacing: 16, isTextBox: true, margin: 0 });
  note(s, "직선을 긋지 않은 것은 이것이 순위상관이기 때문입니다. 값의 비례가 아니라 순서가 맞는다는 뜻입니다. 오른쪽 상자를 먼저 꺼내십시오 — ‘0.02도 차이로 그게 되냐’는 질문이 반드시 나옵니다. 총 온도차가 아니라 식는 속도가 원인일 가능성이 큽니다.");
}

// ═══════════════ 9. 1-3c 집계 단위 ═══════════════
{
  const s = P.addSlide();
  head(s, "1-3  증거 ③", "행 방향으로만 구조가 있다 — 그래서 ‘행’이 비교 단위다");
  img(s, "units", { x: 1.6, y: 1.9, w: 10.1 });
  card(s, M, 4.88, 6.0, 1.84, "행으로만 본 것이 아니다",
    ["셀 하나하나(1,294개)로 봐도 −0.251로 같은 방향이다.",
     "집계 단위를 올릴수록 잡음이 평균돼 상관이 커진다."], false);
  card(s, M + 6.28, 4.88, 6.0, 1.84, "결정적 대비",
    ["행 −0.846  vs  열 −0.021",
     "열 번호로는 냉각이 설명되지 않고, 행 번호로는 설명된다."], true);
  note(s, "이게 이번 분석에서 가장 깔끔한 숫자입니다. 열 방향은 상관이 거의 0인데 행 방향은 −0.846입니다. 트레이 안에서 냉각이 행 단위로 구조화돼 있다는 뜻이고, 2부에서 행을 비교 단위로 쓰는 근거가 됩니다.");
}

// ═══════════════ 10. 1-4 셀 140 ═══════════════
{
  const s = P.addSlide();
  head(s, "1-4  사고", "그래서 이런 일이 생긴다 — 불량인데 양품보다 전류가 낮다");
  img(s, "cell140", { x: 1.6, y: 1.72, w: 10.1 });
  s.addShape(P.ShapeType.roundRect, { x: M, y: 5.42, w: W - 2 * M, h: 1.42, rectRadius: 0.06,
    fill: { color: C.navy } });
  s.addText("140번은 트레이 CFDD011456의 L08 — 맨 뒤 가장자리 행이다.",
    { x: M + 0.4, y: 5.64, w: W - 2 * M - 0.8, h: 0.4, fontFace: F.head, fontSize: 18,
      bold: true, color: C.white, isTextBox: true, margin: 0 });
  s.addText("가장 빨리 식는 자리라 열드리프트가 자가방전 신호를 통째로 눌러버렸다. SDM 전류를 그대로 쓰면 안 된다 — 어떤 형태로든 보정이 먼저다.",
    { x: M + 0.4, y: 6.10, w: W - 2 * M - 0.8, h: 0.5, fontFace: F.body, fontSize: 13.5,
      color: "AFC4DB", lineSpacing: 21, isTextBox: true, margin: 0 });
  note(s, "1부의 결론 슬라이드입니다. 셋은 양수로 튀어 쉽게 잡히는데 140번만 음수입니다. 원시 전류로 줄을 세우면 이 셀이 멀쩡한 양품 1,231개보다 뒤에 묻힙니다.");
}

// ═══════════════ 11. 2부 간지 ═══════════════
{
  const s = P.addSlide();
  titleSlide(s, "2부", "5분이 가능한지\n어떻게 판단했나",
    "기준을 정하고, 왜 3분은 안 되고 5분은 되는지 본다");
  note(s, "판단 기준을 먼저 정하고, 물리로 왜 5분인지 설명한 뒤, 방법과 결과를 보여드립니다.");
}

// ═══════════════ 12. 2-1 판단 기준 ═══════════════
{
  const s = P.addSlide();
  head(s, "2-1  판단 기준", "줄 세웠을 때 마지막 불량이 몇 번째에 있는가");
  img(s, "ranks", { x: 2.57, y: 1.58, w: 8.2 });
  s.addText("전 셀이 똑같은 5분 측정을 받고, 똑같은 점수로 줄을 선다. 위에서 몇 번째까지 보면 불량 4개가 전부 들어오는지 — 그 숫자 하나가 기준이다.",
    { x: M, y: 6.34, w: 8.5, h: 0.6, fontFace: F.body, fontSize: 13.5,
      color: C.ink2, lineSpacing: 21, isTextBox: true, margin: 0 });
  s.addText([{ text: "보정 후 1 · 2 · 18 · 20위", options: { bold: true, color: C.blue, fontSize: 15 } },
             { text: "   /   보정 전 1 · 7 · 28 · 1,232위", options: { color: C.orange, fontSize: 13 } }],
    { x: 9.2, y: 6.46, w: 3.6, h: 0.4, fontFace: F.mono, isTextBox: true, margin: 0, align: "right" });
  note(s, "여기서 강조할 것은 두 가지입니다. 첫째, 걸러낸 20셀은 추가로 검사할 셀이 아니라 불량으로 처리할 셀입니다. 2단계 공정도 적응형도 아닙니다. 둘째, 붉은 선이 셀 140인데 보정의 효과가 이 한 셀에 집중돼 있습니다. 나머지 셋은 보정 전에도 28위 안에 있었습니다.");
}

// ═══════════════ 13. 2-2a 신호 ═══════════════
{
  const s = P.addSlide();
  head(s, "2-2  왜 3분은 안 되는가 ①", "자가방전 신호는 아직 1%도 자라지 않았다");
  img(s, "growth", { x: 1.6, y: 1.70, w: 10.1 });
  card(s, M, 5.70, 6.0, 1.32, "시상수 τ = 7.45시간",
    ["30분까지도 최종값의 6.5%밖에 안 자란다. 우리는 늘 곡선의 맨 앞자락만 본다."], false);
  card(s, M + 6.28, 5.70, 6.0, 1.32, "그래도 3분과 5분은 다르다",
    ["0.67% → 1.11%. 신호가 1.66배 커진다."], true);
  note(s, "자가방전 전류는 시간이 지나야 자랍니다. 얼마나 빨리 자라는지는 셀의 시상수가 정하는데 실측값이 7.45시간입니다. 5분은 그 곡선의 맨 앞 1.1% 구간입니다.");
}

// ═══════════════ 14. 2-2b SNR ═══════════════
{
  const s = P.addSlide();
  head(s, "2-2  왜 3분은 안 되는가 ②", "신호 1.66배 × 기울기 정밀도 1.87배 = 3.1배");
  s.addText([
    { text: "신호 대 잡음비(SNR)", options: { bold: true, color: C.ink } },
    { text: " = 신호 ÷ 잡음. 이 값이 크면 불량 셀이 양품 무리 밖으로 튀어나오고, 작으면 잡음에 묻힌다. 기울기는 점이 많을수록(잡음이 평균된다)·창이 길수록(지렛대가 길어진다) 정확해진다 — 3분은 4점·길이 3, 5분은 6점·길이 5다.", options: {} },
  ], { x: M, y: 1.62, w: W - 2 * M, h: 0.62, fontFace: F.body, fontSize: 13.5,
       color: C.ink2, lineSpacing: 21, isTextBox: true, margin: 0 });
  img(s, "snr4", { x: 3.07, y: 2.26, w: 7.2 });
  card(s, M, 6.06, 6.0, 1.02, "3분이 안 되는 이유",
    ["신호가 덜 자랐고 점이 4개뿐이다. 5분보다 3.1배 불리하다."], false);
  card(s, M + 6.28, 6.06, 6.0, 1.02, "5분 이후에 안 줄어드는 이유",
    ["SNR은 계속 오르는데 순위는 멈춘다. 이미 문턱을 넘었다는 뜻이다."], true);
  note(s, "SNR은 이론 계산값입니다. 기울기 정밀도 1.87배는 최소자승 기울기의 표준오차 계산에서 나옵니다 — 점 개수와 창 길이만으로 정해지는 값입니다. 3분을 1로 두면 5분이 3.1배, 10분 16배, 15분 41배로 계속 오릅니다. 그런데 아래 칸의 실측 순위는 5분에서 20위로 떨어진 뒤 13~20위에서 멈춥니다. 5분에서 이미 문턱을 넘었고 그 뒤는 다른 것이 바닥을 정한다는 뜻입니다. 그래서 5분이 비용 대비 최적점입니다.");
}

// ═══════════════ 15. 2-3 방법 ═══════════════
{
  const s = P.addSlide();
  head(s, "2-3  방법", "측정을 바꾸지 않는다 — 같은 5분 곡선을 다르게 읽는다");
  img(s, "heat", { x: 2.2, y: 1.58, w: 8.9 });
  card(s, M, 5.78, 6.0, 1.30, "왜 같은 행 12셀인가",
    ["트레이 전체는 교란이 덜 상쇄되고, 인접 8셀은 중앙값이 불안정하다. 1-3의 행 −0.846 / 열 −0.021이 근거다."], false);
  card(s, M + 6.28, 5.78, 6.0, 1.30, "왜 끝값이 아니라 기울기인가",
    ["끝값에는 열드리프트가 만든 오프셋이 그대로 들어온다. 기울기는 변화율만 타므로 오프셋이 스스로 빠진다."], true);
  note(s, "칸 안의 숫자는 마지막 불량의 순위를 전체 대비 퍼센트로 적은 것입니다. 1.55%면 1,294셀 중 20번째까지 봐야 한다는 뜻입니다. 전부 5분 창의 전류 데이터로 계산했고 온도는 쓰지 않았습니다. 24개 조합 중 한 칸만 내려갑니다.");
}

// ═══════════════ 16. 2-4 결과 ═══════════════
{
  const s = P.addSlide();
  head(s, "2-4  결과", "3분과 5분 사이에 문턱이 있다");
  img(s, "cliff", { x: 0.5, y: 1.56, w: 7.5 });
  const rows = [
    [{ text: "측정 시간", options: { bold: true } }, { text: "마지막 불량", options: { bold: true } },
     { text: "양품", options: { bold: true } }, { text: "수율 손실", options: { bold: true } }],
    ["3분", "387위", "383셀", "29.60%"],
    ["5분", "20위", "16셀", "1.24%"],
    ["8분", "14위", "10셀", "0.77%"],
    ["10분", "13위", "9셀", "0.70%"],
    ["15분", "16위", "12셀", "0.93%"],
    ["30분", "11위", "7셀", "0.54%"],
  ];
  s.addTable(rows, {
    x: 8.3, y: 1.86, w: 4.4, colW: [1.0, 1.3, 0.95, 1.15],
    fontFace: F.body, fontSize: 11.5, color: C.ink2, border: { type: "solid", color: C.rule, pt: 0.75 },
    fill: { color: C.white }, align: "right", valign: "middle", rowH: 0.36,
  });
  s.addShape(P.ShapeType.rect, { x: 8.3, y: 2.58, w: 4.4, h: 0.36,
    fill: { color: "EAF2FC" }, line: { color: C.blue, width: 1.2 } });
  s.addText([{ text: "5분", options: { bold: true } }, { text: "          20위          16셀        1.24%", options: {} }],
    { x: 8.36, y: 2.6, w: 4.28, h: 0.32, fontFace: F.body, fontSize: 11.5,
      bold: true, color: C.blue, isTextBox: true, margin: 0 });
  s.addText("보정 없이 트레이 전류 끝값만 쓰면 어느 시점에서도 1,230위 아래다 — 사실상 전수 검사다.\n※ 이 기준선은 현행 공정의 판정 방식이 아니다. 보정을 안 했을 때의 비교용으로만 쓴 것이다.",
    { x: 8.3, y: 5.3, w: 4.4, h: 1.0, fontFace: F.body, fontSize: 10.5,
      color: C.muted, lineSpacing: 16, isTextBox: true, margin: 0 });
  note(s, "5분을 10분으로 늘리면 20위가 13위가 됩니다. 측정 시간을 두 배 쓰고 수율 0.5%p를 얻는 거래입니다. 어느 쪽이 맞는지는 라인이 판단할 문제입니다. 그리고 양품이 하나도 안 걸리려면 불량 4개가 정확히 1·2·3·4위여야 하는데, 선별에서 그런 일은 없습니다. 판단 기준은 0인가가 아니라 현행 대비 손해인가입니다.");
}

// ═══════════════ 17. 3부 간지 ═══════════════
{
  const s = P.addSlide();
  titleSlide(s, "3부", "앞으로 할 것", "공정 데이터 결합, 다른 로트 재현");
  note(s, "지금 결과는 SDM 전류 곡선과 셀 번호만으로 얻은 것입니다. 여기에 둘을 더합니다.");
}

// ═══════════════ 18. 3-1 계획 ═══════════════
{
  const s = P.addSlide();
  head(s, "3-1  계획", "지금은 SDM 데이터만 썼다 — 여기에 둘을 더한다");
  img(s, "plan", { x: 1.6, y: 1.66, w: 10.1 });
  card(s, M, 3.94, 6.0, 3.06, "01   공정 데이터 결합",
    ["붙일 것 — 조립·화성 이력 등 셀 단위로 따라붙는 공정 변수",
     "검증 — 트레이 단위로 학습과 평가를 분리한다. 같은 트레이가 양쪽에 들어가면 성능이 부풀려진다",
     "볼 것 — 마지막 불량이 20위보다 올라가는가. 안 올라가면 SDM 단독이 이미 한계라는 뜻이다"], true);
  card(s, M + 6.28, 3.94, 6.0, 3.06, "02   다른 로트 재현",
    ["지금 결과는 한 로트에서 나왔고 불량이 4개뿐이다",
     "볼 것 — 5분 마지막 불량이 다시 20위권인가",
     "볼 것 — 행 U자와 냉각량 상관이 다시 나오는가",
     "볼 것 — L행 편중이 재현되는가. 재현되면 자리가 실제로 불량을 만든다는 뜻이고, 그러면 자리 보정이 오히려 불량을 가린다"], false);
  note(s, "지금 말씀드릴 수 있는 것은 5분이 가능해 보인다까지입니다. 공정 데이터를 붙이고 로트를 늘려서 가능한지 아닌지 판단하는 것이 다음 단계입니다. 찾겠다가 아니라 판단하겠다입니다.");
}

// ═══════════════ 19. 마무리 ═══════════════
{
  const s = P.addSlide();
  s.background = { color: C.navy };
  s.addText("한 문장으로", { x: M, y: 1.9, w: W - 2 * M, h: 0.4,
    fontFace: F.mono, fontSize: 13, color: C.blue, bold: true, charSpacing: 2, isTextBox: true });
  s.addText([
    { text: "SDM 전류에는 ", options: {} },
    { text: "식는 속도", options: { bold: true, color: C.white } },
    { text: "가 섞여 있습니다.\n그것을 ", options: {} },
    { text: "자리로 빼고 기울기로 읽으면", options: { bold: true, color: C.white } },
    { text: "\n5분에 1,294셀 중 ", options: {} },
    { text: "상위 20위", options: { bold: true, color: C.white } },
    { text: " 안에 불량 4개가 전부 들어옵니다.", options: {} },
  ], { x: M, y: 2.5, w: W - 2 * M - 0.6, h: 2.3, fontFace: F.head, fontSize: 26,
       color: "CBDCF0", lineSpacing: 46, isTextBox: true });
  s.addText("3분은 안 됩니다 — 신호와 정밀도가 합쳐 3.1배 모자랍니다.",
    { x: M, y: 4.9, w: W - 2 * M, h: 0.44, fontFace: F.body, fontSize: 16,
      color: C.orange, bold: true, isTextBox: true });
  s.addText("불량이 4개뿐이라 수율 수치는 방향 지표로만 읽어야 하고, 다른 로트 재현이 남아 있습니다.",
    { x: M, y: 5.52, w: W - 2 * M, h: 0.44, fontFace: F.body, fontSize: 13,
      color: "7E97B5", isTextBox: true });
  note(s, "마지막에 한계를 먼저 말하는 것이 낫습니다. 질의응답에서 표본 수를 지적당하기 전에 꺼내면 앞의 숫자가 오히려 더 믿어집니다.");
}

// ═══════════════ 20. 백업 — 판정 규칙 ═══════════════
{
  const s = P.addSlide();
  head(s, "백업", "트레이별 상대평가로 자르면 양품이 덜 걸리는가 — 아니다");
  img(s, "rulebar", { x: 1.6, y: 1.84, w: 10.1 });
  card(s, M, 5.34, 6.0, 1.68, "이유 ①  불량이 트레이에 고르지 않다",
    ["9트레이 중 4개에만 하나씩 있다. 불량 없는 5개 트레이에서 걸러지는 셀은 전부 양품이다."], false);
  card(s, M + 6.28, 5.34, 6.0, 1.68, "이유 ②  자기 마스킹",
    ["불량이 자기 트레이의 산포를 키워 문턱을 밀어올린다. 138번은 자기 자신 때문에 z가 9.38에서 7.36으로 깎였다."], false);
  note(s, "현행 공정 판정은 ML입니다. 여기서 시험한 트레이 상대평가는 현행 규칙이 아니라, 이 점수에 씌울 수 있는 판정 규칙 후보 중 하나입니다. 네 가지를 다 시험했고 전역 순위 컷이 가장 적게 버렸습니다.");
}

// ═══════════════ 21. 백업 — 기각 목록 ═══════════════
{
  const s = P.addSlide();
  head(s, "백업", "해보고 안 된 것들");
  const rows = [
    [{ text: "시도", options: { bold: true } }, { text: "마지막 불량", options: { bold: true } },
     { text: "기각 근거", options: { bold: true } }],
    ["곡선 모양 분해 (직선/곡률)", "1,145~1,275위", "온도가 곡률보다 직선 성분에 6~14배 강하게 붙는다 — 가설과 반대"],
    ["주성분 분석으로 공통성분 제거", "1,275위", "첫 성분이 곧 신호의 크기라, 빼면 신호가 지워진다"],
    ["적응형 측정 시간", "이득 없음", "평균 20.3분을 써야 고정 5분과 같다. 전 셀 동일 공정 제약에도 위배"],
    ["공통 모양 × 트레이별 크기", "941위", "원리적으로 가장 그럴듯했으나 실측에서 52배 졌다"],
  ];
  s.addTable(rows, {
    x: M, y: 1.76, w: W - 2 * M, colW: [3.5, 1.8, 6.81],
    fontFace: F.body, fontSize: 12, color: C.ink2, border: { type: "solid", color: C.rule, pt: 0.75 },
    fill: { color: C.white }, valign: "middle", rowH: 0.52,
  });
  s.addShape(P.ShapeType.roundRect, { x: M, y: 4.9, w: W - 2 * M, h: 1.3, rectRadius: 0.06,
    fill: { color: "EAF2FC" }, line: { color: C.blue, width: 1.2 } });
  s.addText("보정 방법은 원리로 고르면 안 된다.",
    { x: M + 0.36, y: 5.1, w: W - 2 * M - 0.72, h: 0.4, fontFace: F.head, fontSize: 18,
      bold: true, color: C.blue, isTextBox: true, margin: 0 });
  s.addText("가장 그럴듯했던 방식이 실측에서 52배 졌다. 마지막 불량 순위로 골라야 한다.",
    { x: M + 0.36, y: 5.56, w: W - 2 * M - 0.72, h: 0.4, fontFace: F.body, fontSize: 13.5,
      color: C.ink2, isTextBox: true, margin: 0 });
  note(s, "이것도 저것도 해봤고 왜 안 되는지 안다는 것이 결과의 신뢰를 만듭니다. 질문이 나오면 이 표를 띄우십시오.");
}

const OUT = path.join(DIR, "SDM_5분선별_가능성검토.pptx");
P.writeFile({ fileName: OUT }).then(() => console.log("작성:", OUT));
