// Push-optimal Sokoban. A level is a hallway plus rooms off it; a box pushed into the hallway is lost, so the
// level's optimum is its rooms' summed. Each room is solved exactly by A* over pushes with a matching lower
// bound, dead squares and frozen 2x2 blocks pruned; the rooms are stitched with walks and the answer replayed.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const IN = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUT = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const D = [[0, -1, 'u', 'U'], [0, 1, 'd', 'D'], [-1, 0, 'l', 'L'], [1, 0, 'r', 'R']];
const CAP = 400000;

function parse(t) {
  const R = t.split('\n').filter((r) => r.length), h = R.length, w = Math.max(...R.map((r) => r.length));
  const W = new Uint8Array(w * h).fill(1), G = [], B = [];
  let P = -1;
  R.forEach((r, y) => {
    for (let x = 0; x < r.length; x++) {
      const c = r[x], i = y * w + x;
      if (c === '#') continue;
      W[i] = 0;
      if ('.*+'.includes(c)) G.push(i);
      if ('$*'.includes(c)) B.push(i);
      if ('@+'.includes(c)) P = i;
    }
  });
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (!x || !y || x === w - 1 || y === h - 1) W[y * w + x] = 1;
  return { w, W, G: G.sort((a, b) => a - b), B: B.sort((a, b) => a - b), P };
}

// The squares a box can still be pushed onto a goal from, pulled back from the goals over open floor.
function live(w, W, G) {
  const L = new Uint8Array(W.length), q = [];
  for (const g of G) { L[g] = 1; q.push(g); }
  for (let i = 0; i < q.length; i++) for (const [dx, dy] of D) {
    const d = dy * w + dx, a = q[i] - d, b = a - d;
    if (b < 0 || b >= W.length || W[a] || W[b] || L[a]) continue;
    L[a] = 1;
    q.push(a);
  }
  return L;
}

// For each goal, the fewest pushes that bring a box there from each square, the other boxes ignored (-1: never).
function pushDist(w, W, G) {
  return G.map((g) => {
    const t = new Int32Array(W.length).fill(-1), q = [g];
    t[g] = 0;
    for (let i = 0; i < q.length; i++) for (const [dx, dy] of D) {
      const d = dy * w + dx, a = q[i] - d, b = a - d;
      if (b < 0 || b >= W.length || W[a] || W[b] || t[a] >= 0) continue;
      t[a] = t[q[i]] + 1;
      q.push(a);
    }
    return t;
  });
}

// The fewest pushes putting each of n boxes on a distinct one of n goals (Infinity when impossible).
function match(C, n) {
  let best = Infinity;
  const u = new Uint8Array(n);
  const go = (i, s) => {
    if (s >= best) return;
    if (i === n) { best = s; return; }
    for (let j = 0; j < n; j++) {
      if (u[j] || C[i][j] === Infinity) continue;
      u[j] = 1;
      go(i + 1, s + C[i][j]);
      u[j] = 0;
    }
  };
  go(0, 0);
  return best;
}

// The shortest walk from one square to another around walls and the boxes in occ, in lower LURD letters; null if none.
function walk(w, W, occ, from, to) {
  if (from === to) return '';
  const p = new Int32Array(W.length).fill(-1), e = new Array(W.length), q = [from];
  p[from] = from;
  for (let i = 0; i < q.length; i++) {
    const x = q[i];
    if (x === to) break;
    for (const [dx, dy, c] of D) {
      const y = x + dy * w + dx;
      if (W[y] || occ[y] || p[y] >= 0) continue;
      p[y] = x;
      e[y] = c;
      q.push(y);
    }
  }
  if (p[to] < 0) return null;
  let s = '';
  for (let x = to; x !== from; x = p[x]) s = e[x] + s;
  return s;
}

// Push-optimal answer for one room: its boxes onto its goals, the player starting on square P0; null if unsolvable.
function solve(w, W, G, B0, P0) {
  const S = W.length, GM = new Uint8Array(S), dist = pushDist(w, W, G), n = B0.length;
  for (const g of G) GM[g] = 1;
  const L = live(w, W, G);
  if (n !== G.length || B0.some((b) => !L[b])) return null;
  if (B0.every((b) => GM[b])) return '';

  const O = new Uint8Array(S), hs = new Int32Array(S), cs = new Int32Array(S), st = new Int32Array(S);
  let hm = 0, cm = 0;
  const h = (bs) => match(bs.map((b) => dist.map((t) => (t[b] < 0 ? Infinity : t[b]))), n);
  // Marks the squares the player reaches with O in the way (in cs) and answers the lowest of them.
  const low = (f) => {
    cs[f] = ++cm;
    let lo = f, top = 0;
    st[top++] = f;
    while (top > 0) {
      const p = st[--top];
      if (p < lo) lo = p;
      for (const [dx, dy] of D) {
        const q = p + dy * w + dx;
        if (!W[q] && !O[q] && cs[q] !== cm) { cs[q] = cm; st[top++] = q; }
      }
    }
    return lo;
  };

  for (const b of B0) O[b] = 1;
  const k0 = `${low(P0)}:${B0.join(',')}`;
  for (const b of B0) O[b] = 0;
  const XB = [B0.slice()], XP = [P0], XG = [0], XT = [-1], XF = [h(B0)], XQ = [0], XK = [k0], XR = [-1], XD = [-1];
  if (XF[0] === Infinity) return null;
  const seen = new Map([[k0, 0]]);
  // A binary heap of state indices, ordered by f and then by the order the states were made.
  const heap = [];
  const less = (a, b) => XF[a] < XF[b] || (XF[a] === XF[b] && XQ[a] < XQ[b]);
  const put = (i) => {
    heap.push(i);
    for (let k = heap.length - 1, p; k > 0 && less(heap[k], heap[p = (k - 1) >> 1]); k = p) {
      [heap[k], heap[p]] = [heap[p], heap[k]];
    }
  };
  const get = () => {
    const top = heap[0], last = heap.pop();
    if (heap.length) {
      heap[0] = last;
      for (let k = 0, c; (c = 2 * k + 1) < heap.length; k = c) {
        if (c + 1 < heap.length && less(heap[c + 1], heap[c])) c++;
        if (!less(heap[c], heap[k])) break;
        [heap[k], heap[c]] = [heap[c], heap[k]];
      }
    }
    return top;
  };

  put(0);
  let found = -1, exp = 0;
  while (heap.length) {
    const s = get();
    if (seen.get(XK[s]) !== XG[s]) continue;
    const bs = XB[s];
    if (bs.every((b) => GM[b])) { found = s; break; }
    if (exp++ >= CAP) break;
    for (const b of bs) O[b] = 1;
    // The player's region, marked in hs, so each push tests its own square at once.
    hs[XP[s]] = ++hm;
    let top = 0;
    st[top++] = XP[s];
    while (top > 0) {
      const p = st[--top];
      for (const [dx, dy] of D) {
        const q = p + dy * w + dx;
        if (!W[q] && !O[q] && hs[q] !== hm) { hs[q] = hm; st[top++] = q; }
      }
    }
    for (let bi = 0; bi < bs.length; bi++) {
      const b = bs[bi];
      for (let k = 0; k < 4; k++) {
        const [dx, dy, , up] = D[k], d = dy * w + dx, t = b + d;
        if (hs[b - d] !== hm || W[t] || O[t] || !L[t]) continue;
        O[b] = 0;
        O[t] = 1;
        let bad = false;
        // A 2x2 block of walls and boxes holding an off-goal box is frozen: no push can ever free it.
        for (const a of [t, t - 1, t - w, t - w - 1]) {
          const r = [a, a + 1, a + w, a + w + 1];
          if (r.every((q) => W[q] || O[q]) && r.some((q) => O[q] && !GM[q])) { bad = true; break; }
        }
        let nk = null, nb = null;
        if (!bad) {
          nb = bs.slice();
          nb[bi] = t;
          nb.sort((x, y) => x - y);
          nk = `${low(b)}:${nb.join(',')}`;
        }
        O[t] = 0;
        O[b] = 1;
        if (bad || (seen.get(nk) ?? Infinity) <= XG[s] + 1) continue;
        const hh = h(nb);
        if (hh === Infinity) continue;
        const i = XB.length;
        XB.push(nb); XP.push(b); XG.push(XG[s] + 1); XT.push(s); XQ.push(i); XK.push(nk); XR.push(b); XD.push(k);
        XF.push(XG[i] + hh);
        seen.set(nk, XG[i]);
        put(i);
      }
    }
    for (const b of bs) O[b] = 0;
  }
  if (found < 0) return null;

  const path = [];
  for (let s = found; s > 0; s = XT[s]) path.push(s);
  path.reverse();
  let p = P0, mv = '';
  for (const s of path) {
    const d = D[XD[s]], pb = XB[XT[s]], stand = XR[s] - (d[1] * w + d[0]);
    for (const b of pb) O[b] = 1;
    const wk = walk(w, W, O, p, stand);
    for (const b of pb) O[b] = 0;
    if (wk === null) return null;
    mv += wk + d[3];
    p = XR[s];
  }
  return mv;
}

// The level's rooms: the player's row is the hallway, its dead squares are cut, and each part of the floor
// left holding a box is a room. Null when the level is not laid out as rooms.
function rooms(L) {
  const { w, W, G, B, P } = L, lv = live(w, W, G), row = Math.floor(P / w);
  const cut = new Uint8Array(W.length), part = new Int32Array(W.length).fill(-1), out = [];
  for (let i = row * w; i < (row + 1) * w; i++) if (!W[i] && !lv[i]) cut[i] = 1;
  if (B.some((b) => cut[b])) return null;
  for (let s = 0; s < W.length; s++) {
    if (W[s] || cut[s] || part[s] >= 0) continue;
    const sq = [s];
    part[s] = out.length;
    for (let i = 0; i < sq.length; i++) for (const [dx, dy] of D) {
      const q = sq[i] + dy * w + dx;
      if (!W[q] && !cut[q] && part[q] < 0) { part[q] = out.length; sq.push(q); }
    }
    out.push({ sq, b: [], g: [], s: -1 });
  }
  for (const b of B) { const r = out[part[b]]; if (!r) return null; r.b.push(b); }
  for (const g of G) { const r = out[part[g]]; if (!r) return null; r.g.push(g); }
  const R = out.filter((r) => r.b.length);
  if (R.some((r) => r.b.length !== r.g.length)) return null;
  for (const r of R) {
    r.b.sort((a, b) => a - b);
    r.g.sort((a, b) => a - b);
    // The hallway square outside the room's door: where the player enters it.
    for (const sq of r.sq) for (const [dx, dy] of D) {
      const q = sq + dy * w + dx;
      if (W[q] || cut[q] !== 1) continue;
      if (r.s < 0 || q < r.s) r.s = q;
    }
  }
  return R;
}

// Plays moves on the level: null on an illegal step, else the boxes and the player's square it ends on.
function replay(L, bs, p, mv) {
  const { w, W } = L, O = new Uint8Array(W.length);
  for (const b of bs) O[b] = 1;
  for (const c of mv) {
    const d = D.find((v) => v[2] === c.toLowerCase()), nx = p + d[1] * w + d[0], be = nx + d[1] * w + d[0];
    if (W[nx] || (O[nx] && (W[be] || O[be]))) return null;
    if (O[nx]) { O[nx] = 0; O[be] = 1; }
    p = nx;
  }
  const b = [];
  for (let i = 0; i < O.length; i++) if (O[i]) b.push(i);
  return { p, b };
}

// The level's answer: each room solved in turn, the player walking between them, the whole thing replayed.
function answer(t) {
  const L = parse(t), R = rooms(L), S = L.W.length, GM = new Uint8Array(S);
  if (R === null) return '';
  for (const g of L.G) GM[g] = 1;
  let mv = '', p = L.P, bs = L.B.slice();
  const O = new Uint8Array(S);
  for (const r of R) {
    if (r.s < 0) return '';
    const W = L.W.slice();
    for (const o of R) if (o !== r) for (const q of o.sq) W[q] = 1;
    const sol = solve(L.w, W, r.g, r.b, r.s);
    if (sol === null) return '';
    for (const b of bs) O[b] = 1;
    const wk = walk(L.w, L.W, O, p, r.s);
    for (const b of bs) O[b] = 0;
    if (wk === null) return '';
    const done = replay(L, bs, p, wk + sol);
    if (!done) return '';
    mv += wk + sol;
    p = done.p;
    bs = done.b;
  }
  const fin = replay(L, L.B, L.P, mv);
  if (!fin || !fin.b.every((b) => GM[b])) return '';
  return mv;
}

for (const id of readdirSync(IN).sort()) {
  const t = readFileSync(join(IN, id, 'level.txt'), 'utf8');
  mkdirSync(join(OUT, id), { recursive: true });
  writeFileSync(join(OUT, id, 'solution.txt'), answer(t) + '\n');
}
