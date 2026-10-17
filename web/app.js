/* Vancouver 2026 Candidate Explorer - frontend (vanilla JS + Plotly). */
"use strict";
const S = {
  meta: null, cands: [], cmap: {}, analysis: null, sources: {}, race: "mayor", view: "policy", mode: "individual",
  proj: "pca", colorParty: true, hidden: new Set(), selected: null, compare: [], custom: null,
  weights: {}, blocks: { policy: 1, experience: 0.5, actions: 0.5 }, search: "", sort: ["ballot_number", 1],
};
const RACES = [["mayor", "Mayor"], ["council", "City Council"], ["park", "Park Board"], ["school", "School Board"], ["all", "Cross-race (policy only)"]];
// Fixed categorical order (colour follows the party, never its rank). Parties past slot 8 share "other" grey
// and are distinguished by marker shape + legend + label.
const PARTY_ORDER = ["abc", "onecity", "green", "team", "cope", "vanlib", "votevan", "bfv"];
const SYMBOLS = { abc: "circle", onecity: "square", green: "diamond", team: "triangle-up", cope: "x", vanlib: "cross",
  votevan: "triangle-down", bfv: "star", ipp: "hexagon", trueblue: "pentagon", affordablehousing: "hourglass", none: "circle-open" };
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const $ = (q) => document.querySelector(q);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const partyColor = (pid) => { const i = PARTY_ORDER.indexOf(pid); return i >= 0 ? css(`--s${i + 1}`) : css("--other"); };
const VERIF = {
  official_archived_snapshot: '<span class="badge ok">official page (archived)</span>',
  official_registration_plus_secondary_ballot_number: '<span class="badge warn">name official (Elections BC); ballot # secondary</span>',
  secondary_source: '<span class="badge warn">secondary source</span>' };
const fmtPct = (x) => (x == null ? "–" : (100 * x).toFixed(0) + "%");

async function j(url) { const r = await fetch(url); if (!r.ok) throw new Error(`${url}: ${r.status}`); return r.json(); }

async function init() {
  [S.meta, S.cands, S.analysis] = await Promise.all([j("/api/meta"), j("/api/candidates"), j("/api/analysis")]);
  (await j("/api/sources")).forEach((s) => (S.sources[s.source_id] = s));
  S.cands.forEach((c) => (S.cmap[c.candidate_id] = c));
  S.meta.config.topics.forEach((t) => (S.weights[t] = 1));
  banner(); tabs(); sliders(); exportsList();
  $("#viewSel").onchange = (e) => { S.view = e.target.value; S.custom = null; render(); };
  $("#modeSel").onchange = (e) => { S.mode = e.target.value; S.custom = null; render(); };
  $("#projSel").onchange = (e) => { S.proj = e.target.value; render(); };
  $("#colorParty").onchange = (e) => { S.colorParty = e.target.checked; render(); };
  $("#search").oninput = (e) => { S.search = e.target.value.toLowerCase(); table(); };
  $("#weightsBtn").onclick = () => { const p = $("#weightsPanel"); p.hidden = !p.hidden; $("#weightsBtn").setAttribute("aria-expanded", String(!p.hidden)); };
  $("#applyWeights").onclick = applyWeights;
  $("#resetWeights").onclick = () => { Object.keys(S.weights).forEach((k) => (S.weights[k] = 1)); S.blocks = { policy: 1, experience: 0.5, actions: 0.5 }; sliders(); S.custom = null; render(); };
  $("#clearCompare").onclick = () => { S.compare = []; renderCompare(); table(); };
  document.querySelectorAll("#ctable th[data-k]").forEach((th) => (th.onclick = () => {
    const k = th.dataset.k; S.sort = [k, S.sort[0] === k ? -S.sort[1] : 1]; table();
  }));
  const h = new URLSearchParams(location.hash.slice(1));
  if (h.get("race")) S.race = h.get("race");
  if (h.get("c") && S.cmap[h.get("c")]) { S.selected = h.get("c"); S.race = S.cmap[S.selected].race; }
  render();
}

function banner() {
  const m = S.meta, b = m.ballot;
  const items = [
    `<div class="b"><b>Pre-election data.</b> Ballot retrieved ${esc(b.retrieved_at.slice(0, 10))}; election day ${esc(b.election_date)}. No results are included.
      ${b.total} candidates: mayor ${b.counts.mayor}, council ${b.counts.council}, Park Board ${b.counts.park}, school ${b.counts.school}.</div>`,
    `<div class="b warn"><b>How the ballot was verified.</b> vancouver.ca refuses automated access (bot challenge). The mayor, Park Board and school lists are from Internet Archive captures of the City's official pages. The councillor page was never archived, so council <i>ballot numbers</i> come from Daily Hive's ballot-order list, which matches the official pages for all 74 other entries. All ${b.total} names and offices are corroborated by Elections BC's official registered-candidate list.</div>`,
    `<div class="b warn"><b>Coverage is not comprehensive.</b> ${m.coverage.with_policy_position}/${m.coverage.candidates_total} candidates have at least one individual policy statement in collected sources; ${m.coverage.without_any_evidence.length} have no evidence collected. “Not found” means absent from the collected sources, not a position.</div>`,
  ];
  $("#banner").innerHTML = items.join("");
}

function tabs() {
  const el = $("#raceTabs");
  el.innerHTML = RACES.map(([k, l]) => `<button role="tab" data-r="${k}" aria-selected="${k === S.race}">${esc(l)}</button>`).join("");
  el.querySelectorAll("button").forEach((b) => (b.onclick = () => { S.race = b.dataset.r; S.custom = null; S.hidden.clear(); render(); }));
}

function sliders() {
  const tl = S.meta.topic_labels;
  $("#topicSliders").innerHTML = S.meta.config.topics.map((t) =>
    `<div class="slider"><label for="w_${t}">${esc(tl[t] || t)}</label><input type="range" id="w_${t}" min="0" max="2" step="0.25" value="${S.weights[t]}"><output>${S.weights[t]}</output></div>`).join("");
  $("#blockSliders").innerHTML = ["policy", "experience", "actions"].map((b) =>
    `<div class="slider"><label for="b_${b}">${esc(S.meta.views[b])}</label><input type="range" id="b_${b}" min="0" max="2" step="0.25" value="${S.blocks[b]}"><output>${S.blocks[b]}</output></div>`).join("");
  document.querySelectorAll("#weightsPanel input[type=range]").forEach((i) => (i.oninput = () => {
    i.nextElementSibling.textContent = i.value;
    if (i.id.startsWith("w_")) S.weights[i.id.slice(2)] = +i.value; else S.blocks[i.id.slice(2)] = +i.value;
  }));
}

async function applyWeights() {
  $("#wstatus").textContent = "Recomputing…";
  const view = S.view === "votes" ? "policy" : S.view;
  const q = new URLSearchParams({ race: S.race, view, mode: S.mode, weights: JSON.stringify(S.weights), blocks: JSON.stringify(S.blocks) });
  try { S.custom = await j(`/api/map?${q}`); S.customKey = key(); $("#wstatus").textContent = "Custom weights applied (stability metrics shown are for default weights)."; }
  catch (e) { $("#wstatus").textContent = "Failed: " + e.message; }
  render();
}

function setHash() {
  const h = "#" + new URLSearchParams({ race: S.race, ...(S.selected ? { c: S.selected } : {}) }).toString();
  if (location.hash !== h) history.replaceState(null, "", h);
}

const key = () => `${S.race}|${S.view}|${S.mode}`;

function currentMap() {
  if (S.custom && S.customKey === key()) return S.custom;
  if (S.view === "combined") return null;
  const k = `${S.view}:${S.view === "policy" ? S.mode : "individual"}`;
  if (S.race === "all") return S.analysis.cross_race["policy:individual"];
  return S.analysis.races[S.race][k];
}

function raceCands() { return S.cands.filter((c) => S.race === "all" || c.race === S.race); }

async function render() {
  setHash();
  tabs();
  $("#modeCtl").style.display = S.view === "policy" || S.view === "combined" ? "" : "none";
  $("#projCtl").style.display = S.view === "votes" ? "none" : "";
  if (S.race === "all" && S.view !== "policy") { S.view = "policy"; $("#viewSel").value = "policy"; }
  $("#viewSel").querySelectorAll("option").forEach((o) => (o.disabled = S.race === "all" && o.value !== "policy"));
  if (S.view === "combined" && !(S.custom && S.customKey === key())) { await applyWeights(); return; }
  partyChips();
  if (S.view === "votes") renderVotes(); else renderMap();
  table();
  if (S.selected) panel(S.selected);
}

function partyChips() {
  const parties = {};
  raceCands().forEach((c) => (parties[c.party_id] = c.party));
  const ids = Object.keys(parties).sort((a, b) => (PARTY_ORDER.indexOf(a) + 99 * (PARTY_ORDER.indexOf(a) < 0)) - (PARTY_ORDER.indexOf(b) + 99 * (PARTY_ORDER.indexOf(b) < 0)));
  $("#partyFilter").innerHTML = `<span class="hint">Show/hide (hiding never deletes data):</span>` + ids.map((p) =>
    `<button class="chip" data-p="${esc(p)}" aria-pressed="${!S.hidden.has(p)}"><span class="sw" style="background:${partyColor(p)}"></span>${esc(parties[p])}</button>`).join("");
  document.querySelectorAll(".chip").forEach((b) => (b.onclick = () => {
    const p = b.dataset.p; S.hidden.has(p) ? S.hidden.delete(p) : S.hidden.add(p); render();
  }));
}

function plotLayout(xt, yt) {
  const ink = css("--text-2"), grid = css("--border");
  return {
    margin: innerWidth < 600 ? { l: 36, r: 24, t: 24, b: 40 } : { l: 70, r: 70, t: 28, b: 44 }, paper_bgcolor: css("--surface"), plot_bgcolor: css("--surface"),
    font: { family: "system-ui, sans-serif", size: 12, color: ink }, hovermode: "closest", dragmode: "pan",
    xaxis: { title: { text: xt }, zeroline: false, gridcolor: grid, showticklabels: false },
    yaxis: { title: { text: yt }, zeroline: false, gridcolor: grid, showticklabels: false },
    legend: { orientation: "h", y: innerWidth < 600 ? -0.22 : -0.12, font: { size: 11 } }, showlegend: S.colorParty,
  };
}

function renderMap() {
  const m = currentMap();
  const note = [];
  if (S.race === "all") note.push(esc(S.analysis.cross_race.warning));
  if (S.view === "platform") note.push("Party-platform view: candidates of one party share one platform vector and therefore one point. This shows the party, not the individual.");
  if (S.view === "actions") note.push("Mixes independent reporting with campaign-claimed outcomes (labelled per item). Claimed outcomes are unverified.");
  if (S.view === "combined") note.push(esc(m?.note || ""));
  $("#mapNote").innerHTML = note.join("<br>");
  if (!m || !m.points || m.n < 3 || !m.points[0].pca) {
    Plotly.purge("plot");
    $("#plot").innerHTML = `<p class="empty" style="padding:1rem">${esc(m?.note || "Not enough candidates with sufficient evidence for a projection in this view.")}</p>`;
    $("#mapMeta").innerHTML = ""; excludedList(m); return;
  }
  const useU = S.proj === "umap" && m.umap && m.umap.available;
  const stab = (S.analysis.races[S.race]?.[`${S.view}:${S.view === "policy" ? S.mode : "individual"}`] || {}).stability || {};
  const pts = m.points.filter((p) => !S.hidden.has(S.cmap[p.id].party_id));
  const groups = {};
  pts.forEach((p) => { const pid = S.colorParty ? S.cmap[p.id].party_id : "all"; (groups[pid] ||= []).push(p); });
  const traces = Object.entries(groups).map(([pid, ps]) => {
    const unstable = ps.map((p) => stab[p.id]?.unstable);
    return {
      type: "scatter", mode: "markers+text", name: pid === "all" ? "Candidates" : S.cmap[ps[0].id].party,
      x: ps.map((p) => (useU ? p.umap : p.pca)[0]), y: ps.map((p) => (useU ? p.umap : p.pca)[1]),
      text: ps.map((p) => (p.id === S.selected || S.compare.includes(p.id) ? S.cmap[p.id].display_name : "")),
      textposition: "top center", textfont: { size: 11, color: css("--text") }, cliponaxis: false,
      customdata: ps.map((p) => p.id),
      hovertext: ps.map((p) => { const c = S.cmap[p.id]; const s = stab[p.id];
        return `<b>${esc(c.display_name)}</b> (#${c.ballot_number})<br>${esc(c.party)}${s ? `<br>${s.unstable ? "⚠ unstable: " + esc(s.reasons.join("; ")) : "stable under tests"}` : ""}<br>Nearest (original space): ${esc(p.neighbours.slice(0, 3).map((n) => S.cmap[n.id].display_name).join(", "))}`; }),
      hoverinfo: "text",
      marker: {
        size: ps.map((p) => (p.id === S.selected ? 16 : 11)),
        color: pid === "all" ? css("--s1") : partyColor(pid),
        symbol: ps.map((p, i) => { const b = (SYMBOLS[S.cmap[p.id].party_id] || "circle").replace("-open", ""); return unstable[i] ? b + "-open-dot" : SYMBOLS[S.cmap[p.id].party_id] || "circle"; }),
        line: { width: ps.map((p, i) => (p.id === S.selected ? 3 : unstable[i] ? 2 : 1.5)), color: ps.map((p, i) => (p.id === S.selected ? css("--text") : unstable[i] ? partyColor(pid) : css("--surface"))) },
      },
    };
  });
  const ev = m.pca?.explained_variance_ratio || [];
  const xt = useU ? "UMAP 1 (no inherent meaning)" : `PC1 (${fmtPct(ev[0])} of variance)`;
  const yt = useU ? "UMAP 2 (no inherent meaning)" : `PC2 (${fmtPct(ev[1])} of variance)`;
  Plotly.purge("plot"); $("#plot").innerHTML = "";
  Plotly.react("plot", traces, plotLayout(xt, yt), { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["select2d", "lasso2d"] });
  $("#plot").on("plotly_click", (e) => { const id = e.points[0].customdata; select(id); });
  const nUnst = Object.values(stab).filter((s) => s.unstable).length;
  $("#mapMeta").innerHTML = (useU
    ? `UMAP: n_neighbors ${m.umap.n_neighbors}, min_dist ${m.umap.min_dist}, metric ${esc(m.umap.metric)}, random seed ${m.umap.seed}. Axes and inter-cluster distances have no inherent meaning.`
    : `PCA explained variance, first components: ${ev.slice(0, 5).map(fmtPct).join(", ")}. Axes are directions of greatest variance in the embedding space, with no political meaning.`)
    + (S.proj === "umap" && !m.umap?.available ? ` <b>UMAP unavailable here</b>: ${esc(m.umap?.note || "")}. Showing PCA.` : "")
    + (m.same_party_neighbour_rate != null ? `<br>Share of each party candidate's 3 nearest original-space neighbours from the same party: <b>${fmtPct(m.same_party_neighbour_rate)}</b>${S.view === "policy" && S.mode === "with_platform" ? " (inflated by the shared platform vector)" : ""}.` : "")
    + `<br>${m.n} plotted · ${Object.keys(m.insufficient || {}).length} not plotted · ${nUnst} flagged unstable under source removal / re-weighting (ringed markers) · ${Object.values(stab).filter((s) => s.projection_distorted).length} whose 2-D neighbours differ from original-space neighbours (use the neighbour list) · hollow circles = no party.`;
  excludedList(m);
}

function excludedList(m) {
  const ins = m?.insufficient || {};
  const ids = raceCands().map((c) => c.candidate_id).filter((id) => ins[id] || !(m?.points || []).some((p) => p.id === id));
  $("#excluded summary").textContent = `Not plotted (${ids.length})`;
  $("#excludedBody").innerHTML = ids.length ? `<ul>${ids.map((id) => `<li><a href="#" data-c="${id}">${esc(S.cmap[id].display_name)}</a>: ${esc(ins[id] || "no evidence in this view")}</li>`).join("")}</ul>` : "<p>None.</p>";
  $("#excludedBody").querySelectorAll("a[data-c]").forEach((a) => (a.onclick = (e) => { e.preventDefault(); select(a.dataset.c); }));
}

function renderVotes() {
  const V = S.analysis.votes;
  if (!["mayor", "council"].includes(S.race)) {
    Plotly.purge("plot");
    const withBV = raceCands().filter((c) => c.board_votes);
    $("#plot").innerHTML = `<div style="padding:1rem"><p>${S.race === "park" ? "Park Board" : "School Board"} votes are not published as open data. Where minutes PDFs were collected, recorded votes were parsed from them. Coverage is partial, so no agreement map is drawn. Open a candidate's profile to see each parsed motion.</p>
      ${withBV.length ? `<table><tr><th>Candidate</th><th>In favour</th><th>Opposed</th><th>Absent</th><th>Abstained</th><th>Recused</th></tr>${withBV.map((c) => { const k = c.board_votes.counts || {}; return `<tr data-c="${c.candidate_id}"><td><a href="#" data-c="${c.candidate_id}">${esc(c.display_name)}</a></td><td>${k.in_favour || 0}</td><td>${k.opposed || 0}</td><td>${k.absent || 0}</td><td>${k.abstained || 0}</td><td>${k.recused || 0}</td></tr>`; }).join("")}</table>` : "<p>No minutes with named votes for candidates in this race were collected.</p>"}</div>`;
    $("#plot").querySelectorAll("a[data-c]").forEach((a) => (a.onclick = (e) => { e.preventDefault(); select(a.dataset.c); }));
    $("#mapMeta").innerHTML = ""; $("#mapNote").innerHTML = ""; excludedList(null); return;
  }
  $("#mapNote").innerHTML = "Council votes only (City open data). Agreement uses matching contested motions where both members cast For/Against. Absent, abstain, recusal (declared conflict) and ineligible are <b>not</b> counted as votes against. Different councils are never compared: one map per term.";
  const terms = Object.keys(V);
  const traces = [];
  const colors = [css("--s1"), css("--s2")];
  terms.forEach((t, ti) => {
    const e = V[t];
    traces.push({ type: "scatter", mode: "markers+text", name: `Council ${t}`, xaxis: ti ? "x2" : "x", yaxis: ti ? "y2" : "y",
      x: e.coords.map((c) => c[0]), y: e.coords.map((c) => c[1]), text: e.candidates.map((c) => S.cmap[c].display_name),
      textposition: "top center", textfont: { size: 10, color: css("--text") }, customdata: e.candidates, cliponaxis: false,
      marker: { size: 11, color: e.candidates.map((c) => partyColor(S.cmap[c].party_id)), symbol: e.candidates.map((c) => SYMBOLS[S.cmap[c].party_id]), line: { width: 1.5, color: css("--surface") } },
      hovertext: e.candidates.map((c) => `${esc(S.cmap[c].display_name)} · ${esc(S.cmap[c].party)} (2026 ballot party)`), hoverinfo: "text" });
  });
  const L = plotLayout("", "");
  L.grid = { rows: 1, columns: 2, pattern: "independent" };
  L.showlegend = false;
  L.xaxis2 = { ...L.xaxis, title: { text: `2022-2026: dim 1 (${fmtPct(V["2022-2026"].explained[0])})` } };
  L.yaxis2 = { ...L.yaxis };
  L.xaxis.title = { text: `2018-2022: dim 1 (${fmtPct(V["2018-2022"].explained[0])})` };
  L.annotations = terms.map((t, i) => ({ text: `Council ${t}`, xref: "paper", yref: "paper", x: i ? 0.78 : 0.22, y: 1.0, showarrow: false, font: { size: 12 } }));
  Plotly.purge("plot"); $("#plot").innerHTML = "";
  Plotly.react("plot", traces, L, { responsive: true, displaylogo: false });
  $("#plot").on("plotly_click", (e) => select(e.points[0].customdata));
  const e = V["2022-2026"];
  const grid = `<table class="votegrid"><tr><th>Agreement on contested motions, 2022-26 (shared votes)</th>${e.candidates.map((c) => `<th>${esc(S.cmap[c].display_name.split(" ").slice(-1)[0])}</th>`).join("")}</tr>` +
    e.candidates.map((a, i) => `<tr><th>${esc(S.cmap[a].display_name)}</th>${e.candidates.map((b, k) => `<td title="${e.overlap[i][k]} shared contested votes">${i === k ? "–" : fmtPct(e.agreement[i][k])}</td>`).join("")}</tr>`).join("") + "</table>";
  $("#mapMeta").innerHTML = `Classical MDS of (1 − agreement). Party colours show the <b>2026 ballot</b> party; several members changed party during the term (see each profile's affiliations). Source: <a href="${esc(S.meta.votes_dataset.url)}" target="_blank" rel="noopener">City of Vancouver open data</a> (modified ${esc(S.meta.votes_dataset.modified)}); minutes are the official record.<div class="scrollx">${grid}</div>`;
  const others = raceCands().filter((c) => !c.vote_summary).map((c) => c.candidate_id);
  $("#excluded summary").textContent = `No Council voting record (${others.length})`;
  $("#excludedBody").innerHTML = `<p>These candidates have not sat on City Council since electronic voting began in 2016. That means <b>no voting record</b>, not absence or opposition.</p><ul>${others.map((id) => `<li>${esc(S.cmap[id].display_name)}</li>`).join("")}</ul>`;
}

function rowData(c) {
  const f = c.profile.features, m = currentMap();
  const st = (S.analysis.races[c.race]?.[`${S.view}:${S.view === "policy" ? S.mode : "individual"}`] || {}).stability?.[c.candidate_id];
  const plotted = (m?.points || []).some((p) => p.id === c.candidate_id);
  return { ballot_number: c.ballot_number, display_name: c.display_name, party: c.party, years: f.years_elected_office,
    mgmt: f.management_responsibility, votes: f.vote_record, topics: c.profile.coverage.topics_with_positions, sources: f.n_sources,
    mapstatus: S.view === "votes" ? (c.vote_summary ? "record" : "no record") : !plotted ? "insufficient evidence" : st?.unstable ? "unstable" : "plotted" };
}

function table() {
  const rows = raceCands().filter((c) => !S.search || `${c.display_name} ${c.name_cjk || ""} ${c.party} ${c.ballot_number}`.toLowerCase().includes(S.search))
    .map((c) => [c, rowData(c)]);
  const [k, dir] = S.sort;
  rows.sort((a, b) => { const x = a[1][k], y = b[1][k]; return (x == null) - (y == null) || (x > y ? dir : x < y ? -dir : 0); });
  $("#tableTitle").textContent = `Candidates (${rows.length})`;
  const vr = { available: "Council votes", board_minutes_partial: "board minutes (partial)", not_collected_board_minutes: "board minutes (none parsed)", none_no_council_seat: "none" };
  $("#ctable tbody").innerHTML = rows.map(([c, r]) => `<tr data-c="${c.candidate_id}" class="${c.candidate_id === S.selected ? "sel" : ""}">
    <td><input type="checkbox" aria-label="Compare ${esc(c.display_name)}" ${S.compare.includes(c.candidate_id) ? "checked" : ""} data-cmp="${c.candidate_id}"></td>
    <td class="num">${c.ballot_number}</td><td>${esc(c.display_name)}${c.name_cjk ? " " + esc(c.name_cjk) : ""}${c.also_on_ballot_as.length ? ' <span class="badge">also on another ballot</span>' : ""}</td>
    <td><span class="sw" style="background:${partyColor(c.party_id)}"></span> ${esc(c.party)}</td>
    <td class="num">${r.years == null ? '<span class="status">not documented</span>' : r.years}</td>
    <td>${r.mgmt ? "documented" : '<span class="status">not documented</span>'}</td>
    <td>${esc(vr[r.votes] || r.votes)}</td><td class="num">${r.topics}/13</td><td class="num">${r.sources}</td>
    <td>${r.mapstatus === "unstable" ? '<span class="badge warn">unstable</span>' : r.mapstatus === "insufficient evidence" ? '<span class="badge bad">insufficient evidence</span>' : `<span class="badge">${esc(r.mapstatus)}</span>`}</td>
    <td>${VERIF[c.ballot_verification]}</td></tr>`).join("");
  document.querySelectorAll("#ctable tbody tr").forEach((tr) => (tr.onclick = (e) => {
    if (e.target.dataset.cmp) { toggleCompare(e.target.dataset.cmp); return; }
    select(tr.dataset.c);
  }));
}

function toggleCompare(id) {
  const i = S.compare.indexOf(id);
  if (i >= 0) S.compare.splice(i, 1); else { S.compare.push(id); if (S.compare.length > 3) S.compare.shift(); }
  renderCompare(); table(); if (S.view !== "votes") renderMap();
}

function select(id) { S.selected = id; if (S.race !== "all" && S.cmap[id].race !== S.race) { S.race = S.cmap[id].race; render(); return; } panel(id); table(); if (S.view !== "votes") renderMap(); setHash(); }

function citeHTML(it) {
  const s = S.sources[it.source_id] || {};
  const typeLbl = { individual_position: "position", party_commitment: "party platform", elected_experience: "elected office",
    professional_experience: "experience", education: "education", reported_action: "reported action", claimed_outcome: "claimed outcome (unverified)", commentary: "commentary", third_party_assessment: "third-party assessment", shared_campaign_language: "shared campaign language" }[it.evidence_type] || it.evidence_type;
  const forms = (it.housing_forms || []).length ? ` · housing form: ${it.housing_forms.map(esc).join(", ")}` : "";
  return `<div class="ev ${esc(it.evidence_type)}"><p class="q">“${esc(it.claim)}”</p>
    ${it.perspective && (it.evidence_type === "third_party_assessment" || it.evidence_type === "commentary" || it.verification_status === "questionnaire_response") ? `<div class="src"><b>Publisher perspective:</b> ${esc(it.perspective)}</div>` : ""}
    <div class="src">${esc(typeLbl)} · ${esc(it.verification_status.replaceAll("_", " "))} · uncertainty ${esc(it.uncertainty)}${forms}<br>
    <a href="/source/${esc(it.source_id)}?p=${encodeURIComponent(it.passage_id)}#${encodeURIComponent(it.passage_id)}" target="_blank" rel="noopener">${esc(s.publisher || "source")}${it.page ? ", p." + it.page : ""} · passage ${esc(it.passage_id.split(":").slice(1).join(":"))}</a>
    · <a href="${esc(s.url)}" target="_blank" rel="noopener">original ↗</a>${s.via === "wayback" ? " (via Internet Archive)" : ""} · ${esc(it.date || "")}</div></div>`;
}

function stabilityHTML(id) {
  const k = `${S.view}:${S.view === "policy" ? S.mode : "individual"}`;
  const st = S.analysis.races[S.cmap[id].race]?.[k]?.stability?.[id];
  const m = S.analysis.races[S.cmap[id].race]?.[k];
  if (!m) return "";
  if (m.insufficient && m.insufficient[id]) return `<p><span class="badge bad">Insufficient evidence for this view</span> ${esc(m.insufficient[id])}</p>`;
  if (!st) return "";
  const f = (x) => (x == null ? "–" : x);
  return `<details><summary>${st.unstable ? '<span class="badge warn">Unstable placement</span> ' + esc(st.reasons.join("; ")) : '<span class="badge ok">Stable under sensitivity tests</span>'}</summary>
    <dl class="kv"><dt>Source-dropout displacement</dt><dd>${f(st.dropout_displacement)} (median, × spread)</dd>
    <dt>Neighbours kept (dropout)</dt><dd>${f(st.dropout_neighbour_jaccard)}</dd>
    <dt>Topic re-weight displacement</dt><dd>${f(st.weight_displacement)}</dd>
    <dt>UMAP seed displacement</dt><dd>${f(st.umap_seed_displacement)}</dd>
    <dt>2-D vs original neighbours</dt><dd>PCA ${f(st.pca_neighbour_agreement)} · UMAP ${f(st.umap_neighbour_agreement)}${st.projection_distorted ? ' <span class="badge">2-D view distorts neighbours: trust the list, not the dot</span>' : ""}</dd>
    <dt>Platform toggle neighbours</dt><dd>${f(st.platform_toggle_neighbour_jaccard)}</dd></dl></details>`;
}

async function panel(id) {
  const c = S.cmap[id], p = c.profile, f = p.features, tl = S.meta.topic_labels;
  const m = currentMap();
  const pt = (m?.points || []).find((x) => x.id === id);
  const links = c.campaign_links.map((l) => `<a href="${esc(l.url)}" target="_blank" rel="noopener">${esc(l.label)}</a>`).join(" · ");
  const docs = c.official_documents.map((d) => `<a href="${esc(d.url)}" target="_blank" rel="noopener">${esc(d.label)}</a>`).join(" · ");
  const affs = c.affiliations.length ? `<ul>${c.affiliations.map((a) => `<li>${esc(a.organization)}: ${esc(a.role || "")} ${a.start ? esc(a.start) : "start not documented"} – ${a.end ? esc(a.end) : "present / end not documented"} <a href="${esc(a.source_url)}" target="_blank" rel="noopener">source ↗</a></li>`).join("")}</ul>` : '<p class="status">No dated affiliations documented in collected sources.</p>';
  const vs = c.vote_summary;
  const vhtml = vs ? `<dl class="kv"><dt>Recorded motions</dt><dd>${vs.motions_recorded} (${esc(vs.first_vote)} → ${esc(vs.last_vote)})</dd>
      ${Object.entries(vs.counts).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v}${["Absent", "Abstain", "No Vote", "Declared Conflict", "Ineligible"].includes(k) ? ' <span class="status">(not a vote against)</span>' : ""}</dd>`).join("")}
      <dt>Contested motions</dt><dd>${vs.contested_motions}</dd></dl><p class="status">Source: <a href="${esc(vs.source_url)}" target="_blank" rel="noopener">City of Vancouver council voting records</a>. Minutes are the official record.</p>`
    : `<p class="status">${esc(f.vote_record_note)}</p>`;
  const bv = c.board_votes;
  const VL = { in_favour: "In favour", opposed: "Opposed", absent: "Absent (not a vote)", abstained: "Abstained (not a vote)", recused: "Recused / conflict (not a vote)" };
  const bvhtml = bv ? `<p class="status">${esc(bv.note)}</p><dl class="kv">${Object.entries(bv.counts || {}).map(([k, v]) => `<dt>${esc(VL[k] || k)}</dt><dd>${v}</dd>`).join("")}</dl>
      <details><summary>${bv.motions.length} parsed motion(s)</summary>${bv.motions.map((m) => `<div class="ev"><p class="q">${esc(m.date || "")} · <b>${esc(VL[m.vote] || m.vote)}</b> · ${esc(m.result)}${m.is_amendment ? " · amendment" : ""}</p><div class="src">${esc(m.text)} · <a href="/source/${esc(m.source_id)}" target="_blank" rel="noopener">minutes</a></div></div>`).join("")}</details>` : "";
  const pol = Object.entries(p.policy).map(([t, v]) => `<div class="topicrow"><div class="tname">${esc(tl[t] || t)} <span class="status">${v.status === "position_found" ? `${v.n_total} statement(s)` : v.status === "party_platform_only" ? "party platform only" : "not found in collected sources"}</span></div>${v.items.map(citeHTML).join("")}</div>`).join("");
  $("#panel").innerHTML = `<div class="cand-head"><h2>${esc(c.display_name)} ${c.name_cjk ? esc(c.name_cjk) : ""}</h2>
    <div class="meta">#${c.ballot_number} · ${esc(c.office)} · ${esc(c.party)}${c.party_label_on_ballot ? "" : " (no party on ballot)"}</div></div>
    <div class="badges">${VERIF[c.ballot_verification]}${c.official_registration ? `<span class="badge ok">Elections BC registered: ${esc(c.official_registration.name_as_registered)}${c.official_registration.affiliation_registered ? " (" + esc(c.official_registration.affiliation_registered) + ")" : ""}</span>` : ""}
      ${c.also_on_ballot_as.map((o) => `<span class="badge">also running: ${esc(S.cmap[o].office)}</span>`).join("")}
      <span class="badge">${p.coverage.topics_with_positions}/13 topics with own statements</span><span class="badge">${f.n_sources} sources</span></div>
    ${S.view !== "votes" ? stabilityHTML(id) : ""}
    <p style="font-size:.84rem">${links ? "Campaign links (from official page): " + links + "<br>" : ""}${docs ? "Official documents: " + docs + "<br>" : ""}<a href="${esc(c.official_candidate_page)}" target="_blank" rel="noopener">Official candidate page ↗</a></p>
    <h3>Structured features</h3><dl class="kv"><dt>Years in elected office</dt><dd>${f.years_elected_office ?? "not documented"} <span class="status">(${esc(f.years_basis)})</span></dd>
      <dt>Management responsibility</dt><dd>${f.management_responsibility ? "documented" : "not documented (missing, not zero)"}</dd>
      <dt>Vote record</dt><dd>${esc(f.vote_record.replaceAll("_", " "))}</dd><dt>Claims</dt><dd>${f.n_claims} (${f.n_canonical_claims} after de-duplication)</dd></dl>
    ${pt ? `<h3>Nearest in original embedding space (${esc(S.meta.views[S.view] || S.view)})</h3><div id="nbs">${pt.neighbours.map((n) => `<div class="nb"><a href="#" data-c="${n.id}">${esc(S.cmap[n.id].display_name)}</a> <span class="status">${esc(S.cmap[n.id].party)} · cosine ${n.cosine.toFixed(3)}</span> <button class="btn" data-why="${n.id}">why?</button><div class="support" hidden></div></div>`).join("")}</div>` : ""}
    <h3>Affiliations</h3>${affs}
    <h3>Experience & governance</h3>${p.experience.length ? p.experience.map(citeHTML).join("") : '<p class="status">No experience statements found in collected sources.</p>'}
    <h3>Recorded votes</h3>${vhtml}${bvhtml}${c.caucus_departures ? `<h3>Departures from ABC caucus majority</h3><p style="font-size:.85rem">${c.caucus_departures.n} of ${c.caucus_departures.of} contested votes cast while an ABC caucus member. <span class="status">${esc(c.caucus_departures.method)} Shown as evidence of independent votes; a departure does not mean agreement with any other party.</span></p><details><summary>List</summary>${c.caucus_departures.items.map((d) => `<div class="ev"><p class="q">${esc(d.date)} · voted <b>${esc(d.vote)}</b>; caucus majority ${esc(d.caucus_majority)} (${esc(d.caucus_split)}) · ${esc(d.decision)}</p><div class="src">${esc(d.motion)} · <a href="${esc(S.meta.votes_dataset.url)}" target="_blank" rel="noopener">open data</a></div></div>`).join("")}</details>` : ""}
    <h3>Reported actions & claimed outcomes</h3>${p.actions.length ? p.actions.map(citeHTML).join("") : '<p class="status">None found in collected sources.</p>'}
    <h3>Policy positions by topic</h3>${pol}
    ${p.shared_language.length ? `<h3>Shared campaign language (same text on other ${esc(c.party)} candidates' pages; excluded from individual maps)</h3>${p.shared_language.map(citeHTML).join("")}` : ""}
    ${p.assessments.length ? `<h3>Third-party assessments (labelled by publisher)</h3>${p.assessments.map(citeHTML).join("")}` : ""}
    ${p.commentary.length ? `<h3>Commentary (secondary context, not evidence)</h3>${p.commentary.map(citeHTML).join("")}` : ""}
    ${c.gaps.length ? `<h3>Coverage gaps logged</h3><ul>${c.gaps.map((g) => `<li>${esc(g.what)} <span class="status">(tried: ${esc(g.tried)})</span></li>`).join("")}</ul>` : ""}
    <p><button class="btn" id="cmpBtn">${S.compare.includes(id) ? "Remove from" : "Add to"} comparison</button> <a href="/api/claims/${esc(id)}" target="_blank">All claims (JSON)</a></p>`;
  $("#cmpBtn").onclick = () => { toggleCompare(id); panel(id); };
  document.querySelectorAll("#panel a[data-c]").forEach((a) => (a.onclick = (e) => { e.preventDefault(); select(a.dataset.c); }));
  document.querySelectorAll("#panel button[data-why]").forEach((b) => (b.onclick = async () => {
    const box = b.nextElementSibling; box.hidden = !box.hidden; if (box.dataset.loaded) return;
    box.textContent = "Loading…";
    const view = S.view === "combined" ? "policy" : S.view;
    const sp = await j(`/api/support?${new URLSearchParams({ view, a: id, b: b.dataset.why, mode: S.mode })}`);
    box.dataset.loaded = 1;
    box.innerHTML = sp.length ? sp.map((x) => `<div><b>${esc(tl[x.topic] || x.topic)}</b> (cosine ${x.cosine})<br>
      ${esc(c.display_name)}: “${esc(x.a.text)}” <a href="/source/${esc(x.a.source_id)}?p=${encodeURIComponent(x.a.passage_id)}#${encodeURIComponent(x.a.passage_id)}" target="_blank">[cite]</a><br>
      ${esc(S.cmap[b.dataset.why].display_name)}: “${esc(x.b.text)}” <a href="/source/${esc(x.b.source_id)}?p=${encodeURIComponent(x.b.passage_id)}#${encodeURIComponent(x.b.passage_id)}" target="_blank">[cite]</a></div>`).join("<hr>")
      : "No same-topic passage pair found.";
  }));
}

function renderCompare() {
  const el = $("#compare");
  el.hidden = !S.compare.length;
  if (!S.compare.length) return;
  const cs = S.compare.map((id) => S.cmap[id]), tl = S.meta.topic_labels;
  const row = (label, f) => `<tr><th>${label}</th>${cs.map((c) => `<td>${f(c)}</td>`).join("")}</tr>`;
  const topicCell = (c, t) => { const v = c.profile.policy[t]; return v.status === "position_found" ? v.items.map(citeHTML).join("") : `<span class="status">${v.status === "party_platform_only" ? "party platform only" : "not found"}</span>`; };
  $("#compareBody").innerHTML = `<table><thead><tr><th></th>${cs.map((c) => `<th>${esc(c.display_name)}<br><span class="status">#${c.ballot_number} ${esc(c.office)} · ${esc(c.party)}</span></th>`).join("")}</tr></thead><tbody>
    ${row("Years elected", (c) => c.profile.features.years_elected_office ?? '<span class="status">not documented</span>')}
    ${row("Vote record", (c) => esc(c.profile.features.vote_record.replaceAll("_", " ")))}
    ${row("Experience", (c) => c.profile.experience.slice(0, 3).map(citeHTML).join("") || '<span class="status">none found</span>')}
    ${row("Actions / outcomes", (c) => c.profile.actions.slice(0, 2).map(citeHTML).join("") || '<span class="status">none found</span>')}
    ${S.meta.config.topics.map((t) => row(esc(tl[t] || t), (c) => topicCell(c, t))).join("")}</tbody></table>`;
}

function exportsList() {
  const files = [["candidates.csv", "Candidates (CSV)"], ["candidates.json", "Candidates with profiles (JSON)"], ["claims.csv", "All claims (CSV)"],
    ["claims.jsonl", "All claims (JSONL)"], ["sources.csv", "Source registry with hashes (CSV)"], ["council_votes_candidates.csv", "Council votes of candidates (CSV)"],
    ["council_vote_agreement_pairs.csv", "Pairwise vote agreement (CSV)"], ["chunk_embeddings.npy", "Chunk embeddings (NumPy)"],
    ["chunk_embeddings_index.csv", "Chunk embedding index (CSV)"], ["topic_vectors_policy.npz", "Topic vectors: policy (NPZ)"],
    ["topic_vectors_platform.npz", "Topic vectors: platform (NPZ)"], ["topic_vectors_experience.npz", "Topic vectors: experience (NPZ)"],
    ["topic_vectors_actions.npz", "Topic vectors: actions (NPZ)"], ["analysis.json", "All maps, neighbours & stability (JSON)"],
    ["analysis_config.yaml", "Analysis configuration (YAML)"], ["embedding_model.json", "Embedding model record"],
    ["access_log.jsonl", "Fetch / access log (incl. blocked sources)"], ["methods_report.html", "Methods report (HTML)"],
    ["methods_report.md", "Methods report (Markdown)"], ["vancouver-2026-explorer-export.zip", "Everything (ZIP)"]];
  const e = S.meta.embedding;
  $("#exportsBody").innerHTML = `<p class="hint">Embedding model: ${esc(e.model)} (${esc(e.backend)}, ${e.dimensions} dims, revision ${esc((e.revision || "n/a").slice(0, 12))}). Config sha256 ${esc(S.meta.config_sha256.slice(0, 12))}. Generated ${esc(S.meta.generated_at)}.</p>
    <ul>${files.map(([f, l]) => `<li><a href="/exports/${f}" download>${esc(l)}</a></li>`).join("")}</ul>
    <h3>Inaccessible sources (${S.meta.inaccessible_sources.length})</h3><details><summary>Show log</summary><ul style="columns:1">${S.meta.inaccessible_sources.map((a) => `<li>${esc(a.outcome)}: ${esc(a.url)}</li>`).join("")}</ul></details>`;
}

window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => render());
init().catch((e) => { document.body.insertAdjacentHTML("afterbegin", `<p style="padding:1rem;color:red">Failed to load: ${esc(e.message)}</p>`); });
