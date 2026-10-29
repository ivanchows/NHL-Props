// PTO closing-odds pull for NHL SOG + saves (2026-10-29). Run in a PTO odds-screen tab with `nhl_props.pto serve` running.
// Port of ../NBAProps/pto-scanner/pull_resume_combos.js. Uses the PAGE'S Apollo client (it signs requests).
// Settings in window.PTO_PULL before running: {from, to (unix s), blackout: [[a, b], ...] unix s, skip: {"<market>__<gameId>": 1}}.
// Pauses (sends nothing) inside blackout windows: the NBA snapshot owns port 8766 then. Stop: window.__pull.stop = true.
// 🔴 Finding gql: only require webpack modules whose source mentions "Document" (requiring all 12k froze the tab).
(() => {
  const CFG = window.PTO_PULL || {};
  const BOOKS = new Set(["FAN_DUEL", "MGM", "DRAFT_KINGS", "CAESARS", "PINNACLE", "BET_ONLINE_AG", "BET_365", "PROPHET_BETTING",
    "FLIFF", "THE_SCORE", "BET_RIVERS", "BALLY_BET", "BOOKMAKER_EU", "MY_BOOKIE", "FANATICS", "NO_VIG", "ONYX_ODDS",
    "PROP_BUILDER", "KALSHI", "HARD_ROCK_IN"]);
  const MK = CFG.markets || ["PLAYER_PROP_SHOTS_ON_GOAL", "PLAYER_PROP_SHOTS_SAVED"];
  const FROM = CFG.from, TO = CFG.to || Math.floor(Date.now() / 1000), BL = CFG.blackout || [], SKIP = CFG.skip || {};
  const S = (window.__pull = { named: new Set(), pending: new Set(), errors: [], streak: 0, stop: false, phase: "find", done: 0, total: 0, games: 0 });
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const blocked = () => { const t = Date.now() / 1000; return BL.some(([a, b]) => a <= t && t < b); };
  const post = (m) => window.postMessage({ ptohist: { raw: true, nhl: true, ...m } }, "*");

  let A0 = window.__apollo;
  if (!A0) for (const el of document.querySelectorAll("*")) {
    const k = Object.keys(el).find((k) => k.startsWith("__reactFiber"));
    for (let f = k && el[k]; f; f = f.return) { const c = f.memoizedProps?.client; if (c?.query && c?.link) { A0 = c; break; } }
    if (A0) break;
  }
  let gql = window.__gql;
  if (!gql) {
    let req; const ch = Object.keys(window).find((k) => k.startsWith("webpackChunk"));
    window[ch].push([[Symbol("pto" + Math.random())], {}, (r) => { req = r; }]);
    for (const id of Object.keys(req.m).filter((id) => { const s = req.m[id].toString(); return s.includes('"Document"') && s.includes("fragment") && s.length < 60000; })) {
      let m; try { m = req(id); } catch (e) { continue; }
      if (!m || typeof m !== "object") continue;
      for (const k of Object.keys(m)) { try { const d = typeof m[k] === "function" && m[k](["query X { a }"]); if (d?.kind === "Document") { gql = m[k]; break; } } catch (e) {} }
      if (gql) break;
    }
  }
  const apollo = { query: (o) => Promise.race([A0.query({ ...o, fetchPolicy: "no-cache" }), new Promise((_, rej) => setTimeout(() => rej(new Error("timeout 60s")), 60000))]) };
  const G = gql([`query PtoGames($league: LeagueEnum!, $request: InputGameRequestType) { games(league: $league, request: $request) { ... on IceHockeyGameType { id startDateTime isCompleted isCancelled awayTeam { abbreviations } homeTeam { abbreviations } } } }`]);
  const Q = gql([`query PtoSlim($league: LeagueEnum!, $t: BetMarketTypeEnumTypeTwo!, $g: Guid!) { m: completedBetMarket(league: $league, gameId: $g, betMarketTypeEnum: $t) { conditions { playerId overUnder betValue } listings { siteId americanOdds maxWager } isWinner isPush isVoided } }`]);
  const PQ = gql([`query GetPlayers($ids: [UInt]!, $league: LeagueEnum!) { players(ids: $ids, league: $league) { id name } }`]);
  const waitOpen = async () => { while (blocked() && !S.stop) { S.phase = "blackout"; await sleep(20000); } };
  // Retries forever on errors (30 s apart) but gives up after 20 in a row.
  const q = async (o) => { for (;;) { if (S.stop) throw new Error("stopped"); await waitOpen();
    try { const r = await apollo.query(o); S.streak = 0; return r; }
    catch (e) { S.errors.push(String(e).slice(0, 120)); if (++S.streak > 20) { S.stop = true; throw e; } await sleep(30000); } } };

  (async () => {
    S.phase = "games";
    const seen = new Map(); let after = FROM;
    for (;;) {
      const g = (await q({ query: G, variables: { league: "NHL", request: { take: 50, orderBy: "ASC", afterDateTime: after } } })).data.games;
      let fresh = 0;
      for (const x of g) if (!seen.has(x.id)) { seen.set(x.id, x); fresh++; }
      S.games = seen.size;
      if (!g.length || g.at(-1).startDateTime > TO) break;
      after = fresh ? g.at(-1).startDateTime : after + 1;
      await sleep(1200);
    }
    const games = [...seen.values()].filter((x) => x.startDateTime >= FROM && x.startDateTime <= TO && x.isCompleted && !x.isCancelled)
      .map((x) => ({ id: x.id, start: x.startDateTime, away: x.awayTeam?.abbreviations?.[0], home: x.homeTeam?.abbreviations?.[0] }));
    await waitOpen();
    post({ kind: "games", games });
    S.phase = "odds"; S.total = games.length * MK.length;
    const names = async () => { const ids = [...S.pending]; if (!ids.length) return;
      const r = await q({ query: PQ, variables: { ids, league: "NHL" } });
      await waitOpen(); post({ kind: "players", players: r.data.players.map((p) => ({ id: p.id, name: p.name })) });
      ids.forEach((i) => { S.named.add(i); S.pending.delete(i); }); };
    for (const [gi, g] of games.entries()) {
      for (const m of MK) {
        if (SKIP[`${m}__${g.id}`]) { S.done++; continue; }
        const r = await q({ query: Q, variables: { league: "NHL", t: m, g: g.id } });
        const rows = [];
        for (const b of r.data.m || []) {
          const c = b.conditions[0]; if (c?.playerId == null || b.isVoided) continue;
          if (!S.named.has(c.playerId)) S.pending.add(c.playerId);
          for (const l of b.listings) if (l.americanOdds != null && BOOKS.has(l.siteId))
            rows.push([c.playerId, c.overUnder.toLowerCase(), c.betValue, l.siteId, l.americanOdds, l.maxWager, b.isWinner, b.isPush]);
        }
        await waitOpen();
        post({ kind: "odds", market: m, game_id: g.id, rows, note: `game ${gi + 1}/${games.length}` });
        S.done++;
        await sleep(1200);
      }
      if (S.pending.size >= 40 || gi === games.length - 1) { await names(); await sleep(1200); }
    }
    post({ kind: "done" }); S.phase = "finished";
  })().catch((e) => { S.phase = "failed: " + String(e).slice(0, 200); });
  return "started";
})();
