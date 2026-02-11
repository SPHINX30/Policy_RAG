// ==UserScript==
// @name         Embassy Month Sweep + Greenday Alarm + 2-min Hard Reload
// @namespace    http://tampermonkey.net/
// @version      6.1
// @description  Sets location, sweeps Jan..TARGET_MONTH reliably (native setter + verify + prev/next fallback), detects ONLY greenday in group-first, alarms + downloads log once per new results, hard-reloads page every 2 minutes until slots found.
// @match        https://www.usvisascheduling.com/en-US/schedule/?reschedule=true
// @run-at       document-idle
// @grant        none
// ==/UserScript==

(function () {
  "use strict";

  // ===============================
  // CONFIG
  // ===============================
  const LOCATION_VALUE = "2b6bf614-b0db-ec11-a7b4-001dd80234f6"; // ULAANBAATAR

  // Month index: 0=Jan, 1=Feb, 2=Mar, 3=Apr, 4=May, 5=Jun, 6=Jul, 7=Aug, 8=Sep, 9=Oct, 10=Nov, 11=Dec
  const TARGET_YEAR = 2026;
  const TARGET_MONTH = 8; // <-- CHANGE THIS (7=Aug). If you want June, use 5.

  const SWEEP_START_MONTH = 1; // Jan (0)
  const SWEEP_END_MONTH = TARGET_MONTH; // inclusive

  const START_DELAY_MS = 1500;
  const RETRY_EVERY_MS = 800;
  const MAX_RETRY_MS = 30000;

  const RENDER_WAIT_MS = 900;
  const SET_WAIT_TIMEOUT_MS = 5000;

  const SWEEP_INTERVAL_MS = 120000; // run full sweep every 2 minutes

  // Hard full page reload (until slots found)
  const HARD_RELOAD_MS = 180000; // 2 minutes

  // Alarm
  const ALARM_FREQ_HZ = 2000;
  const ALARM_VOLUME = 1.8;
  const BEEP_MS = 500;
  const GAP_MS = 400;
  const LOOP_MS = 1800;

  const MONTH_NAMES = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
  ];

  // ===============================
  // STATE
  // ===============================
  let lastResultsSignature = "";
  let sweepRunning = false;

  let audioCtx = null;
  let audioArmed = false;
  let alarmLoopId = null;

  let slotsFoundEver = false;
  let hardReloadTimer = null;
  let sweepTimer = null;

  // ===============================
  // Helpers
  // ===============================
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  function nowStamp() {
    const d = new Date();
    const pad = (n) => String(n).padStart(2, "0");
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}_${pad(d.getHours())}-${pad(d.getMinutes())}-${pad(d.getSeconds())}`;
  }

  function downloadTxt(filename, text) {
    const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  // ===============================
  // Audio (needs 1 real user gesture; cannot bypass)
  // ===============================
  function ensureAudioContext() {
    if (!audioCtx) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      audioCtx = new AudioCtx();
    }
    return audioCtx;
  }

  async function armAudioOnce() {
    if (audioArmed) return;
    try {
      const ctx = ensureAudioContext();
      if (ctx.state === "suspended") await ctx.resume();
      audioArmed = true;
      console.log("🔊 Audio armed (first user gesture captured).");
    } catch {
      console.log("❌ Audio still blocked.");
    }
  }

  window.addEventListener("pointerdown", armAudioOnce, {
    once: true,
    capture: true,
  });
  window.addEventListener("keydown", armAudioOnce, {
    once: true,
    capture: true,
  });

  function startAlarm() {
    if (!audioArmed) {
      console.log("🔇 Slots found but audio blocked (no user gesture yet).");
      document.title = "🚨 SLOT FOUND (CLICK ONCE)";
      return;
    }
    if (alarmLoopId) return;

    document.title = "🚨 SLOT FOUND!";
    console.log("🚨🚨🚨 ALARM STARTED 🚨🚨🚨");

    const ctx = ensureAudioContext();
    if (ctx.state === "suspended") ctx.resume().catch(() => {});

    const beep = () => {
      const o = ctx.createOscillator();
      const g = ctx.createGain();
      o.type = "square";
      o.frequency.value = ALARM_FREQ_HZ;
      g.gain.value = ALARM_VOLUME;
      o.connect(g);
      g.connect(ctx.destination);
      o.start();
      setTimeout(() => o.stop(), BEEP_MS);
    };

    beep();
    setTimeout(beep, GAP_MS);

    alarmLoopId = setInterval(() => {
      beep();
      setTimeout(beep, GAP_MS);
    }, LOOP_MS);
  }

  // ===============================
  // DOM helpers
  // ===============================
  function getDatepickerRoot() {
    return document.querySelector("#datepicker");
  }

  function getGroupFirst() {
    const dp = getDatepickerRoot();
    return dp ? dp.querySelector(".ui-datepicker-group-first") : null;
  }

  function getSelects() {
    const g = getGroupFirst() || getDatepickerRoot();
    if (!g) return null;
    const yearSel =
      g.querySelector("select.ui-datepicker-year") ||
      document.querySelector("select.ui-datepicker-year");
    const monthSel =
      g.querySelector("select.ui-datepicker-month") ||
      document.querySelector("select.ui-datepicker-month");
    if (!yearSel || !monthSel) return null;
    return { yearSel, monthSel };
  }

  function getDisplayedYM() {
    const g = getGroupFirst() || getDatepickerRoot();
    if (!g) return null;

    const yearSel = g.querySelector("select.ui-datepicker-year");
    const monthSel = g.querySelector("select.ui-datepicker-month");

    if (yearSel && monthSel) {
      return {
        year: parseInt(yearSel.value, 10),
        month: parseInt(monthSel.value, 10), // 0=Jan
      };
    }

    const title = g.querySelector(".ui-datepicker-title");
    if (title)
      return { year: NaN, month: NaN, title: title.textContent.trim() };

    return null;
  }

  function getDisplayedTitleText() {
    const g = getGroupFirst() || getDatepickerRoot();
    const title = g ? g.querySelector(".ui-datepicker-title") : null;
    return title ? title.textContent.trim() : "(no title)";
  }

  function nativeSetSelectValue(sel, valueStr) {
    const desc = Object.getOwnPropertyDescriptor(
      HTMLSelectElement.prototype,
      "value",
    );
    if (desc && typeof desc.set === "function") {
      desc.set.call(sel, valueStr);
    } else {
      sel.value = valueStr;
    }
    sel.dispatchEvent(new Event("input", { bubbles: true }));
    sel.dispatchEvent(new Event("change", { bubbles: true }));
  }

  function clickPrevNext(direction /* "prev"|"next" */) {
    const dp = getDatepickerRoot();
    if (!dp) return false;
    const btn = dp.querySelector(
      direction === "prev" ? "a.ui-datepicker-prev" : "a.ui-datepicker-next",
    );
    if (!btn) return false;
    btn.click();
    return true;
  }

  function ymToIndex(year, month) {
    return year * 12 + month;
  }

  // ===============================
  // Reliable month/year set
  // ===============================
  async function setMonthYearReliable(targetYear, targetMonth) {
    const selects = getSelects();
    if (!selects) return false;

    // Try select-based set
    nativeSetSelectValue(selects.yearSel, String(targetYear));
    nativeSetSelectValue(selects.monthSel, String(targetMonth));

    // Wait until UI actually reflects target
    const start = Date.now();
    while (Date.now() - start < SET_WAIT_TIMEOUT_MS) {
      await sleep(RENDER_WAIT_MS);
      const ym = getDisplayedYM();
      if (ym && ym.year === targetYear && ym.month === targetMonth) return true;
    }

    // Fallback: click prev/next until it matches
    let current = getDisplayedYM();
    if (!current || isNaN(current.year) || isNaN(current.month)) return false;

    const want = ymToIndex(targetYear, targetMonth);
    for (let i = 0; i < 36; i++) {
      const have = ymToIndex(current.year, current.month);
      if (have === want) return true;

      const dir = have < want ? "next" : "prev";
      if (!clickPrevNext(dir)) break;

      await sleep(RENDER_WAIT_MS);
      current = getDisplayedYM();
      if (!current || isNaN(current.year) || isNaN(current.month)) break;
    }

    current = getDisplayedYM();
    return !!(
      current &&
      current.year === targetYear &&
      current.month === targetMonth
    );
  }

  // ===============================
  // Slot scan (ONLY group-first)
  // ===============================
  function scanGreendaysGroupFirst() {
    const g = getGroupFirst();
    if (!g) return [];
    const links = g.querySelectorAll("td.greenday > a");
    const days = [];
    links.forEach((a) => {
      const d = parseInt(a.textContent.trim(), 10);
      if (!isNaN(d)) days.push(d);
    });
    return Array.from(new Set(days)).sort((a, b) => a - b);
  }

  // ===============================
  // Location
  // ===============================
  function selectLocation() {
    const postSelect = document.querySelector("#post_select");
    if (!postSelect) return false;
    if (postSelect.value !== LOCATION_VALUE) {
      postSelect.value = LOCATION_VALUE;
      postSelect.dispatchEvent(new Event("input", { bubbles: true }));
      postSelect.dispatchEvent(new Event("change", { bubbles: true }));
      postSelect.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      console.log("✅ Location set");
    }
    return true;
  }

  // ===============================
  // Sweep Jan..Target
  // ===============================
  async function sweepMonths() {
    if (sweepRunning) return;
    sweepRunning = true;

    try {
      const found = []; // {year, month, days}

      for (let m = SWEEP_START_MONTH; m <= SWEEP_END_MONTH; m++) {
        if (alarmLoopId) break; // if alarm already running, stop wasting time

        const ok = await setMonthYearReliable(TARGET_YEAR, m);
        console.log(
          `[sweep] Set to ${TARGET_YEAR}-${String(m).padStart(2, "0")} (${MONTH_NAMES[m]}). UI title: ${getDisplayedTitleText()}`,
        );

        if (!ok) {
          console.log(
            `[sweep] ❌ Failed to move UI to monthIndex=${m}. Stopping sweep.`,
          );
          break;
        }

        const days = scanGreendaysGroupFirst();
        if (days.length) {
          console.log(
            `✅ SLOT(S) FOUND ${TARGET_YEAR}-${MONTH_NAMES[m]} (monthIndex=${m}):`,
            days.join(", "),
          );
          found.push({ year: TARGET_YEAR, month: m, days });
        } else {
          console.log(
            `No slots in ${TARGET_YEAR}-${MONTH_NAMES[m]} at ${new Date().toLocaleTimeString()}`,
          );
        }
      }

      // Restore UI to target month at end
      await setMonthYearReliable(TARGET_YEAR, TARGET_MONTH);
      console.log(
        `[sweep] Restored to target (${TARGET_YEAR}-${MONTH_NAMES[TARGET_MONTH]}). UI title: ${getDisplayedTitleText()}`,
      );

      if (found.length) {
        slotsFoundEver = true; // IMPORTANT: stop hard reload even if audio is blocked
        startAlarm();

        const sig = found
          .map((r) => `${r.year}-${r.month}:${r.days.join(".")}`)
          .join("|");
        if (sig !== lastResultsSignature) {
          lastResultsSignature = sig;

          const now = new Date();
          const content =
            `SLOTS FOUND (MONTH SWEEP)\n` +
            `Timestamp: ${now.toString()}\n` +
            `Location value: ${LOCATION_VALUE}\n` +
            `Sweep: year=${TARGET_YEAR}, months=${SWEEP_START_MONTH}..${SWEEP_END_MONTH}\n\n` +
            found
              .map(
                (r) =>
                  `Year=${r.year} MonthIndex=${r.month} (${MONTH_NAMES[r.month]}) Days=${r.days.join(", ")}`,
              )
              .join("\n") +
            `\n\nURL: ${location.href}\n`;

          console.log("📄 Downloaded slots log.");
        } else {
          console.log(
            "ℹ️ Slots found but same signature as last time (no new download).",
          );
        }
      }
    } finally {
      sweepRunning = false;
    }
  }

  // ===============================
  // Hard full reload every 2 minutes (until slots found)
  // ===============================
  function startHardReloadTimer() {
    if (hardReloadTimer) return;

    hardReloadTimer = setInterval(() => {
      if (slotsFoundEver) return; // stop reloading if slots found

      console.log("🔄 Hard reload triggered");
      location.reload();
    }, HARD_RELOAD_MS);

    console.log("⏱️ Hard reload every 2 minutes enabled.");
  }

  // ===============================
  // Boot
  // ===============================
  async function boot() {
    startHardReloadTimer(); // ALWAYS enable hard reload (even if calendar isn't ready yet)
    const start = Date.now();
    while (Date.now() - start < MAX_RETRY_MS) {
      const okLoc = selectLocation();
      const dp = getDatepickerRoot();
      const selects = getSelects();

      if (okLoc && dp && selects) {
        // ensure we're on target first
        await setMonthYearReliable(TARGET_YEAR, TARGET_MONTH);
        console.log(`[boot] UI title: ${getDisplayedTitleText()}`);

        await sweepMonths();

        if (!sweepTimer) {
          sweepTimer = setInterval(() => {
            if (!alarmLoopId) sweepMonths();
          }, SWEEP_INTERVAL_MS);
          console.log(
            `⏱️ Sweep timer enabled: every ${Math.round(SWEEP_INTERVAL_MS / 1000)}s.`,
          );
        }

        return;
      }
      await sleep(RETRY_EVERY_MS);
    }

    console.log(
      "⏳ Boot timeout: datepicker/selects not ready. (Hard reload will keep trying.)",
    );
  }

  setTimeout(boot, START_DELAY_MS);
  window.addEventListener("load", boot);
})();
