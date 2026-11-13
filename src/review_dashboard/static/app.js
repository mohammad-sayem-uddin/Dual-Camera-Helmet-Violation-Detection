(() => {
  "use strict";

  const state = {
    actor: null,
    enums: null,
    view: "overview",
    queue: [],
    pairs: [],
    resumeId: null,
    currentItem: null,
    pairTiming: null,
    serverRecord: null,
    prevId: null,
    nextId: null,
    draft: null,
    dirty: false,
    locked: false,
    autosaveTimer: null,
    frameInfoTimer: null,

    // Playback & sync engine
    currentRefTime: 0.0,
    playing: false,
    lastTickMs: 0,
    rafId: null,
    proxyMode: false,

    // Zoom & pan state per camera
    zoom: {
      front: { scale: 1.0, tx: 0, ty: 0 },
      back: { scale: 1.0, tx: 0, ty: 0 },
    },

    // Adjudication state
    adjOverview: null,
    adjDetail: null,
  };

  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));

  function esc(val) {
    return String(val ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  function showToast(msg, kind = "") {
    const el = $("#toast");
    el.textContent = msg;
    el.className = `toast ${kind}`;
    clearTimeout(el._hideTimer);
    el._hideTimer = setTimeout(() => {
      el.className = "toast hidden";
    }, 3400);
  }

  function setSaveIndicator(text, color = "") {
    const el = $("#save-indicator");
    el.textContent = text;
    el.style.color = color || "#cbd5e1";
  }

  async function api(url, opts = {}) {
    const headers = { ...(opts.headers || {}) };
    if (state.actor) {
      headers["X-Reviewer-Id"] = state.actor;
    }
    if (opts.body && typeof opts.body === "object") {
      headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(opts.body);
    }
    const resp = await fetch(url, { ...opts, headers, credentials: "same-origin" });
    const ct = resp.headers.get("Content-Type") || "";
    const data = ct.includes("application/json") ? await resp.json() : await resp.text();
    if (!resp.ok) {
      const err = new Error((data && data.error) || `HTTP ${resp.status}`);
      err.status = resp.status;
      err.details = data && data.details;
      throw err;
    }
    return data;
  }

  function localBackupKey(actor, wqId) {
    return `gt_v1_1_draft:${actor}:${wqId}`;
  }

  function defaultDraftFromItem(item) {
    const isDual = item.camera_scope === "DUAL_VIEW_WINDOW";
    const isFront = item.camera_scope === "FRONT_WINDOW";
    const isBack = item.camera_scope === "BACK_WINDOW";
    const hint = item.back_hint_interval || item.front_hint_interval || [
      item.overlap_start_ref_sec ?? 0,
      item.overlap_end_ref_sec ?? 1,
    ];
    return {
      workqueue_id: item.workqueue_id,
      pair_id: item.pair_id,
      item_link_status: isDual ? "LINKED_BOTH_VIEWS" : isFront ? "FRONT_VIEW_ONLY" : isBack ? "BACK_VIEW_ONLY" : "AMBIGUOUS_HUMAN_UNCERTAIN",
      is_non_motorcycle_transit: false,
      uncertainty: { is_uncertain: false, reasons: [] },
      reviewer_notes: "",
      motorcycles: [
        {
          moto_id: "M1",
          cross_camera_link_status: isDual ? "LINKED_BOTH_VIEWS" : isFront ? "FRONT_VIEW_ONLY" : "BACK_VIEW_ONLY",
          visible_front: isDual || isFront,
          visible_back: isDual || isBack,
          start_t_ref_sec: hint ? Number(hint[0]) : null,
          end_t_ref_sec: hint ? Number(hint[1]) : null,
          rear_plate: {
            visibility_tier: isFront ? "PLATE_NOT_VISIBLE_FRONT_ONLY" : "PLATE_UNREADABLE_MOTION_BLUR",
            plate_text_normalized: "",
            notes: "",
          },
          notes: "",
        },
      ],
      riders: [
        {
          rider_id: "R1",
          moto_id: "M1",
          role: "DRIVER",
          helmet_state: "HELMET",
          front_helmet_evidence: isFront || isDual ? "HELMET" : "NOT_VISIBLE",
          back_helmet_evidence: isBack || isDual ? "HELMET" : "NOT_VISIBLE",
          is_uncertain: false,
          start_t_ref_sec: null,
          end_t_ref_sec: null,
          notes: "",
        },
      ],
      violation_events: [
        {
          event_id: "E1",
          moto_id: "M1",
          rider_ids: ["R1"],
          event_type: "NO_VIOLATION",
          start_t_ref_sec: hint ? Number(hint[0]) : Number((item.min_t_ref_sec || 0).toFixed(3)),
          end_t_ref_sec: hint ? Number(hint[1]) : Number((item.min_t_ref_sec || 0).toFixed(3)),
          camera_evidence: isDual ? "BOTH_CAMERAS" : isFront ? "FRONT_ONLY" : "BACK_ONLY",
          keyframe_ref_t_sec: null,
          is_uncertain: false,
          notes: "",
        },
      ],
      snapshots: [],
      annotation_elapsed_sec: 0,
    };
  }

  // ------------------------------------------------------------------
  // Navigation & Session Initialization
  // ------------------------------------------------------------------
  function updateViewVisibility() {
    $$(".view-panel").forEach((sec) => sec.classList.add("hidden"));
    $$(".nav-btn").forEach((btn) => btn.classList.remove("active"));

    if (!state.actor) {
      $("#view-no-actor").classList.remove("hidden");
      $("#sidebar-progress-card").classList.add("hidden");
      return;
    }

    const activeBtn = $(`.nav-btn[data-view="${state.view}"]`);
    if (activeBtn) activeBtn.classList.add("active");
    const targetPanel = $(`#view-${state.view}`);
    if (targetPanel) targetPanel.classList.remove("hidden");

    const isAdj = state.actor === "adjudicator";
    $("#sidebar-progress-card").classList.toggle("hidden", isAdj);
    $("#adj-lock-tag").textContent = isAdj ? "Coordinator" : "Restricted";
    $("#adj-lock-tag").style.background = isAdj ? "#dcfce7" : "#fef3c7";
    $("#adj-lock-tag").style.color = isAdj ? "#15803d" : "#b45309";
  }

  async function switchView(nextView) {
    if (state.view === "review" && nextView !== "review") {
      stopPlayback();
      unloadVideos();
      api("/api/video/release", { method: "POST" }).catch(() => {});
    }
    state.view = nextView;
    updateViewVisibility();
    if (!state.actor) return;

    if (nextView === "overview") await loadReviewerData();
    else if (nextView === "queue") await loadReviewerData();
    else if (nextView === "history") await loadHistoryView();
    else if (nextView === "adjudication") await loadAdjudicationView();
  }

  async function setSessionActor(newActor) {
    if (state.dirty && state.currentItem && !state.locked) {
      await saveDraftNow({ silent: true });
    }
    stopPlayback();
    unloadVideos();
    state.currentItem = null;
    state.draft = null;
    state.dirty = false;
    state.adjDetail = null;

    const res = await api("/api/session", { method: "POST", body: { actor: newActor || "" } });
    state.actor = res.actor;
    $("#actor-select").value = state.actor || "";

    if (!state.actor) {
      state.view = "overview";
      updateViewVisibility();
      return;
    }

    if (state.actor === "adjudicator") {
      state.view = "adjudication";
      updateViewVisibility();
      await loadAdjudicationView();
    } else {
      if (state.view === "adjudication" || !state.view) {
        state.view = "overview";
      }
      updateViewVisibility();
      await loadReviewerData();
    }
    showToast(`Switched profile to ${res.actor_label || "none"}`, "ok");
  }

  async function refreshHealth() {
    try {
      const h = await api("/api/health");
      $("#health-meta").textContent =
        `v1.1 Status: ${h.release_human_verification_status} · Items: ${h.workqueue_count} · Pairs: ${h.pairs_count} · Active caps: ${h.video_diagnostics.open_captures}`;
    } catch (_e) {
      $("#health-meta").textContent = "Server reachable";
    }
  }

  // ------------------------------------------------------------------
  // Reviewer Queue, Pairs & Overview
  // ------------------------------------------------------------------
  async function loadReviewerData() {
    if (state.actor !== "reviewer_a" && state.actor !== "reviewer_b") return;
    const [qRes, pRes] = await Promise.all([
      api("/api/reviewer/queue"),
      api("/api/reviewer/pairs"),
    ]);
    state.queue = qRes.items;
    state.resumeId = qRes.resume_workqueue_id;
    state.pairs = pRes.pairs;

    const c = qRes.counts;
    $("#stat-submitted").textContent = c.submitted;
    $("#stat-draft").textContent = c.draft;
    $("#stat-pending").textContent = c.not_started;
    $("#stat-total").textContent = c.total;
    $("#sidebar-progress-fill").style.width = `${c.total ? Math.round((c.submitted / c.total) * 100) : 0}%`;

    $("#ov-total").textContent = c.total;
    $("#ov-sub").textContent = c.submitted;
    $("#ov-draft").textContent = c.draft;
    $("#ov-pend").textContent = c.not_started;
    $("#overview-subtitle").textContent = `Logged in as ${state.actor.toUpperCase()} — Blind Independent Queue`;

    renderPairsGrid();
    populatePairFilter();
    renderQueueTable();
  }

  function renderPairsGrid() {
    const host = $("#pairs-summary-list");
    host.innerHTML = state.pairs
      .map(
        (p) => `
      <div class="pair-chip" data-filter-pair="${esc(p.pair_id)}">
        <div><strong class="mono">${esc(p.pair_id)}</strong></div>
        <div class="small muted">${p.my_submitted}/${p.items_total} done · Δ=${p.global_offset_sec}s</div>
      </div>`
      )
      .join("");
    host.querySelectorAll(".pair-chip").forEach((el) => {
      el.addEventListener("click", () => {
        $("#q-pair-filter").value = el.dataset.filterPair;
        switchView("queue");
      });
    });
  }

  function populatePairFilter() {
    const sel = $("#q-pair-filter");
    const prev = sel.value;
    sel.innerHTML = `<option value="">All 25 Pairs</option>` +
      state.pairs.map((p) => `<option value="${esc(p.pair_id)}">${esc(p.pair_id)} (${p.my_submitted}/${p.items_total})</option>`).join("");
    sel.value = prev;
  }

  function renderQueueTable() {
    const q = $("#q-search").value.trim().toLowerCase();
    const pairF = $("#q-pair-filter").value;
    const statusF = $("#q-status-filter").value;
    const scopeF = $("#q-scope-filter").value;
    const bucketF = $("#q-bucket-filter").value;

    const rows = state.queue.filter((it) => {
      if (pairF && it.pair_id !== pairF) return false;
      if (statusF !== "ALL" && it.my_status !== statusF) return false;
      if (scopeF !== "ALL" && it.camera_scope !== scopeF) return false;
      if (bucketF !== "ALL" && it.sampling_bucket !== bucketF) return false;
      if (q && !it.workqueue_id.toLowerCase().includes(q) && !it.pair_id.toLowerCase().includes(q)) return false;
      return true;
    });

    $("#queue-table-body").innerHTML = rows
      .map((it) => {
        const fHint = it.front_transit_ref_sec_hint ? `F:${it.front_transit_ref_sec_hint}` : "";
        const bHint = it.back_transit_ref_sec_hint ? `B:${it.back_transit_ref_sec_hint}` : "";
        const hintTxt = [fHint, bHint].filter(Boolean).join(" | ") || "Full pair scan";
        return `
        <tr>
          <td class="mono"><strong>${esc(it.workqueue_id)}</strong></td>
          <td class="mono">${esc(it.pair_id)}</td>
          <td>${it.transit_index_in_pair}</td>
          <td><span class="scope-badge">${esc(it.camera_scope)}</span></td>
          <td class="mono">${it.global_offset_sec.toFixed(3)}s</td>
          <td class="mono small">${esc(hintTxt)}</td>
          <td><span class="status-pill ${esc(it.my_status)}">${esc(it.my_status)}</span></td>
          <td class="small muted">${it.my_updated_at ? esc(it.my_updated_at.slice(0, 19).replace("T", " ")) : "--"}</td>
          <td><button class="btn btn-xs btn-primary" data-open-wq="${esc(it.workqueue_id)}">${it.my_status === "SUBMITTED" ? "Inspect" : "Annotate"}</button></td>
        </tr>`;
      })
      .join("");

    $("#queue-table-body").querySelectorAll("[data-open-wq]").forEach((btn) => {
      btn.addEventListener("click", () => openWorkqueueItem(btn.dataset.openWq));
    });
  }

  // ------------------------------------------------------------------
  // Video Review Workbench + Dual-Camera Sync Engine
  // ------------------------------------------------------------------
  function unloadVideos() {
    ["front", "back"].forEach((cam) => {
      const vid = $(`#video-${cam}`);
      if (vid) {
        vid.pause();
        vid.removeAttribute("src");
        vid.load();
      }
    });
  }

  function applyZoomTransform(cam) {
    const vid = $(`#video-${cam}`);
    const z = state.zoom[cam];
    vid.style.transform = `translate(${z.tx}px, ${z.ty}px) scale(${z.scale})`;
    $(`#zoom-${cam}-label`).textContent = `${z.scale.toFixed(1)}x`;
  }

  function resetAllZoom() {
    state.zoom.front = { scale: 1.0, tx: 0, ty: 0 };
    state.zoom.back = { scale: 1.0, tx: 0, ty: 0 };
    applyZoomTransform("front");
    applyZoomTransform("back");
  }

  async function openWorkqueueItem(wqId) {
    if (!wqId) return;
    if (state.dirty && state.currentItem && !state.locked) {
      await saveDraftNow({ silent: true });
    }
    stopPlayback();

    const data = await api(`/api/reviewer/items/${encodeURIComponent(wqId)}`);
    const prevPair = state.currentItem ? state.currentItem.pair_id : null;

    state.currentItem = data.item;
    state.pairTiming = data.pair_timing;
    state.serverRecord = data.annotation;
    state.prevId = data.prev_workqueue_id;
    state.nextId = data.next_workqueue_id;
    state.locked = Boolean(data.annotation && data.annotation.status === "SUBMITTED");

    if (data.annotation && data.annotation.payload) {
      state.draft = JSON.parse(JSON.stringify(data.annotation.payload));
    } else {
      state.draft = defaultDraftFromItem(data.item);
    }
    state.dirty = false;

    state.view = "review";
    updateViewVisibility();
    $("#review-empty-notice").classList.add("hidden");
    $("#review-workbench").classList.remove("hidden");

    // Populate header
    $("#rv-wq-id").textContent = data.item.workqueue_id;
    $("#rv-pair-id").textContent = data.item.pair_id;
    $("#rv-scope").textContent = data.item.camera_scope;
    const sp = $("#rv-status-pill");
    sp.textContent = data.item.my_status;
    sp.className = `status-pill ${data.item.my_status}`;
    const hints = [
      data.item.front_transit_ref_sec_hint ? `F: ${data.item.front_transit_ref_sec_hint}` : "",
      data.item.back_transit_ref_sec_hint ? `B: ${data.item.back_transit_ref_sec_hint}` : "",
    ].filter(Boolean).join(" | ");
    $("#rv-hints-pill").textContent = hints ? `Hint: ${hints}` : "Hint: Full pair inspect";

    $("#btn-prev-item").disabled = !state.prevId;
    $("#btn-next-item").disabled = !state.nextId;

    setSaveIndicator(
      state.locked
        ? `SUBMITTED (Locked v${data.annotation.version})`
        : data.annotation
        ? `Saved Draft v${data.annotation.version}`
        : "New Item (Unsaved)",
      state.locked ? "#86efac" : "#cbd5e1"
    );

    // Check interrupted localStorage backup
    checkLocalRecoveryBanner(wqId, data.annotation);

    // Load video sources cleanly when pair switches
    if (prevPair !== data.item.pair_id || !$("#video-front").getAttribute("src")) {
      resetAllZoom();
      loadPairVideos(data.item.pair_id);
    }

    setupTimelineForItem(data.item, data.pair_timing);
    renderAnnotationForm();
    refreshHealth();
  }

  function loadPairVideos(pairId) {
    const modeParam = state.proxyMode ? "?mode=proxy_h264" : "";
    ["front", "back"].forEach((cam) => {
      const vid = $(`#video-${cam}`);
      const triggerFallback = (reason) => {
        if (state.proxyMode) return;
        showToast(`${reason}: switching to Smooth 720p H.264 Proxy...`, "");
        state.proxyMode = true;
        $("#chk-proxy-mode").checked = true;
        loadPairVideos(pairId);
        setTimeout(() => seekToRefTime(state.currentRefTime, { fetchFrameInfo: true }), 250);
      };
      vid.onerror = () => triggerFallback(`Codec error on ${cam}`);
      vid.onloadeddata = () => {
        if (vid.videoWidth === 0 || vid.videoHeight === 0) {
          triggerFallback(`No native video decoder for ${cam}`);
        }
      };
      vid.src = `/api/video/${encodeURIComponent(pairId)}/${cam}${modeParam}`;
      vid.playbackRate = Number($("#playback-rate").value || 0.5);
    });
  }

  function getHintInterval(item) {
    if (item.back_hint_interval) return item.back_hint_interval;
    if (item.front_hint_interval) return item.front_hint_interval;
    if (item.overlap_end_ref_sec > item.overlap_start_ref_sec) {
      return [item.overlap_start_ref_sec, item.overlap_end_ref_sec];
    }
    return [item.min_t_ref_sec, item.max_t_ref_sec];
  }

  function setupTimelineForItem(item, timing) {
    const minT = timing.min_t_ref_sec;
    const maxT = timing.max_t_ref_sec;
    const span = Math.max(0.001, maxT - minT);
    const scrubber = $("#timeline-scrubber");
    scrubber.min = String(minT);
    scrubber.max = String(maxT);
    scrubber.step = "0.005";

    const pct = (t) => `${Math.max(0, Math.min(100, ((t - minT) / span) * 100)).toFixed(2)}%`;
    const widthPct = (a, b) => `${Math.max(0.4, Math.min(100, ((b - a) / span) * 100)).toFixed(2)}%`;

    const frontStartRef = timing.global_offset_sec;
    const frontEndRef = timing.global_offset_sec + timing.front_duration_sec;
    $("#vis-front-band").style.left = pct(frontStartRef);
    $("#vis-front-band").style.width = widthPct(frontStartRef, frontEndRef);

    $("#vis-back-band").style.left = pct(0);
    $("#vis-back-band").style.width = widthPct(0, timing.back_duration_sec);

    $("#vis-overlap-band").style.left = pct(timing.overlap_start_ref_sec);
    $("#vis-overlap-band").style.width = widthPct(timing.overlap_start_ref_sec, timing.overlap_end_ref_sec);

    const hint = getHintInterval(item);
    const hintEl = $("#vis-hint-band");
    if (item.front_hint_interval || item.back_hint_interval) {
      hintEl.classList.remove("hidden");
      hintEl.style.left = pct(hint[0]);
      hintEl.style.width = widthPct(hint[0], hint[1]);
    } else {
      hintEl.classList.add("hidden");
    }

    $("#tel-offset").textContent = `${timing.global_offset_sec.toFixed(3)}s`;
    $("#tel-eps").textContent = `${timing.residual_error_ms.toFixed(1)}`;
    $("#tel-sync-quality").textContent = `Quality: ${timing.sync_quality}`;

    const initialT = Math.max(minT, Math.min(maxT, hint[0]));
    seekToRefTime(initialT, { fetchFrameInfo: true });
    renderTimelineEventMarkers();
  }

  function renderTimelineEventMarkers() {
    if (!state.pairTiming || !state.draft) return;
    const host = $("#vis-event-markers");
    const minT = state.pairTiming.min_t_ref_sec;
    const span = Math.max(0.001, state.pairTiming.max_t_ref_sec - minT);
    host.innerHTML = (state.draft.violation_events || [])
      .map((ev, idx) => {
        if (ev.start_t_ref_sec == null || ev.end_t_ref_sec == null) return "";
        const left = Math.max(0, Math.min(100, ((ev.start_t_ref_sec - minT) / span) * 100));
        const width = Math.max(0.6, ((ev.end_t_ref_sec - ev.start_t_ref_sec) / span) * 100);
        return `<div class="track-ev-mark" data-jump-ev="${idx}" style="left:${left.toFixed(2)}%;width:${width.toFixed(2)}%" title="${esc(ev.event_id)}: ${esc(ev.event_type)} (${ev.start_t_ref_sec.toFixed(3)}..${ev.end_t_ref_sec.toFixed(3)})"></div>`;
      })
      .join("");
    host.querySelectorAll("[data-jump-ev]").forEach((el) => {
      el.addEventListener("click", () => {
        const ev = state.draft.violation_events[Number(el.dataset.jumpEv)];
        if (ev && ev.start_t_ref_sec != null) {
          stopPlayback();
          seekToRefTime(ev.start_t_ref_sec, { fetchFrameInfo: true });
        }
      });
    });
  }

  function syncVideosToRefTime(tRef, isPlayingTick = false) {
    if (!state.pairTiming) return;
    const t = Math.max(state.pairTiming.min_t_ref_sec, Math.min(state.pairTiming.max_t_ref_sec, Number(tRef)));
    state.currentRefTime = t;
    const tFront = t - state.pairTiming.global_offset_sec;

    const backIn = t >= -0.02 && t <= state.pairTiming.back_duration_sec + 0.02;
    const frontIn = tFront >= -0.02 && tFront <= state.pairTiming.front_duration_sec + 0.02;
    const inOverlap = t >= state.pairTiming.overlap_start_ref_sec - 1e-4 && t <= state.pairTiming.overlap_end_ref_sec + 1e-4;

    $("#tel-tref").textContent = `${t.toFixed(3)}s`;
    $("#tel-tfront").textContent = `${tFront.toFixed(3)}s`;
    $("#timeline-scrubber").value = String(t);

    const obBadge = $("#tel-overlap-badge");
    obBadge.textContent = inOverlap ? "COMMON OVERLAP" : "SINGLE-CAMERA REGION";
    obBadge.className = `overlap-pill ${inOverlap ? "in-overlap" : ""}`;

    $("#oob-back").classList.toggle("hidden", backIn);
    if (!backIn) {
      $("#oob-back").textContent = t < 0
        ? `BACK STARTS AT t_ref = 0.000s (in ${(-t).toFixed(2)}s)`
        : `BACK ENDED AT ${state.pairTiming.back_duration_sec.toFixed(2)}s`;
    }

    $("#oob-front").classList.toggle("hidden", frontIn);
    if (!frontIn) {
      const fStartRef = state.pairTiming.global_offset_sec;
      const fEndRef = fStartRef + state.pairTiming.front_duration_sec;
      $("#oob-front").textContent = tFront < 0
        ? `FRONT STARTS AT t_ref = ${fStartRef.toFixed(3)}s`
        : `FRONT ENDED AT t_ref = ${fEndRef.toFixed(3)}s`;
    }

    const vidB = $("#video-back");
    const vidF = $("#video-front");
    const rate = Number($("#playback-rate").value || 0.5);

    const applyTarget = (vid, targetSec, inRange) => {
      if (!inRange) {
        if (!vid.paused) vid.pause();
        return;
      }
      const clamped = Math.max(0, Math.min(vid.duration || targetSec + 10, targetSec));
      if (!isPlayingTick) {
        if (!vid.paused) vid.pause();
        if (Math.abs((vid.currentTime || 0) - clamped) > 0.003) {
          try { vid.currentTime = clamped; } catch (_e) {}
        }
      } else {
        if (vid.playbackRate !== rate) vid.playbackRate = rate;
        if (vid.paused) {
          try { vid.currentTime = clamped; } catch (_e) {}
          vid.play().catch(() => {});
        } else if (Math.abs((vid.currentTime || 0) - clamped) > 0.06) {
          try { vid.currentTime = clamped; } catch (_e) {}
        }
      }
    };

    applyTarget(vidB, t, backIn);
    applyTarget(vidF, tFront, frontIn);
  }

  function seekToRefTime(tRef, { fetchFrameInfo = false } = {}) {
    syncVideosToRefTime(tRef, false);
    if (fetchFrameInfo) {
      scheduleFrameInfoLookup();
    }
  }

  function scheduleFrameInfoLookup() {
    clearTimeout(state.frameInfoTimer);
    state.frameInfoTimer = setTimeout(async () => {
      if (!state.pairTiming) return;
      try {
        const info = await api(
          `/api/frame_info/${encodeURIComponent(state.pairTiming.pair_id)}?t_ref=${state.currentRefTime.toFixed(6)}`
        );
        $("#meta-front-pts").textContent = info.front.in_range
          ? `PTS: ${info.front.actual_pts_norm_sec.toFixed(3)}s | F: ${info.front.frame_index}/${info.front.frame_count} (Δ=${info.front.delta_pts_ms.toFixed(1)}ms)`
          : "PTS: OUT OF RANGE";
        $("#meta-back-pts").textContent = info.back.in_range
          ? `PTS: ${info.back.actual_pts_norm_sec.toFixed(3)}s | F: ${info.back.frame_index}/${info.back.frame_count} (Δ=${info.back.delta_pts_ms.toFixed(1)}ms)`
          : "PTS: OUT OF RANGE";
      } catch (_e) {}
    }, 90);
  }

  function togglePlayback() {
    if (state.playing) stopPlayback();
    else startPlayback();
  }

  function startPlayback() {
    if (!state.pairTiming || state.playing) return;
    state.playing = true;
    state.lastTickMs = performance.now();
    $("#btn-play-pause").textContent = "❚❚ Pause";

    const tick = (nowMs) => {
      if (!state.playing || !state.pairTiming) return;
      const dt = Math.min(0.1, Math.max(0, (nowMs - state.lastTickMs) / 1000.0));
      state.lastTickMs = nowMs;
      const rate = Number($("#playback-rate").value || 0.5);
      let nextT = state.currentRefTime + dt * rate;

      const loopOn = $("#chk-loop-hint").checked && state.currentItem;
      if (loopOn) {
        const [hs, he] = getHintInterval(state.currentItem);
        const loopStart = Math.max(state.pairTiming.min_t_ref_sec, hs - 0.5);
        const loopEnd = Math.min(state.pairTiming.max_t_ref_sec, he + 0.5);
        if (nextT > loopEnd) nextT = loopStart;
      } else if (nextT >= state.pairTiming.max_t_ref_sec) {
        syncVideosToRefTime(state.pairTiming.max_t_ref_sec, false);
        stopPlayback();
        scheduleFrameInfoLookup();
        return;
      }

      syncVideosToRefTime(nextT, true);
      state.rafId = requestAnimationFrame(tick);
    };
    state.rafId = requestAnimationFrame(tick);
  }

  function stopPlayback() {
    state.playing = false;
    if (state.rafId) {
      cancelAnimationFrame(state.rafId);
      state.rafId = null;
    }
    $("#btn-play-pause").textContent = "▶ Play";
    ["front", "back"].forEach((cam) => {
      const v = $(`#video-${cam}`);
      if (v && !v.paused) v.pause();
    });
  }

  async function stepFrames(deltaFrames) {
    if (!state.pairTiming) return;
    stopPlayback();
    const backIn = state.currentRefTime >= -0.02 && state.currentRefTime <= state.pairTiming.back_duration_sec + 0.02;
    const anchor = backIn ? "back" : "front";
    try {
      const info = await api(
        `/api/frame_info/${encodeURIComponent(state.pairTiming.pair_id)}?t_ref=${state.currentRefTime.toFixed(6)}&step_frames=${deltaFrames}&anchor=${anchor}`
      );
      seekToRefTime(info.t_ref_sec, { fetchFrameInfo: false });
      $("#meta-front-pts").textContent = info.front.in_range
        ? `PTS: ${info.front.actual_pts_norm_sec.toFixed(3)}s | F: ${info.front.frame_index}/${info.front.frame_count} (Δ=${info.front.delta_pts_ms.toFixed(1)}ms)`
        : "PTS: OUT OF RANGE";
      $("#meta-back-pts").textContent = info.back.in_range
        ? `PTS: ${info.back.actual_pts_norm_sec.toFixed(3)}s | F: ${info.back.frame_index}/${info.back.frame_count} (Δ=${info.back.delta_pts_ms.toFixed(1)}ms)`
        : "PTS: OUT OF RANGE";
    } catch (_e) {
      seekToRefTime(state.currentRefTime + deltaFrames * (1 / 60), { fetchFrameInfo: true });
    }
  }

  // ------------------------------------------------------------------
  // Annotation Form Rendering & State Binding
  // ------------------------------------------------------------------
  function markDirty() {
    if (state.locked || !state.draft || !state.currentItem) return;
    state.dirty = true;
    setSaveIndicator("Unsaved changes...", "#fcd34d");
    try {
      localStorage.setItem(
        localBackupKey(state.actor, state.currentItem.workqueue_id),
        JSON.stringify({ savedAt: Date.now(), draft: state.draft })
      );
    } catch (_e) {}

    clearTimeout(state.autosaveTimer);
    state.autosaveTimer = setTimeout(() => {
      saveDraftNow({ silent: true });
    }, 1200);
    renderTimelineEventMarkers();
  }

  function checkLocalRecoveryBanner(wqId, serverRecord) {
    const banner = $("#recovery-banner");
    banner.classList.add("hidden");
    if (state.locked) return;
    const raw = localStorage.getItem(localBackupKey(state.actor, wqId));
    if (!raw) return;
    try {
      const parsed = JSON.parse(raw);
      const serverJson = serverRecord ? JSON.stringify(serverRecord.payload) : "";
      const localJson = JSON.stringify(parsed.draft);
      if (localJson !== serverJson) {
        banner.classList.remove("hidden");
      }
    } catch (_e) {}
  }

  function renderAnnotationForm() {
    const d = state.draft;
    if (!d) return;
    const disabled = state.locked ? "disabled" : "";

    $("#form-locked-banner").classList.toggle("hidden", !state.locked);
    $("#fld-non-moto").checked = Boolean(d.is_non_motorcycle_transit);
    $("#fld-non-moto").disabled = state.locked;
    $("#fld-item-link").value = d.item_link_status || "AMBIGUOUS_HUMAN_UNCERTAIN";
    $("#fld-item-link").disabled = state.locked;
    $("#fld-is-uncertain").checked = Boolean(d.uncertainty && d.uncertainty.is_uncertain);
    $("#fld-is-uncertain").disabled = state.locked;
    $("#fld-reviewer-notes").value = d.reviewer_notes || "";
    $("#fld-reviewer-notes").disabled = state.locked;

    // Uncertainty chips
    const tags = (state.enums && state.enums.uncertainty_reason_tags) || [];
    const selectedTags = new Set((d.uncertainty && d.uncertainty.reasons) || []);
    $("#uncertainty-chips").innerHTML = tags
      .map(
        (tag) =>
          `<button type="button" class="chip-toggle ${selectedTags.has(tag) ? "selected" : ""}" data-unc-tag="${esc(tag)}" ${disabled}>${esc(tag)}</button>`
      )
      .join("");
    $("#uncertainty-chips").querySelectorAll("[data-unc-tag]").forEach((btn) => {
      btn.addEventListener("click", () => {
        if (state.locked) return;
        const tag = btn.dataset.uncTag;
        const arr = new Set(d.uncertainty.reasons || []);
        if (arr.has(tag)) arr.delete(tag);
        else arr.add(tag);
        d.uncertainty.reasons = Array.from(arr);
        if (d.uncertainty.reasons.length > 0) d.uncertainty.is_uncertain = true;
        markDirty();
        renderAnnotationForm();
      });
    });

    $("#btn-add-moto").disabled = state.locked || d.is_non_motorcycle_transit;
    $("#btn-add-rider").disabled = state.locked || d.is_non_motorcycle_transit || d.motorcycles.length === 0;
    $("#btn-add-event").disabled = state.locked || d.is_non_motorcycle_transit || d.motorcycles.length === 0;
    $("#btn-save-draft").disabled = state.locked;
    $("#btn-reset-draft").disabled = state.locked;
    $("#btn-submit-item").disabled = state.locked;

    renderMotorcyclesList();
    renderRidersList();
    renderEventsList();
    renderSnapshotsList();
  }

  function renderMotorcyclesList() {
    const d = state.draft;
    const host = $("#motos-container");
    if (d.motorcycles.length === 0) {
      host.innerHTML = `<div class="muted small">${d.is_non_motorcycle_transit ? "Marked Not a Motorcycle Transit (0 motorcycles expected)." : "No motorcycles added yet."}</div>`;
      return;
    }
    const dis = state.locked ? "disabled" : "";
    host.innerHTML = d.motorcycles
      .map(
        (m, idx) => `
      <div class="entity-card">
        <div class="entity-card-hdr">
          <strong class="mono">${esc(m.moto_id)}</strong>
          <button type="button" class="btn btn-xs btn-danger" data-del-moto="${idx}" ${dis}>Remove</button>
        </div>
        <div class="field-row">
          <label>Link Status
            <select data-m-field="${idx}:cross_camera_link_status" ${dis}>
              ${["LINKED_BOTH_VIEWS", "FRONT_VIEW_ONLY", "BACK_VIEW_ONLY", "AMBIGUOUS_HUMAN_UNCERTAIN"]
                .map((v) => `<option value="${v}" ${m.cross_camera_link_status === v ? "selected" : ""}>${v}</option>`)
                .join("")}
            </select>
          </label>
          <label class="checkbox-box"><input type="checkbox" data-m-chk="${idx}:visible_front" ${m.visible_front ? "checked" : ""} ${dis}/> Front View</label>
          <label class="checkbox-box"><input type="checkbox" data-m-chk="${idx}:visible_back" ${m.visible_back ? "checked" : ""} ${dis}/> Back View</label>
        </div>
        <div class="field-row">
          <label>Start t_ref (s)
            <div class="inline-input-btn">
              <input type="number" step="0.001" data-m-num="${idx}:start_t_ref_sec" value="${m.start_t_ref_sec ?? ""}" ${dis}/>
              <button type="button" class="btn btn-xs btn-secondary" data-m-cap="${idx}:start_t_ref_sec" ${dis}>Now</button>
            </div>
          </label>
          <label>End t_ref (s)
            <div class="inline-input-btn">
              <input type="number" step="0.001" data-m-num="${idx}:end_t_ref_sec" value="${m.end_t_ref_sec ?? ""}" ${dis}/>
              <button type="button" class="btn btn-xs btn-secondary" data-m-cap="${idx}:end_t_ref_sec" ${dis}>Now</button>
            </div>
          </label>
        </div>
        <div class="field-row">
          <label>Rear Plate Tier
            <select data-m-plate="${idx}:visibility_tier" ${dis}>
              ${["PLATE_CLEAR", "PLATE_PARTIAL", "PLATE_UNREADABLE_MOTION_BLUR", "PLATE_NOT_VISIBLE_FRONT_ONLY"]
                .map((v) => `<option value="${v}" ${(m.rear_plate?.visibility_tier) === v ? "selected" : ""}>${v}</option>`)
                .join("")}
            </select>
          </label>
          <label>Normalized Plate Text
            <input type="text" data-m-plate="${idx}:plate_text_normalized" value="${esc(m.rear_plate?.plate_text_normalized || "")}" placeholder="e.g. DHA-1234" ${dis}/>
          </label>
        </div>
      </div>`
      )
      .join("");

    host.querySelectorAll("[data-del-moto]").forEach((b) =>
      b.addEventListener("click", () => {
        const i = Number(b.dataset.delMoto);
        const removedId = d.motorcycles[i].moto_id;
        d.motorcycles.splice(i, 1);
        d.riders = d.riders.filter((r) => r.moto_id !== removedId);
        d.violation_events = d.violation_events.filter((e) => e.moto_id !== removedId);
        markDirty();
        renderAnnotationForm();
      })
    );
    host.querySelectorAll("[data-m-field]").forEach((s) =>
      s.addEventListener("change", () => {
        const [i, k] = s.dataset.mField.split(":");
        d.motorcycles[Number(i)][k] = s.value;
        markDirty();
      })
    );
    host.querySelectorAll("[data-m-chk]").forEach((c) =>
      c.addEventListener("change", () => {
        const [i, k] = c.dataset.mChk.split(":");
        d.motorcycles[Number(i)][k] = c.checked;
        markDirty();
      })
    );
    host.querySelectorAll("[data-m-num]").forEach((inp) =>
      inp.addEventListener("input", () => {
        const [i, k] = inp.dataset.mNum.split(":");
        d.motorcycles[Number(i)][k] = inp.value === "" ? null : Number(inp.value);
        markDirty();
      })
    );
    host.querySelectorAll("[data-m-cap]").forEach((b) =>
      b.addEventListener("click", () => {
        const [i, k] = b.dataset.mCap.split(":");
        d.motorcycles[Number(i)][k] = Number(state.currentRefTime.toFixed(3));
        markDirty();
        renderMotorcyclesList();
      })
    );
    host.querySelectorAll("[data-m-plate]").forEach((inp) =>
      inp.addEventListener(inp.tagName === "SELECT" ? "change" : "input", () => {
        const [i, k] = inp.dataset.mPlate.split(":");
        d.motorcycles[Number(i)].rear_plate = d.motorcycles[Number(i)].rear_plate || {};
        d.motorcycles[Number(i)].rear_plate[k] = inp.value;
        markDirty();
      })
    );
  }

  function renderRidersList() {
    const d = state.draft;
    const host = $("#riders-container");
    if (d.riders.length === 0) {
      host.innerHTML = `<div class="muted small">No riders added yet.</div>`;
      return;
    }
    const dis = state.locked ? "disabled" : "";
    const motoOpts = d.motorcycles.map((m) => m.moto_id);
    host.innerHTML = d.riders
      .map(
        (r, idx) => `
      <div class="entity-card">
        <div class="entity-card-hdr">
          <strong class="mono">${esc(r.rider_id)} → ${esc(r.moto_id)}</strong>
          <button type="button" class="btn btn-xs btn-danger" data-del-rider="${idx}" ${dis}>Remove</button>
        </div>
        <div class="field-row">
          <label>Motorcycle
            <select data-r-field="${idx}:moto_id" ${dis}>
              ${motoOpts.map((mid) => `<option value="${esc(mid)}" ${r.moto_id === mid ? "selected" : ""}>${esc(mid)}</option>`).join("")}
            </select>
          </label>
          <label>Role
            <select data-r-field="${idx}:role" ${dis}>
              ${["DRIVER", "PILLION", "UNCERTAIN"].map((v) => `<option value="${v}" ${r.role === v ? "selected" : ""}>${v}</option>`).join("")}
            </select>
          </label>
          <label>Helmet State
            <select data-r-field="${idx}:helmet_state" ${dis}>
              ${["HELMET", "NO_HELMET", "UNCERTAIN", "NOT_VISIBLE"].map((v) => `<option value="${v}" ${r.helmet_state === v ? "selected" : ""}>${v}</option>`).join("")}
            </select>
          </label>
        </div>
        <div class="field-row">
          <label>Front View Evidence
            <select data-r-field="${idx}:front_helmet_evidence" ${dis}>
              ${["HELMET", "NO_HELMET", "UNCERTAIN", "NOT_VISIBLE"].map((v) => `<option value="${v}" ${r.front_helmet_evidence === v ? "selected" : ""}>${v}</option>`).join("")}
            </select>
          </label>
          <label>Back View Evidence
            <select data-r-field="${idx}:back_helmet_evidence" ${dis}>
              ${["HELMET", "NO_HELMET", "UNCERTAIN", "NOT_VISIBLE"].map((v) => `<option value="${v}" ${r.back_helmet_evidence === v ? "selected" : ""}>${v}</option>`).join("")}
            </select>
          </label>
        </div>
      </div>`
      )
      .join("");

    host.querySelectorAll("[data-del-rider]").forEach((b) =>
      b.addEventListener("click", () => {
        const i = Number(b.dataset.delRider);
        const rid = d.riders[i].rider_id;
        d.riders.splice(i, 1);
        d.violation_events.forEach((ev) => {
          ev.rider_ids = (ev.rider_ids || []).filter((id) => id !== rid);
        });
        markDirty();
        renderAnnotationForm();
      })
    );
    host.querySelectorAll("[data-r-field]").forEach((s) =>
      s.addEventListener("change", () => {
        const [i, k] = s.dataset.rField.split(":");
        d.riders[Number(i)][k] = s.value;
        markDirty();
        renderEventsList();
      })
    );
  }

  function renderEventsList() {
    const d = state.draft;
    const host = $("#events-container");
    if (d.violation_events.length === 0) {
      host.innerHTML = `<div class="muted small">No temporal events added yet.</div>`;
      return;
    }
    const dis = state.locked ? "disabled" : "";
    const motoOpts = d.motorcycles.map((m) => m.moto_id);

    host.innerHTML = d.violation_events
      .map((ev, idx) => {
        const validRiders = d.riders.filter((r) => r.moto_id === ev.moto_id);
        const selRids = new Set(ev.rider_ids || []);
        return `
        <div class="entity-card">
          <div class="entity-card-hdr">
            <strong class="mono">${esc(ev.event_id)} (${esc(ev.moto_id)})</strong>
            <div>
              <button type="button" class="btn btn-xs btn-secondary" data-jump-ev="${idx}">Jump</button>
              <button type="button" class="btn btn-xs btn-danger" data-del-ev="${idx}" ${dis}>Remove</button>
            </div>
          </div>
          <div class="field-row">
            <label>Motorcycle
              <select data-e-field="${idx}:moto_id" ${dis}>
                ${motoOpts.map((mid) => `<option value="${esc(mid)}" ${ev.moto_id === mid ? "selected" : ""}>${esc(mid)}</option>`).join("")}
              </select>
            </label>
            <label>Event Type
              <select data-e-field="${idx}:event_type" ${dis}>
                ${["DRIVER_NO_HELMET", "PILLION_NO_HELMET", "BOTH_NO_HELMET", "NO_VIOLATION", "INSUFFICIENT_VISUAL_EVIDENCE"]
                  .map((v) => `<option value="${v}" ${ev.event_type === v ? "selected" : ""}>${v}</option>`)
                  .join("")}
              </select>
            </label>
            <label>Camera Evidence
              <select data-e-field="${idx}:camera_evidence" ${dis}>
                ${["BOTH_CAMERAS", "FRONT_ONLY", "BACK_ONLY", "UNCERTAIN"]
                  .map((v) => `<option value="${v}" ${ev.camera_evidence === v ? "selected" : ""}>${v}</option>`)
                  .join("")}
              </select>
            </label>
          </div>
          <div class="field-row">
            <label>Start t_ref (s)
              <div class="inline-input-btn">
                <input type="number" step="0.001" data-e-num="${idx}:start_t_ref_sec" value="${ev.start_t_ref_sec ?? ""}" ${dis}/>
                <button type="button" class="btn btn-xs btn-secondary" data-e-cap="${idx}:start_t_ref_sec" ${dis}>Now</button>
              </div>
            </label>
            <label>End t_ref (s)
              <div class="inline-input-btn">
                <input type="number" step="0.001" data-e-num="${idx}:end_t_ref_sec" value="${ev.end_t_ref_sec ?? ""}" ${dis}/>
                <button type="button" class="btn btn-xs btn-secondary" data-e-cap="${idx}:end_t_ref_sec" ${dis}>Now</button>
              </div>
            </label>
            <div style="display:flex;align-items:flex-end;">
              <button type="button" class="btn btn-xs btn-secondary" data-e-point="${idx}" ${dis} title="Set both start and end to current playhead for a 1-frame point event">1-Frame Point [t, t]</button>
            </div>
          </div>
          <div class="field-block">
            <span class="field-label small">Linked Riders on ${esc(ev.moto_id)}:</span>
            <div class="chips-wrap">
              ${validRiders.length
                ? validRiders
                    .map(
                      (r) =>
                        `<button type="button" class="chip-toggle ${selRids.has(r.rider_id) ? "selected" : ""}" data-e-rider="${idx}:${esc(r.rider_id)}" ${dis}>${esc(r.rider_id)} (${esc(r.role)}: ${esc(r.helmet_state)})</button>`
                    )
                    .join("")
                : `<span class="small muted">No riders attached to ${esc(ev.moto_id)}</span>`}
            </div>
          </div>
        </div>`;
      })
      .join("");

    host.querySelectorAll("[data-jump-ev]").forEach((b) =>
      b.addEventListener("click", () => {
        const ev = d.violation_events[Number(b.dataset.jumpEv)];
        if (ev && ev.start_t_ref_sec != null) {
          stopPlayback();
          seekToRefTime(ev.start_t_ref_sec, { fetchFrameInfo: true });
        }
      })
    );
    host.querySelectorAll("[data-del-ev]").forEach((b) =>
      b.addEventListener("click", () => {
        d.violation_events.splice(Number(b.dataset.delEv), 1);
        markDirty();
        renderEventsList();
      })
    );
    host.querySelectorAll("[data-e-field]").forEach((s) =>
      s.addEventListener("change", () => {
        const [i, k] = s.dataset.eField.split(":");
        const ev = d.violation_events[Number(i)];
        ev[k] = s.value;
        if (k === "moto_id") {
          ev.rider_ids = d.riders.filter((r) => r.moto_id === s.value).map((r) => r.rider_id);
        }
        markDirty();
        renderEventsList();
      })
    );
    host.querySelectorAll("[data-e-num]").forEach((inp) =>
      inp.addEventListener("input", () => {
        const [i, k] = inp.dataset.eNum.split(":");
        d.violation_events[Number(i)][k] = inp.value === "" ? null : Number(inp.value);
        markDirty();
      })
    );
    host.querySelectorAll("[data-e-cap]").forEach((b) =>
      b.addEventListener("click", () => {
        const [i, k] = b.dataset.eCap.split(":");
        d.violation_events[Number(i)][k] = Number(state.currentRefTime.toFixed(3));
        markDirty();
        renderEventsList();
      })
    );
    host.querySelectorAll("[data-e-point]").forEach((b) =>
      b.addEventListener("click", () => {
        const t = Number(state.currentRefTime.toFixed(3));
        const ev = d.violation_events[Number(b.dataset.ePoint)];
        ev.start_t_ref_sec = t;
        ev.end_t_ref_sec = t;
        markDirty();
        renderEventsList();
      })
    );
    host.querySelectorAll("[data-e-rider]").forEach((b) =>
      b.addEventListener("click", () => {
        if (state.locked) return;
        const [i, rid] = b.dataset.eRider.split(":");
        const ev = d.violation_events[Number(i)];
        const s = new Set(ev.rider_ids || []);
        if (s.has(rid)) s.delete(rid);
        else s.add(rid);
        ev.rider_ids = Array.from(s);
        markDirty();
        renderEventsList();
      })
    );
  }

  function renderSnapshotsList() {
    const snaps = (state.draft && state.draft.snapshots) || [];
    const host = $("#snapshots-container");
    if (snaps.length === 0) {
      host.innerHTML = `<span class="muted small">No frame snapshots captured yet. Use "Snap" on a camera header.</span>`;
      return;
    }
    host.innerHTML = snaps
      .map(
        (s, idx) => `
      <div class="snap-thumb">
        <img src="${esc(s.url)}" alt="${esc(s.snapshot_id)}" />
        <div class="mono">${esc(s.camera)} F${s.frame_index}</div>
        <div class="mono">t=${Number(s.t_ref_sec).toFixed(3)}s</div>
        ${state.locked ? "" : `<button type="button" class="btn btn-xs btn-secondary" data-jump-snap="${idx}">Jump</button>`}
      </div>`
      )
      .join("");
    host.querySelectorAll("[data-jump-snap]").forEach((b) =>
      b.addEventListener("click", () => {
        const s = snaps[Number(b.dataset.jumpSnap)];
        if (s && s.t_ref_sec != null) seekToRefTime(s.t_ref_sec, { fetchFrameInfo: true });
      })
    );
  }

  function showValidationErrors(errors) {
    const box = $("#validation-box");
    if (!errors || errors.length === 0) {
      box.classList.add("hidden");
      box.innerHTML = "";
      return;
    }
    box.className = "validation-box";
    box.innerHTML = `<strong>Validation Issues:</strong><ul>${errors.map((e) => `<li>${esc(e)}</li>`).join("")}</ul>`;
  }

  async function saveDraftNow({ silent = false } = {}) {
    if (!state.currentItem || !state.draft || state.locked) return;
    clearTimeout(state.autosaveTimer);
    showValidationErrors([]);
    try {
      const res = await api(
        `/api/reviewer/items/${encodeURIComponent(state.currentItem.workqueue_id)}/draft`,
        { method: "PUT", body: state.draft }
      );
      state.serverRecord = res.annotation;
      state.currentItem = res.item;
      state.dirty = false;
      localStorage.removeItem(localBackupKey(state.actor, state.currentItem.workqueue_id));
      $("#recovery-banner").classList.add("hidden");
      $("#rv-status-pill").textContent = res.item.my_status;
      $("#rv-status-pill").className = `status-pill ${res.item.my_status}`;
      setSaveIndicator(`Saved Draft v${res.annotation.version}`, "#93c5fd");
      if (!silent) showToast(`Saved draft v${res.annotation.version}`, "ok");
    } catch (err) {
      setSaveIndicator("Save failed (kept locally)", "#fca5a5");
      if (err.details && err.details.errors) showValidationErrors(err.details.errors);
      if (!silent) showToast(err.message, "error");
    }
  }

  function openSubmitConfirmation() {
    if (!state.currentItem || !state.draft || state.locked) return;
    $("#confirm-wq-id").textContent = state.currentItem.workqueue_id;
    const summaryText = `Link: ${state.draft.item_link_status} | Motorcycles: ${state.draft.motorcycles.length} | Riders: ${state.draft.riders.length} | Events: ${state.draft.violation_events.length}`;
    $("#confirm-summary-preview").textContent = summaryText;
    $("#confirm-modal").classList.remove("hidden");
  }

  async function executeSubmitNow() {
    $("#confirm-modal").classList.add("hidden");
    if (!state.currentItem || !state.draft || state.locked) return;
    showValidationErrors([]);
    try {
      const res = await api(
        `/api/reviewer/items/${encodeURIComponent(state.currentItem.workqueue_id)}/submit`,
        { method: "POST", body: state.draft }
      );
      state.serverRecord = res.annotation;
      state.currentItem = res.item;
      state.draft = JSON.parse(JSON.stringify(res.annotation.payload));
      state.dirty = false;
      state.locked = true;
      localStorage.removeItem(localBackupKey(state.actor, state.currentItem.workqueue_id));
      $("#recovery-banner").classList.add("hidden");
      $("#rv-status-pill").textContent = "SUBMITTED";
      $("#rv-status-pill").className = "status-pill SUBMITTED";
      setSaveIndicator(`SUBMITTED (Locked v${res.annotation.version})`, "#86efac");
      renderAnnotationForm();
      await loadReviewerData();
      showToast(`Submitted ${state.currentItem.workqueue_id} cleanly!`, "ok");
    } catch (err) {
      if (err.details && err.details.errors) showValidationErrors(err.details.errors);
      showToast(`Submission blocked: ${err.message}`, "error");
    }
  }

  async function inspectFrameModal(cam) {
    if (!state.pairTiming) return;
    stopPlayback();
    const url = `/api/frame/${encodeURIComponent(state.pairTiming.pair_id)}/${cam}?t_ref=${state.currentRefTime.toFixed(6)}&fmt=png`;
    $("#frame-modal-title").textContent = `${state.pairTiming.pair_id} · ${cam.toUpperCase()} PTS-Exact Frame`;
    $("#frame-modal-meta").textContent = `Requested t_ref = ${state.currentRefTime.toFixed(3)}s`;
    $("#frame-modal-img").src = `${url}&_=${Date.now()}`;
    $("#frame-modal").classList.remove("hidden");
  }

  async function captureSnapshot(cam) {
    if (!state.currentItem || state.locked) return;
    stopPlayback();
    try {
      const res = await api("/api/reviewer/snapshots", {
        method: "POST",
        body: {
          workqueue_id: state.currentItem.workqueue_id,
          camera: cam,
          t_ref_sec: state.currentRefTime,
          ext: "jpg",
        },
      });
      state.draft.snapshots = state.draft.snapshots || [];
      state.draft.snapshots.push(res.snapshot);
      markDirty();
      renderSnapshotsList();
      showToast(`Captured ${cam} snapshot F${res.snapshot.frame_index}`, "ok");
    } catch (err) {
      showToast(`Snapshot failed: ${err.message}`, "error");
    }
  }

  // ------------------------------------------------------------------
  // History & Reviewer Export
  // ------------------------------------------------------------------
  async function loadHistoryView() {
    if (state.actor !== "reviewer_a" && state.actor !== "reviewer_b") return;
    const bundle = await api("/api/reviewer/history");
    $("#history-records-body").innerHTML = bundle.records
      .map(
        (r) => `
      <tr>
        <td class="mono">${esc(r.workqueue_id)}</td>
        <td class="mono">${esc(r.pair_id)}</td>
        <td><span class="status-pill ${esc(r.status)}">${esc(r.status)}</span></td>
        <td>v${r.version}</td>
        <td class="small">${esc(r.summary.link_status)}</td>
        <td>${r.summary.rider_count}</td>
        <td class="small">${esc(r.summary.violation_event_type)}</td>
        <td><button class="btn btn-xs btn-secondary" data-hist-open="${esc(r.workqueue_id)}">Open</button></td>
      </tr>`
      )
      .join("") || `<tr><td colspan="8" class="muted">No annotations recorded yet for ${esc(state.actor)}.</td></tr>`;

    $("#history-records-body").querySelectorAll("[data-hist-open]").forEach((b) =>
      b.addEventListener("click", () => openWorkqueueItem(b.dataset.histOpen))
    );

    $("#history-timeline-body").innerHTML = bundle.timeline
      .map(
        (h) => `
      <tr>
        <td class="mono small">${esc(h.created_at.slice(0, 19).replace("T", " "))}</td>
        <td class="mono">${esc(h.workqueue_id)}</td>
        <td>${esc(h.action)}</td>
        <td>v${h.version}</td>
        <td class="mono small">${esc(h.payload_sha256.slice(0, 12))}...</td>
      </tr>`
      )
      .join("") || `<tr><td colspan="5" class="muted">No timeline events yet.</td></tr>`;
  }

  // ------------------------------------------------------------------
  // Adjudication Coordinator View (Strictly Forbidden for Reviewers A & B)
  // ------------------------------------------------------------------
  async function loadAdjudicationView() {
    const isAdj = state.actor === "adjudicator";
    $("#adj-restricted-banner").classList.toggle("hidden", isAdj);
    $("#adj-workspace").classList.toggle("hidden", !isAdj);
    if (!isAdj) return;

    const ov = await api("/api/adjudication/overview");
    state.adjOverview = ov;
    const c = ov.counts;
    $("#adj-cnt-partial").textContent = c.pending + c.a_only + c.b_only;
    $("#adj-cnt-agreed").textContent = c.ready_agreed;
    $("#adj-cnt-disagree").textContent = c.ready_disagreed;
    $("#adj-cnt-done").textContent = c.adjudicated;

    const iaa = ov.inter_annotator_agreement;
    const fmtK = (obj) =>
      obj.kappa == null
        ? `N=${obj.n} (Need >=1 paired)`
        : `κ = ${obj.kappa.toFixed(3)} · Agree ${(obj.observed_agreement * 100).toFixed(1)}% (N=${obj.n})`;
    $("#adj-kappa-grid").innerHTML = `
      <div class="card kpi-card"><div class="kpi-label">Binary Violation</div><div class="small mono">${esc(fmtK(iaa.binary_violation))}</div></div>
      <div class="card kpi-card"><div class="kpi-label">Driver Helmet</div><div class="small mono">${esc(fmtK(iaa.rider_helmet))}</div></div>
      <div class="card kpi-card"><div class="kpi-label">Cross-Camera Link</div><div class="small mono">${esc(fmtK(iaa.cross_camera_link))}</div></div>
    `;

    $("#adj-table-body").innerHTML = ov.items
      .map((it) => {
        const canOpen = it.both_submitted;
        return `
        <tr>
          <td class="mono"><strong>${esc(it.workqueue_id)}</strong></td>
          <td class="mono">${esc(it.pair_id)}</td>
          <td class="small">${esc(it.raw_priority_stratum)}<br/><span class="muted">${esc(it.source_origin)}</span></td>
          <td>${it.reviewer_a_submitted ? "✅ Submitted" : "⏳ Pending"}</td>
          <td>${it.reviewer_b_submitted ? "✅ Submitted" : "⏳ Pending"}</td>
          <td>${it.both_submitted ? (it.disagreement_count ? `<span class="status-pill DISAGREEMENT">${it.disagreement_count} diffs</span>` : `<span class="status-pill SUBMITTED">Agreed</span>`) : "--"}</td>
          <td><span class="status-pill">${esc(it.workflow_stage)}</span></td>
          <td><button class="btn btn-xs ${canOpen ? "btn-primary" : "btn-secondary"}" data-adj-open="${esc(it.workqueue_id)}" ${canOpen ? "" : "disabled"}>${canOpen ? "Adjudicate" : "Locked (<2 subs)"}</button></td>
        </tr>`;
      })
      .join("");

    $("#adj-table-body").querySelectorAll("[data-adj-open]").forEach((b) =>
      b.addEventListener("click", () => openAdjudicationItem(b.dataset.adjOpen))
    );
  }

  async function openAdjudicationItem(wqId) {
    try {
      const detail = await api(`/api/adjudication/items/${encodeURIComponent(wqId)}`);
      state.adjDetail = detail;
      $("#adj-detail-panel").classList.remove("hidden");
      $("#adj-video-strip").classList.add("hidden");
      $("#adj-item-id").textContent = detail.item.workqueue_id;
      $("#adj-pair-id").textContent = detail.item.pair_id;

      const diffs = detail.disagreements || [];
      $("#adj-diff-pills").className = diffs.length ? "validation-box warn" : "validation-box";
      $("#adj-diff-pills").innerHTML = diffs.length
        ? `<strong>Disagreements Detected (${diffs.length}):</strong> ` +
          diffs.map((d) => `<span class="mono">${esc(d.field)}: A=${esc(d.reviewer_a)} vs B=${esc(d.reviewer_b)}</span>`).join(" · ")
        : `<strong>Reviewer A & Reviewer B Agreed on all summary fields.</strong> Confirm adjudication record if finalizing.`;

      $("#adj-sub-a").textContent = JSON.stringify(detail.submission_a.payload, null, 2);
      $("#adj-sub-b").textContent = JSON.stringify(detail.submission_b.payload, null, 2);

      const dec = (detail.adjudication && detail.adjudication.decision) || null;
      const base = detail.submission_a.summary;
      $("#adj-status").value = (dec && dec.status) || "ADJUDICATED";
      $("#adj-link").value = (dec && dec.final_link_status) || base.link_status || "LINKED_BOTH_VIEWS";
      $("#adj-event").value = (dec && dec.final_event_type) || base.violation_event_type || "NO_VIOLATION";
      $("#adj-rider-count").value = dec && dec.final_rider_count != null ? dec.final_rider_count : base.rider_count;
      $("#adj-driver-helmet").value = (dec && dec.final_driver_helmet) || base.driver_helmet || "";
      $("#adj-pillion-helmet").value = (dec && dec.final_pillion_helmet) || base.pillion_helmet || "";
      $("#adj-plate-tier").value = (dec && dec.final_plate_visibility_tier) || base.plate_visibility_tier || "PLATE_NOT_VISIBLE_FRONT_ONLY";
      $("#adj-plate-text").value = (dec && dec.final_plate_text_normalized) || base.plate_text_normalized || "";
      $("#adj-win-start").value = dec && dec.final_violation_start_t_ref_sec != null ? dec.final_violation_start_t_ref_sec : "";
      $("#adj-win-end").value = dec && dec.final_violation_end_t_ref_sec != null ? dec.final_violation_end_t_ref_sec : "";
      $("#adj-reason").value = (dec && dec.disagreement_resolution_reason) || "";
      $("#adj-notes").value = (dec && dec.adjudicator_notes) || "";
    } catch (err) {
      showToast(err.message, "error");
    }
  }

  async function saveAdjudicationNow() {
    if (!state.adjDetail) return;
    const wqId = state.adjDetail.item.workqueue_id;
    const body = {
      status: $("#adj-status").value,
      final_link_status: $("#adj-link").value,
      final_event_type: $("#adj-event").value,
      final_rider_count: $("#adj-rider-count").value === "" ? null : Number($("#adj-rider-count").value),
      final_driver_helmet: $("#adj-driver-helmet").value,
      final_pillion_helmet: $("#adj-pillion-helmet").value,
      final_plate_visibility_tier: $("#adj-plate-tier").value,
      final_plate_text_normalized: $("#adj-plate-text").value,
      final_violation_start_t_ref_sec: $("#adj-win-start").value === "" ? null : Number($("#adj-win-start").value),
      final_violation_end_t_ref_sec: $("#adj-win-end").value === "" ? null : Number($("#adj-win-end").value),
      disagreement_resolution_reason: $("#adj-reason").value,
      adjudicator_notes: $("#adj-notes").value,
    };
    try {
      const updated = await api(`/api/adjudication/items/${encodeURIComponent(wqId)}`, {
        method: "POST",
        body,
      });
      state.adjDetail = updated;
      await loadAdjudicationView();
      showToast(`Saved adjudication for ${wqId} (${updated.adjudication.status})`, "ok");
    } catch (err) {
      showToast(`Adjudication error: ${err.message}`, "error");
    }
  }

  // ------------------------------------------------------------------
  // Event Listeners & Bootstrapping
  // ------------------------------------------------------------------
  function bindEvents() {
    $$(".nav-btn").forEach((btn) =>
      btn.addEventListener("click", () => switchView(btn.dataset.view))
    );
    $$("[data-choose-actor]").forEach((btn) =>
      btn.addEventListener("click", () => setSessionActor(btn.dataset.chooseActor))
    );
    $("#actor-select").addEventListener("change", (e) => setSessionActor(e.target.value));

    $("#btn-backup").addEventListener("click", async () => {
      try {
        const res = await api("/api/system/backup", { method: "POST", body: { label: "ui" } });
        showToast(`Backup created: ${res.backup_filename}`, "ok");
      } catch (err) {
        showToast(err.message, "error");
      }
    });

    $("#btn-resume-sidebar").addEventListener("click", () => openWorkqueueItem(state.resumeId || "WQ_001"));
    $("#btn-overview-resume").addEventListener("click", () => openWorkqueueItem(state.resumeId || "WQ_001"));
    $("#btn-export-mine").addEventListener("click", () => {
      window.location.href = "/api/reviewer/export?download=1";
    });
    $("#btn-download-my-export").addEventListener("click", () => {
      window.location.href = "/api/reviewer/export?download=1";
    });

    ["q-search", "q-pair-filter", "q-status-filter", "q-scope-filter", "q-bucket-filter"].forEach((id) => {
      $(`#${id}`).addEventListener("input", renderQueueTable);
      $(`#${id}`).addEventListener("change", renderQueueTable);
    });

    $("#btn-prev-item").addEventListener("click", () => state.prevId && openWorkqueueItem(state.prevId));
    $("#btn-next-item").addEventListener("click", () => state.nextId && openWorkqueueItem(state.nextId));

    $("#layout-select").addEventListener("change", (e) => {
      $("#dual-video-grid").className = `dual-video-grid layout-${e.target.value}`;
    });
    $("#chk-proxy-mode").addEventListener("change", (e) => {
      state.proxyMode = e.target.checked;
      if (state.pairTiming) {
        stopPlayback();
        loadPairVideos(state.pairTiming.pair_id);
        setTimeout(() => seekToRefTime(state.currentRefTime, { fetchFrameInfo: false }), 150);
      }
    });

    // Zoom & pan per viewport
    $$("[data-zoom-cam]").forEach((b) =>
      b.addEventListener("click", () => {
        const cam = b.dataset.zoomCam;
        const delta = Number(b.dataset.zoomDelta);
        state.zoom[cam].scale = Math.max(1.0, Math.min(5.0, state.zoom[cam].scale + delta));
        if (state.zoom[cam].scale <= 1.01) {
          state.zoom[cam].tx = 0;
          state.zoom[cam].ty = 0;
        }
        applyZoomTransform(cam);
      })
    );
    $$("[data-zoom-reset]").forEach((b) =>
      b.addEventListener("click", () => {
        state.zoom[b.dataset.zoomReset] = { scale: 1.0, tx: 0, ty: 0 };
        applyZoomTransform(b.dataset.zoomReset);
      })
    );
    $$(".video-viewport").forEach((vp) => {
      let dragging = false;
      let startX = 0;
      let startY = 0;
      vp.addEventListener("mousedown", (e) => {
        if (!vp.dataset.cam) return;
        dragging = true;
        startX = e.clientX;
        startY = e.clientY;
      });
      window.addEventListener("mousemove", (e) => {
        if (!dragging || !vp.dataset.cam) return;
        const cam = vp.dataset.cam;
        state.zoom[cam].tx += e.clientX - startX;
        state.zoom[cam].ty += e.clientY - startY;
        startX = e.clientX;
        startY = e.clientY;
        applyZoomTransform(cam);
      });
      window.addEventListener("mouseup", () => {
        dragging = false;
      });
    });

    $$("[data-inspect-cam]").forEach((b) =>
      b.addEventListener("click", () => inspectFrameModal(b.dataset.inspectCam))
    );
    $$("[data-snap-cam]").forEach((b) =>
      b.addEventListener("click", () => captureSnapshot(b.dataset.snapCam))
    );
    $("#btn-close-frame-modal").addEventListener("click", () => $("#frame-modal").classList.add("hidden"));

    // Transport & scrubber
    $("#timeline-scrubber").addEventListener("input", (e) => {
      stopPlayback();
      seekToRefTime(Number(e.target.value), { fetchFrameInfo: true });
    });
    $("#btn-play-pause").addEventListener("click", togglePlayback);
    $("#btn-step-back").addEventListener("click", () => stepFrames(-1));
    $("#btn-step-fwd").addEventListener("click", () => stepFrames(1));
    $("#btn-jump-back").addEventListener("click", () => {
      stopPlayback();
      seekToRefTime(state.currentRefTime - 0.5, { fetchFrameInfo: true });
    });
    $("#btn-jump-fwd").addEventListener("click", () => {
      stopPlayback();
      seekToRefTime(state.currentRefTime + 0.5, { fetchFrameInfo: true });
    });
    $("#btn-jump-hint").addEventListener("click", () => {
      if (!state.currentItem) return;
      stopPlayback();
      seekToRefTime(getHintInterval(state.currentItem)[0], { fetchFrameInfo: true });
    });
    $("#btn-jump-overlap").addEventListener("click", () => {
      if (!state.pairTiming) return;
      stopPlayback();
      seekToRefTime(state.pairTiming.overlap_start_ref_sec, { fetchFrameInfo: true });
    });

    // Item form bindings
    $("#fld-non-moto").addEventListener("change", (e) => {
      state.draft.is_non_motorcycle_transit = e.target.checked;
      if (e.target.checked) {
        state.draft.item_link_status = "NOT_A_MOTORCYCLE_TRANSIT";
        state.draft.motorcycles = [];
        state.draft.riders = [];
        state.draft.violation_events = [];
      } else {
        state.draft.item_link_status = "AMBIGUOUS_HUMAN_UNCERTAIN";
      }
      markDirty();
      renderAnnotationForm();
    });
    $("#fld-item-link").addEventListener("change", (e) => {
      state.draft.item_link_status = e.target.value;
      if (e.target.value === "NOT_A_MOTORCYCLE_TRANSIT") {
        state.draft.is_non_motorcycle_transit = true;
        state.draft.motorcycles = [];
        state.draft.riders = [];
        state.draft.violation_events = [];
      } else {
        state.draft.is_non_motorcycle_transit = false;
      }
      markDirty();
      renderAnnotationForm();
    });
    $("#fld-is-uncertain").addEventListener("change", (e) => {
      state.draft.uncertainty.is_uncertain = e.target.checked;
      markDirty();
    });
    $("#fld-reviewer-notes").addEventListener("input", (e) => {
      state.draft.reviewer_notes = e.target.value;
      markDirty();
    });

    $("#btn-add-moto").addEventListener("click", () => {
      const n = state.draft.motorcycles.length + 1;
      let candidate = `M${n}`;
      const existing = new Set(state.draft.motorcycles.map((m) => m.moto_id));
      let c = n;
      while (existing.has(candidate)) {
        c += 1;
        candidate = `M${c}`;
      }
      state.draft.motorcycles.push({
        moto_id: candidate,
        cross_camera_link_status: "LINKED_BOTH_VIEWS",
        visible_front: true,
        visible_back: true,
        start_t_ref_sec: Number(state.currentRefTime.toFixed(3)),
        end_t_ref_sec: Number(state.currentRefTime.toFixed(3)),
        rear_plate: { visibility_tier: "PLATE_UNREADABLE_MOTION_BLUR", plate_text_normalized: "", notes: "" },
        notes: "",
      });
      markDirty();
      renderAnnotationForm();
    });

    $("#btn-add-rider").addEventListener("click", () => {
      const existing = new Set(state.draft.riders.map((r) => r.rider_id));
      let c = state.draft.riders.length + 1;
      while (existing.has(`R${c}`)) c += 1;
      const defaultMoto = state.draft.motorcycles[0].moto_id;
      const sameMoto = state.draft.riders.filter((r) => r.moto_id === defaultMoto);
      state.draft.riders.push({
        rider_id: `R${c}`,
        moto_id: defaultMoto,
        role: sameMoto.some((r) => r.role === "DRIVER") ? "PILLION" : "DRIVER",
        helmet_state: "HELMET",
        front_helmet_evidence: "NOT_VISIBLE",
        back_helmet_evidence: "NOT_VISIBLE",
        is_uncertain: false,
        start_t_ref_sec: null,
        end_t_ref_sec: null,
        notes: "",
      });
      markDirty();
      renderAnnotationForm();
    });

    $("#btn-add-event").addEventListener("click", () => {
      const existing = new Set(state.draft.violation_events.map((ev) => ev.event_id));
      let c = state.draft.violation_events.length + 1;
      while (existing.has(`E${c}`)) c += 1;
      const defaultMoto = state.draft.motorcycles[0].moto_id;
      const riderIds = state.draft.riders.filter((r) => r.moto_id === defaultMoto).map((r) => r.rider_id);
      const t = Number(state.currentRefTime.toFixed(3));
      state.draft.violation_events.push({
        event_id: `E${c}`,
        moto_id: defaultMoto,
        rider_ids: riderIds,
        event_type: "NO_VIOLATION",
        start_t_ref_sec: t,
        end_t_ref_sec: t,
        camera_evidence: "BOTH_CAMERAS",
        keyframe_ref_t_sec: t,
        is_uncertain: false,
        notes: "",
      });
      markDirty();
      renderEventsList();
    });

    $("#btn-restore-local").addEventListener("click", () => {
      const raw = localStorage.getItem(localBackupKey(state.actor, state.currentItem.workqueue_id));
      if (!raw) return;
      state.draft = JSON.parse(raw).draft;
      markDirty();
      renderAnnotationForm();
      $("#recovery-banner").classList.add("hidden");
      showToast("Restored local browser draft.", "ok");
    });
    $("#btn-discard-local").addEventListener("click", () => {
      localStorage.removeItem(localBackupKey(state.actor, state.currentItem.workqueue_id));
      $("#recovery-banner").classList.add("hidden");
    });

    $("#btn-save-draft").addEventListener("click", () => saveDraftNow({ silent: false }));
    $("#btn-reset-draft").addEventListener("click", () => {
      if (!state.currentItem) return;
      state.draft = state.serverRecord
        ? JSON.parse(JSON.stringify(state.serverRecord.payload))
        : defaultDraftFromItem(state.currentItem);
      state.dirty = false;
      localStorage.removeItem(localBackupKey(state.actor, state.currentItem.workqueue_id));
      renderAnnotationForm();
      showValidationErrors([]);
      setSaveIndicator("Reverted to saved version", "#cbd5e1");
    });
    $("#btn-submit-item").addEventListener("click", openSubmitConfirmation);
    $("#btn-cancel-submit").addEventListener("click", () => $("#confirm-modal").classList.add("hidden"));
    $("#btn-confirm-submit").addEventListener("click", executeSubmitNow);

    // Adjudication controls
    $("#btn-adj-export").addEventListener("click", async () => {
      try {
        const res = await api("/api/adjudication/export");
        showToast(`Exported v1.1 bundle (${res.executed_human_units} units) to ${res.exports_dir}`, "ok");
      } catch (err) {
        showToast(err.message, "error");
      }
    });
    $("#btn-adj-open-video").addEventListener("click", () => {
      if (!state.adjDetail) return;
      const pid = state.adjDetail.item.pair_id;
      const strip = $("#adj-video-strip");
      strip.classList.remove("hidden");
      $("#adj-video-front").src = `/api/video/${encodeURIComponent(pid)}/front`;
      $("#adj-video-back").src = `/api/video/${encodeURIComponent(pid)}/back`;
    });
    $("#btn-save-adjudication").addEventListener("click", saveAdjudicationNow);

    // Global keyboard shortcuts (disabled while typing inside inputs/textarea/select)
    window.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") {
        e.preventDefault();
        if (state.view === "review") saveDraftNow({ silent: false });
        return;
      }
      if (state.view !== "review" || !state.pairTiming) return;
      const tag = (document.activeElement && document.activeElement.tagName) || "";
      if (["INPUT", "TEXTAREA", "SELECT"].includes(tag)) return;

      if (e.code === "Space") {
        e.preventDefault();
        togglePlayback();
      } else if (e.key === "[" || (e.key === "ArrowLeft" && !e.shiftKey)) {
        e.preventDefault();
        stepFrames(-1);
      } else if (e.key === "]" || (e.key === "ArrowRight" && !e.shiftKey)) {
        e.preventDefault();
        stepFrames(1);
      } else if (e.key === "ArrowLeft" && e.shiftKey) {
        e.preventDefault();
        stopPlayback();
        seekToRefTime(state.currentRefTime - 0.5, { fetchFrameInfo: true });
      } else if (e.key === "ArrowRight" && e.shiftKey) {
        e.preventDefault();
        stopPlayback();
        seekToRefTime(state.currentRefTime + 0.5, { fetchFrameInfo: true });
      } else if (e.key.toLowerCase() === "l") {
        e.preventDefault();
        $("#chk-loop-hint").checked = !$("#chk-loop-hint").checked;
      }
    });

    window.addEventListener("beforeunload", (e) => {
      if (state.dirty && !state.locked) {
        e.preventDefault();
        e.returnValue = "";
      }
    });
  }

  async function init() {
    bindEvents();
    const [sess, enums] = await Promise.all([api("/api/session"), api("/api/meta/enums")]);
    state.enums = enums;
    state.actor = sess.actor;
    $("#actor-select").value = state.actor || "";
    refreshHealth();

    if (state.actor === "adjudicator") {
      state.view = "adjudication";
      updateViewVisibility();
      await loadAdjudicationView();
    } else if (state.actor) {
      state.view = "overview";
      updateViewVisibility();
      await loadReviewerData();
    } else {
      updateViewVisibility();
    }
  }

  init().catch((err) => {
    showToast(`Init error: ${err.message}`, "error");
  });
})();
