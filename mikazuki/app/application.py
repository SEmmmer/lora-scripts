import asyncio
import mimetypes
import os
import socket
import sys
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException

from mikazuki.app.config import app_config
from mikazuki.app.api import load_schemas, load_presets
from mikazuki.app.api import router as api_router
# from mikazuki.app.ipc import router as ipc_router
from mikazuki.app.proxy import router as proxy_router
from mikazuki.log import log
from mikazuki.utils.devices import check_torch_gpu

mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")
if os.path.exists("./frontend/dist/index.html"):
    FRONTEND_STATIC_DIR = "frontend/dist"
    FRONTEND_INDEX_FILE = "./frontend/dist/index.html"
elif os.path.exists("./frontend/index.html"):
    FRONTEND_STATIC_DIR = "frontend"
    FRONTEND_INDEX_FILE = "./frontend/index.html"
else:
    FRONTEND_STATIC_DIR = None
    FRONTEND_INDEX_FILE = None

_WORKER_MODE_GUARD_INJECTION = """
<style id="mikazuki-worker-rank-guard-style">
[data-mikazuki-worker-hidden="1"] { display: none !important; }
</style>
<script id="mikazuki-worker-rank-guard">
(function () {
  if (window.__MIKAZUKI_WORKER_RANK_GUARD__) return;
  window.__MIKAZUKI_WORKER_RANK_GUARD__ = true;

  var state = {
    confirmed: false,
    lastRank: 0,
    boundInput: null,
    busy: false
  };

  var DIST_HEADING_RE = /(分布式训练|distributed training)/i;
  var DIST_ITEM_RE = /(enable_distributed_training|machine_rank|num_machines|num_processes|main_process_ip|main_process_port|nccl_socket_ifname|gloo_socket_ifname|sync_from_main_settings|sync_config_from_main|sync_main_toml|sync_ssh_user|sync_ssh_port|sync_ssh_password)/i;

  function getSchemaForm() {
    return (
      document.querySelector(".k-form") ||
      document.querySelector(".schema-container form") ||
      document.querySelector("form")
    );
  }

  function getSchemaItems() {
    var form = getSchemaForm();
    if (!form) return [];
    return form.querySelectorAll(".k-schema-item");
  }

  function toInt(value) {
    var n = parseInt(value, 10);
    return Number.isFinite(n) ? n : 0;
  }

  function findMachineRankField() {
    var items = getSchemaItems();
    for (var i = 0; i < items.length; i++) {
      var item = items[i];
      var title = item.querySelector("h3");
      if (!title) continue;
      var text = (title.textContent || "").trim();
      if (!/machine_rank/i.test(text)) continue;
      var input = item.querySelector("input");
      if (!input) continue;
      return { item: item, input: input };
    }
    return null;
  }

  function clearWorkerHidden(form) {
    var hidden = form.querySelectorAll("[data-mikazuki-worker-hidden='1']");
    for (var i = 0; i < hidden.length; i++) {
      hidden[i].removeAttribute("data-mikazuki-worker-hidden");
    }
  }

  function collapseNonDistributedModules(form) {
    var children = Array.prototype.slice.call(form.children || []);
    var hasHeading = false;
    var keepCurrent = false;
    var foundDistributedHeading = false;

    for (var i = 0; i < children.length; i++) {
      var el = children[i];
      if (el.tagName === "H2") {
        hasHeading = true;
        var headingText = (el.textContent || "").trim();
        keepCurrent = DIST_HEADING_RE.test(headingText);
        if (keepCurrent) foundDistributedHeading = true;
        continue;
      }
      if (hasHeading && !keepCurrent) {
        el.setAttribute("data-mikazuki-worker-hidden", "1");
      }
    }

    if (!hasHeading || !foundDistributedHeading) {
      var items = form.querySelectorAll(".k-schema-item");
      for (var j = 0; j < items.length; j++) {
        var item = items[j];
        var title = item.querySelector("h3");
        var text = (title && title.textContent ? title.textContent : "").trim();
        if (!DIST_ITEM_RE.test(text)) {
          item.setAttribute("data-mikazuki-worker-hidden", "1");
        }
      }
    }
  }

  function applyWorkerMode(enabled) {
    var form = getSchemaForm();
    if (!form) return;
    clearWorkerHidden(form);
    if (enabled) {
      collapseNonDistributedModules(form);
    }
  }

  function setRankInputValue(input, rank) {
    input.value = String(rank);
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
  }

  function onRankMaybeChanged(force) {
    var field = findMachineRankField();
    if (!field) return;
    if (state.busy) return;

    var rank = toInt(field.input.value);
    if (!force && rank === state.lastRank) return;
    state.lastRank = rank;

    if (rank !== 0 && !state.confirmed) {
      var ok = window.confirm("检测到 machine_rank 不为 0，将把当前机器设置为从机。确认后会折叠除分布式配置外的其他模块。是否继续？");
      if (!ok) {
        state.busy = true;
        setRankInputValue(field.input, 0);
        state.lastRank = 0;
        state.confirmed = false;
        applyWorkerMode(false);
        state.busy = false;
        return;
      }
      state.confirmed = true;
      applyWorkerMode(true);
      return;
    }

    if (rank === 0) {
      state.confirmed = false;
      applyWorkerMode(false);
      return;
    }

    if (state.confirmed) {
      applyWorkerMode(true);
    }
  }

  function bindRankListeners() {
    var field = findMachineRankField();
    if (!field) return false;
    if (state.boundInput === field.input) return true;

    state.boundInput = field.input;
    state.lastRank = toInt(field.input.value);
    field.input.addEventListener("input", function () { onRankMaybeChanged(false); });
    field.input.addEventListener("change", function () { onRankMaybeChanged(false); });

    var buttons = field.item.querySelectorAll(".el-input-number__increase, .el-input-number__decrease");
    for (var i = 0; i < buttons.length; i++) {
      buttons[i].addEventListener("click", function () {
        setTimeout(function () { onRankMaybeChanged(false); }, 0);
      });
    }
    return true;
  }

  function tick() {
    bindRankListeners();
    if (state.confirmed) applyWorkerMode(true);
  }

  var observer = new MutationObserver(function () {
    bindRankListeners();
    if (state.confirmed) applyWorkerMode(true);
  });
  observer.observe(document.documentElement, { childList: true, subtree: true });

  window.addEventListener("load", function () { bindRankListeners(); });
  setInterval(tick, 400);
})();
</script>
"""

_SCHEMA_BOOTSTRAP_INJECTION = """
<script id="mikazuki-schema-bootstrap">
(function () {
  if (window.__MIKAZUKI_SCHEMA_BOOTSTRAP__) return;
  window.__MIKAZUKI_SCHEMA_BOOTSTRAP__ = true;

  function hasSharedSchemaInLocalStorage() {
    try {
      var raw = localStorage.getItem("schemas");
      if (!raw) return false;
      var list = JSON.parse(raw);
      if (!Array.isArray(list)) return false;
      for (var i = 0; i < list.length; i++) {
        var item = list[i];
        if (item && item.name === "shared" && typeof item.schema === "string" && item.schema.length > 0) {
          return true;
        }
      }
      return false;
    } catch (_e) {
      return false;
    }
  }

  try {
    if (hasSharedSchemaInLocalStorage()) return;

    // Load schema cache before app bootstrap to avoid first-render race.
    var xhr = new XMLHttpRequest();
    xhr.open("GET", "/api/schemas/all", false);
    xhr.send(null);

    if (xhr.status < 200 || xhr.status >= 300) return;
    var payload = JSON.parse(xhr.responseText || "{}");
    var schemas = payload && payload.status === "success" && payload.data && Array.isArray(payload.data.schemas)
      ? payload.data.schemas
      : null;
    if (!schemas || schemas.length === 0) return;

    localStorage.setItem("schemas", JSON.stringify(schemas));
    window.__MIKAZUKI_SCHEMA_BOOTSTRAPPED__ = true;
  } catch (_err) {
    // Ignore bootstrap errors and let frontend fallback logic continue.
  }
})();
</script>
"""

_STAGED_RESOLUTION_PREVIEW_INJECTION = """
<style id="mikazuki-staged-resolution-preview-style">
#mikazuki-staged-resolution-preview-block {
  margin: 8px 0 12px 0;
  padding: 10px 12px;
  border: 1px solid #d8e4f3;
  border-radius: 6px;
  background: #f6f9ff;
  font-size: 12px;
  line-height: 1.5;
}
#mikazuki-staged-resolution-preview-block .stage-title {
  font-weight: 600;
  margin-bottom: 6px;
}
#mikazuki-staged-resolution-preview-block .stage-note {
  color: #606266;
  margin-top: 6px;
}
#mikazuki-staged-resolution-preview-block table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 8px;
}
#mikazuki-staged-resolution-preview-block th,
#mikazuki-staged-resolution-preview-block td {
  border: 1px solid #dfe7f3;
  padding: 4px 6px;
  text-align: left;
}
#mikazuki-staged-resolution-status {
  margin: 8px 0 8px 0;
  padding: 6px 10px;
  border: 1px dashed #c8d6ea;
  border-radius: 6px;
  background: #f8fbff;
  color: #606266;
  font-size: 12px;
}
</style>
<script id="mikazuki-staged-resolution-preview">
(function () {
  if (window.__MIKAZUKI_STAGED_RESOLUTION_PREVIEW__) return;
  window.__MIKAZUKI_STAGED_RESOLUTION_PREVIEW__ = true;

  var PHASES = [
    { side: 512, ratioKey: "staged_resolution_ratio_512", defaultRatioPercent: 40, sampleScale: 4.0 },
    { side: 768, ratioKey: "staged_resolution_ratio_768", defaultRatioPercent: 30, sampleScale: 1.78 },
    { side: 1024, ratioKey: "staged_resolution_ratio_1024", defaultRatioPercent: 30, sampleScale: 1.0 }
  ];
  var state = {
    notifiedResolutionFix: false,
    trainTypeInput: null,
    trainTypeValue: "",
    scheduled: false,
    updating: false,
    trainImageCountKey: "",
    trainImageCount: null,
    trainImageCountLoading: false,
    trainImageCountError: ""
  };

  function getSchemaForm() {
    return (
      document.querySelector(".k-form") ||
      document.querySelector(".schema-container form") ||
      document.querySelector("form")
    );
  }

  function getSchemaItems() {
    var form = getSchemaForm();
    if (!form) return [];
    return form.querySelectorAll(".k-schema-item");
  }

  function toInt(value, fallback) {
    var text = String(value == null ? "" : value).trim();
    if (text === "") return fallback;
    var n = parseInt(text, 10);
    return Number.isFinite(n) ? n : fallback;
  }

  function toFloat(value, fallback) {
    var text = String(value == null ? "" : value).trim();
    if (text === "") return fallback;
    var n = parseFloat(text);
    return Number.isFinite(n) ? n : fallback;
  }

  function ceilToMultiple(value, base) {
    if (!Number.isFinite(base) || base <= 0) return value;
    return Math.ceil(value / base) * base;
  }

  function gcdInt(a, b) {
    var x = Math.abs(toInt(a, 0));
    var y = Math.abs(toInt(b, 0));
    while (y !== 0) {
      var t = x % y;
      x = y;
      y = t;
    }
    return x <= 0 ? 1 : x;
  }

  function lcmInt(a, b) {
    var x = Math.max(1, toInt(a, 1));
    var y = Math.max(1, toInt(b, 1));
    return Math.abs(x * y) / gcdInt(x, y);
  }

  function getItemSearchText(item, title) {
    var titleText = (title && title.textContent ? title.textContent : "").trim();
    var itemText = (item && item.textContent ? item.textContent : "").trim();
    return (titleText + " " + itemText).toLowerCase();
  }

  function findSchemaField(patterns) {
    var items = getSchemaItems();
    if (!patterns || patterns.length === 0) return null;
    for (var i = 0; i < items.length; i++) {
      var item = items[i];
      var title = item.querySelector("h3");
      var searchText = getItemSearchText(item, title);
      for (var j = 0; j < patterns.length; j++) {
        var re = patterns[j];
        if (re.test(searchText)) {
          return { item: item, title: title, searchText: searchText };
        }
      }
    }
    return null;
  }

  function findInput(item, preferCheckbox) {
    if (!item) return null;
    if (preferCheckbox) {
      var cb = item.querySelector("input[type='checkbox']");
      if (cb) return cb;
    }
    var inputs = item.querySelectorAll("input");
    for (var i = 0; i < inputs.length; i++) {
      var input = inputs[i];
      var type = (input.getAttribute("type") || "").toLowerCase();
      if (type === "hidden") continue;
      if (type === "checkbox" && !preferCheckbox) continue;
      return input;
    }
    return null;
  }

  function dispatchInput(input) {
    if (!input) return;
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
  }

  function readBool(item) {
    var input = findInput(item, true);
    if (!input) return false;
    var type = (input.getAttribute("type") || "").toLowerCase();
    if (type === "checkbox") return !!input.checked;
    var raw = String(input.value || "").trim().toLowerCase();
    return raw === "1" || raw === "true" || raw === "yes" || raw === "on";
  }

  function readInt(item, fallback) {
    var input = findInput(item, false);
    if (!input) return fallback;
    return toInt(input.value, fallback);
  }

  function readFloat(item, fallback) {
    var input = findInput(item, false);
    if (!input) return fallback;
    return toFloat(input.value, fallback);
  }

  function readText(item) {
    var input = findInput(item, false);
    if (!input) return "";
    return String(input.value || "");
  }

  function setText(item, value) {
    var input = findInput(item, false);
    if (!input) return false;
    if (String(input.value || "").trim() === String(value).trim()) return false;
    input.value = value;
    dispatchInput(input);
    return true;
  }

  function requestTrainImageCount(trainDataDir) {
    var key = String(trainDataDir == null ? "" : trainDataDir).trim();
    if (!key) {
      state.trainImageCountKey = "";
      state.trainImageCount = null;
      state.trainImageCountLoading = false;
      state.trainImageCountError = "";
      return;
    }

    if (state.trainImageCountKey === key && (state.trainImageCountLoading || state.trainImageCount !== null || state.trainImageCountError)) {
      return;
    }

    state.trainImageCountKey = key;
    state.trainImageCount = null;
    state.trainImageCountLoading = true;
    state.trainImageCountError = "";

    fetch("/api/staged_resolution_train_image_count", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ train_data_dir: key })
    })
      .then(function (resp) { return resp.json(); })
      .then(function (data) {
        if (state.trainImageCountKey !== key) return;
        if (!data || data.status !== "success" || !data.data) {
          state.trainImageCount = null;
          state.trainImageCountError = data && data.message ? String(data.message) : "failed to count train images";
          return;
        }
        var n = toInt(data.data.total_train_images_with_repeats, 0);
        if (n > 0) {
          state.trainImageCount = n;
          state.trainImageCountError = "";
        } else {
          state.trainImageCount = null;
          state.trainImageCountError = "total_train_images_with_repeats is invalid";
        }
      })
      .catch(function () {
        if (state.trainImageCountKey !== key) return;
        state.trainImageCount = null;
        state.trainImageCountError = "failed to count train images";
      })
      .finally(function () {
        if (state.trainImageCountKey !== key) return;
        state.trainImageCountLoading = false;
        scheduleUpdate();
      });
  }

  function normalizeTrainTypeValue(raw) {
    return String(raw == null ? "" : raw).trim().toLowerCase();
  }

  function bindTrainTypeListener() {
    var trainTypeField = findSchemaField([
      /model_train_type/i,
      /训练种类/,
      /train type/
    ]);
    if (!trainTypeField) return;

    var input = findInput(trainTypeField.item, false);
    if (!input || state.trainTypeInput === input) return;

    state.trainTypeInput = input;
    state.trainTypeValue = normalizeTrainTypeValue(input.value);

    var handler = function () {
      var now = normalizeTrainTypeValue(input.value);
      if (now === state.trainTypeValue) return;
      state.trainTypeValue = now;

      // Keep preview refresh behavior aligned with model-train-type UI updates.
      setTimeout(update, 0);
      setTimeout(update, 120);
    };
    input.addEventListener("input", handler);
    input.addEventListener("change", handler);
    input.addEventListener("blur", handler);
  }

  function parseResolution(rawValue) {
    var raw = String(rawValue == null ? "" : rawValue).trim().toLowerCase().replace(/x/g, ",");
    if (!raw) return null;
    var parts = raw.split(",").map(function (x) { return x.trim(); }).filter(function (x) { return x !== ""; });
    if (parts.length === 1) {
      var side = toInt(parts[0], -1);
      if (side <= 0) return null;
      return [side, side];
    }
    var width = toInt(parts[0], -1);
    var height = toInt(parts[1], -1);
    if (width <= 0 || height <= 0) return null;
    return [width, height];
  }

  function isBaseResolution1024(rawValue) {
    var pair = parseResolution(rawValue);
    return !!pair && pair[0] === 1024 && pair[1] === 1024;
  }

  function ensurePreviewBlock(anchorItem) {
    if (!anchorItem || !anchorItem.parentNode) return null;
    var form = getSchemaForm();
    if (!form) return null;
    var block = form.querySelector("#mikazuki-staged-resolution-preview-block");
    if (!block) {
      block = document.createElement("div");
      block.id = "mikazuki-staged-resolution-preview-block";
    }

    if (anchorItem.nextSibling !== block) {
      anchorItem.parentNode.insertBefore(block, anchorItem.nextSibling);
    }
    return block;
  }

  function ensureStatusBlock(anchorItem) {
    if (!anchorItem || !anchorItem.parentNode) return null;
    var form = getSchemaForm();
    if (!form) return null;
    var block = form.querySelector("#mikazuki-staged-resolution-status");
    if (!block) {
      block = document.createElement("div");
      block.id = "mikazuki-staged-resolution-status";
    }
    if (anchorItem.previousSibling !== block) {
      anchorItem.parentNode.insertBefore(block, anchorItem);
    }
    return block;
  }

  function resolvePhaseRatios() {
    var ratios = {};
    var total = 0;
    var hasPositive = false;

    for (var i = 0; i < PHASES.length; i++) {
      var phase = PHASES[i];
      var ratioField = findSchemaField([
        new RegExp(phase.ratioKey, "i"),
        new RegExp(String(phase.side) + ".*占比"),
        new RegExp("占比.*" + String(phase.side))
      ]);
      var value = ratioField ? readFloat(ratioField.item, phase.defaultRatioPercent) : phase.defaultRatioPercent;
      if (!Number.isFinite(value)) {
        return { ok: false, message: phase.ratioKey + " 不是有效数字" };
      }
      if (value < 0 || value > 100) {
        return { ok: false, message: phase.ratioKey + " 超出范围: " + value + "（仅允许 0~100）" };
      }

      ratios[phase.side] = value;
      total += value;
      if (value > 0) hasPositive = true;
    }

    if (total > 100 + 1e-9) {
      return { ok: false, message: "阶段占比总和不能大于 100，当前为 " + total.toFixed(4) };
    }
    if (!hasPositive) {
      return { ok: false, message: "阶段占比总和为 0，至少需要一个阶段大于 0" };
    }
    return { ok: true, ratios: ratios, total: total };
  }

  function buildPhasePreview(baseBatch, baseGradAccum, baseEpochs, saveEveryEpochs, baseSampleEveryEpochs, previewEnabled, phaseRatioPercents, totalTrainImages, numProcesses) {
    var rows = [];
    var totalEpochs = 0;
    var safeTrainImages = Math.max(0, toInt(totalTrainImages, 0));
    var safeNumProcesses = Math.max(1, toInt(numProcesses, 1));
    var hasOversizedBatchPhase = false;
    var totalLoopEpochs = 0;
    for (var i = 0; i < PHASES.length; i++) {
      var phase = PHASES[i];
      var side = phase.side;
      var ratioPercent = toFloat((phaseRatioPercents || {})[side], phase.defaultRatioPercent);
      if (!Number.isFinite(ratioPercent)) ratioPercent = phase.defaultRatioPercent;
      if (ratioPercent < 0) ratioPercent = 0;
      if (ratioPercent > 100) ratioPercent = 100;
      var ratio = ratioPercent / 100.0;
      var phaseBatch = Math.max(1, Math.floor(baseBatch * (1024 * 1024) / (side * side)));
      var phaseGradAccum = baseGradAccum <= 1 ? 1 : baseGradAccum;
      var effectiveBatchRatio = (phaseBatch * phaseGradAccum) / (baseBatch * baseGradAccum);
      var rawEpochs = Math.ceil(baseEpochs * ratio * effectiveBatchRatio);
      var phaseSaveEveryEpochs = Math.max(1, Math.ceil(saveEveryEpochs * phase.sampleScale));
      var phaseSampleEveryEpochs = Math.max(1, Math.ceil(baseSampleEveryEpochs * phase.sampleScale));
      var epochRoundBase = phaseSaveEveryEpochs;
      if (previewEnabled) {
        epochRoundBase = lcmInt(phaseSaveEveryEpochs, phaseSampleEveryEpochs);
      }
      var phaseEpochs = 0;
      if (ratioPercent > 0) {
        phaseEpochs = Math.max(1, ceilToMultiple(rawEpochs, epochRoundBase));
      }
      totalEpochs += phaseEpochs;

      var batchesPerEpoch = null;
      var stepsPerEpoch = null;
      var phaseTargetSamples = null;
      var phaseGlobalBatch = null;
      var phaseLoopEpochs = null;
      var loopEpochEquivalent = null;
      if (safeTrainImages > 0 && phaseEpochs > 0) {
        batchesPerEpoch = Math.max(1, Math.ceil(safeTrainImages / phaseBatch));
        stepsPerEpoch = Math.max(1, Math.ceil(batchesPerEpoch / phaseGradAccum));
        phaseTargetSamples = phaseEpochs * safeTrainImages;
        phaseGlobalBatch = phaseBatch * phaseGradAccum * safeNumProcesses;
        var phaseTargetSteps = Math.max(1, Math.ceil(phaseTargetSamples / Math.max(1, phaseGlobalBatch)));
        phaseLoopEpochs = Math.max(1, Math.ceil(phaseTargetSteps / Math.max(1, stepsPerEpoch)));
        loopEpochEquivalent = (stepsPerEpoch * phaseGlobalBatch) / safeTrainImages;
        totalLoopEpochs += phaseLoopEpochs;
        if (phaseBatch > safeTrainImages) {
          hasOversizedBatchPhase = true;
        }
      }
      rows.push({
        side: side,
        ratioPercent: ratioPercent,
        batch: phaseBatch,
        gradAccum: phaseGradAccum,
        effectiveBatchRatio: effectiveBatchRatio,
        saveEveryEpochs: phaseSaveEveryEpochs,
        sampleEveryEpochs: phaseSampleEveryEpochs,
        sampleScale: phase.sampleScale,
        rawEpochs: rawEpochs,
        epochs: phaseEpochs,
        loopEpochs: phaseLoopEpochs,
        phaseTargetSamples: phaseTargetSamples,
        globalBatch: phaseGlobalBatch,
        batchesPerEpoch: batchesPerEpoch,
        stepsPerEpoch: stepsPerEpoch,
        loopEpochEquivalent: loopEpochEquivalent,
        epochRoundBase: epochRoundBase,
        rawFormula: "ceil(" + baseEpochs + " * (" + ratioPercent + " / 100) * ((" + phaseBatch + "*" + phaseGradAccum + ") / (" + baseBatch + "*" + baseGradAccum + ")))",
        actualFormula: ratioPercent > 0
          ? (previewEnabled
              ? "ceil_to_multiple(raw, lcm(save=" + phaseSaveEveryEpochs + ", sample=" + phaseSampleEveryEpochs + ")=" + epochRoundBase + ")"
              : "ceil_to_multiple(raw, save=" + phaseSaveEveryEpochs + ")")
          : "ratio=0 -> skipped"
      });
    }
    return {
      rows: rows,
      totalEpochs: totalEpochs,
      previewEnabled: !!previewEnabled,
      baseGradAccum: baseGradAccum,
      saveEveryEpochs: saveEveryEpochs,
      baseSampleEveryEpochs: baseSampleEveryEpochs,
      ratioSumPercent: rows.reduce(function (acc, row) { return acc + row.ratioPercent; }, 0),
      totalTrainImages: safeTrainImages,
      numProcesses: safeNumProcesses,
      totalLoopEpochs: totalLoopEpochs,
      hasOversizedBatchPhase: hasOversizedBatchPhase
    };
  }

  function renderPreview(block, preview) {
    var html = "";
    html += '<div class="stage-title">阶段分辨率预览（启用后将按以下计划训练）</div>';
    html += "<table>";
    html += "<thead><tr><th>分辨率</th><th>占比(%)</th><th>Batch</th><th>梯度累加步数</th><th>CKPT 每 N Epoch</th><th>Sample 每 N Epoch</th><th>Raw Epoch</th><th>等效 Epoch(样本口径)</th><th>Loop Epoch(训练器口径)</th><th>取整基数</th></tr></thead><tbody>";
    for (var i = 0; i < preview.rows.length; i++) {
      var row = preview.rows[i];
      html += "<tr>";
      html += "<td>" + row.side + "x" + row.side + "</td>";
      html += "<td>" + row.ratioPercent + "</td>";
      html += "<td>" + row.batch + "</td>";
      html += "<td>" + row.gradAccum + "（与1024基准一致）</td>";
      html += "<td>" + row.saveEveryEpochs + " (x" + row.sampleScale + ")</td>";
      html += "<td>" + row.sampleEveryEpochs + " (x" + row.sampleScale + ")</td>";
      html += "<td>" + row.rawEpochs + "</td>";
      html += "<td>" + row.epochs + "</td>";
      html += "<td>" + (row.loopEpochs == null ? "-" : row.loopEpochs) + "</td>";
      html += "<td>" + row.epochRoundBase + "</td>";
      html += "</tr>";
    }
    html += "</tbody></table>";
    html += '<div class="stage-note">总等效 Epoch（阶段累计）: ' + preview.totalEpochs + "；占比总和: " + preview.ratioSumPercent + "%（要求 ≤ 100%）</div>";
    if (preview.totalTrainImages > 0) {
      html += '<div class="stage-note">数据集样本数（含 repeats）: ' + preview.totalTrainImages + "；num_processes: " + preview.numProcesses + "；总 Loop Epoch 估算: " + preview.totalLoopEpochs + "</div>";
    } else {
      html += '<div class="stage-note">未拿到数据集样本数，Loop Epoch 暂无法精确估算（可正常训练，不影响后端真实计算）。</div>';
    }
    html += '<div class="stage-note">Raw 公式: raw_epoch = ceil(base_epoch * phase_ratio * ((phase_batch*phase_grad_accum) / (base_batch*base_grad_accum)))</div>';
    html += '<div class="stage-note">梯度累加规则: x=1 时全阶段保持 1；x>1 时全阶段保持 x（以 1024 基准为准），当前 x=' + preview.baseGradAccum + "</div>";
    if (preview.previewEnabled) {
      html += '<div class="stage-note">实际公式: actual_epoch = ceil_to_multiple(raw_epoch, lcm(phase_save_every_n_epochs, phase_sample_every_n_epochs))</div>';
    } else {
      html += '<div class="stage-note">实际公式: actual_epoch = ceil_to_multiple(raw_epoch, phase_save_every_n_epochs)（当前未启用训练预览图）</div>';
    }
    if (preview.hasOversizedBatchPhase) {
      html += '<div class="stage-note">注意：存在 phase_batch > 数据集样本数。DataLoader 会重复采样补满 batch，不会留空位，因此 Loop Epoch 可能小于等效 Epoch。</div>';
    }
    html += '<div class="stage-note">CKPT 规则: 1024=x, 768=ceil(1.78x), 512=ceil(4x)，x=' + preview.saveEveryEpochs + "（可调占比）</div>";
    html += '<div class="stage-note">Sample 规则: 1024=x, 768=ceil(1.78x), 512=ceil(4x)，x=' + preview.baseSampleEveryEpochs + "（可调占比）</div>";
    html += "<table><thead><tr><th>阶段</th><th>Raw 验算</th><th>实际验算</th></tr></thead><tbody>";
    for (var j = 0; j < preview.rows.length; j++) {
      var r = preview.rows[j];
      html += "<tr><td>" + r.side + "x" + r.side + "</td><td>" + r.rawFormula + " = " + r.rawEpochs + "</td><td>" + r.actualFormula + " = " + r.epochs + "</td></tr>";
    }
    html += "</tbody></table>";
    if (block.innerHTML !== html) {
      block.innerHTML = html;
    }
    if (block.style.display !== "") {
      block.style.display = "";
    }
  }

  function renderInvalidPreview(block, message) {
    var note = message || "请先填写有效的 max_train_epochs / train_batch_size。";
    var html = '<div class="stage-title">阶段分辨率预览</div><div class="stage-note">' + note + "</div>";
    if (block.innerHTML !== html) {
      block.innerHTML = html;
    }
    if (block.style.display !== "") {
      block.style.display = "";
    }
  }

  function maybeRenameSectionTitle(enableField) {
    if (!enableField || !enableField.item || !enableField.item.parentElement) return;
    var prev = enableField.item.previousElementSibling;
    while (prev) {
      if (prev.tagName === "H2") {
        var text = (prev.textContent || "").trim();
        if (/混合分辨率训练/.test(text)) {
          prev.textContent = text.replace(/混合分辨率训练/g, "阶段分辨率训练");
        }
        break;
      }
      prev = prev.previousElementSibling;
    }

    if (enableField.title) {
      var itemTitle = (enableField.title.textContent || "").trim();
      if (/混合分辨率/.test(itemTitle)) {
        enableField.title.textContent = itemTitle.replace(/混合分辨率/g, "阶段分辨率");
      }
    }
  }

  function updateStatus(statusBlock, message) {
    if (!statusBlock) return;
    var next = String(message == null ? "" : message);
    if (statusBlock.textContent !== next) {
      statusBlock.textContent = next;
    }
    if (statusBlock.style.display !== "") {
      statusBlock.style.display = "";
    }
  }

  function update() {
    if (state.updating) return;
    state.updating = true;
    try {
    bindTrainTypeListener();

    var enableField = findSchemaField([
      /enable_mixed_resolution_training/i,
      /阶段分辨率/,
      /混合分辨率/
    ]);
    if (!enableField) return;
    maybeRenameSectionTitle(enableField);

    var enabled = readBool(enableField.item);
    var statusBlock = ensureStatusBlock(enableField.item);
    var previewBlock = ensurePreviewBlock(enableField.item);
    if (!previewBlock) return;

    if (!enabled) {
      state.notifiedResolutionFix = false;
      if (statusBlock) {
        updateStatus(statusBlock, "阶段分辨率脚本已加载。当前未启用“阶段分辨率训练”。");
      }
      if (previewBlock.style.display !== "none") {
        previewBlock.style.display = "none";
      }
      return;
    }

    var resolutionField = findSchemaField([
      /(^|[^a-z])resolution([^a-z]|$)/i,
      /训练图片分辨率/,
      /训练分辨率/
    ]);
    if (resolutionField) {
      var resolutionText = readText(resolutionField.item);
      if (!isBaseResolution1024(resolutionText)) {
        var changed = setText(resolutionField.item, "1024,1024");
        if (changed && !state.notifiedResolutionFix) {
          state.notifiedResolutionFix = true;
          setTimeout(function () {
            window.alert("已启用阶段分辨率训练：该模式固定以 1024,1024 为基础分辨率，已自动调整当前训练分辨率。");
          }, 0);
        }
      }
    }

    var batchField = findSchemaField([
      /train_batch_size/i,
      /批量大小/
    ]);
    var epochsField = findSchemaField([
      /max_train_epochs/i,
      /最大训练\\s*epoch/,
      /最大训练轮/
    ]);
    var gradAccumField = findSchemaField([
      /gradient_accumulation_steps/i,
      /梯度累加/
    ]);
    var saveEveryField = findSchemaField([
      /save_every_n_epochs/i,
      /每\\s*n\\s*epoch/,
      /自动保存/
    ]);
    var previewEnableField = findSchemaField([
      /enable_preview/i,
      /启用训练预览图/
    ]);
    var sampleEveryField = findSchemaField([
      /sample_every_n_epochs/i,
      /每\\s*n\\s*个\\s*epoch\\s*生成/
    ]);
    var trainDataDirField = findSchemaField([
      /train_data_dir/i,
      /训练数据集路径/
    ]);
    var numProcessesField = findSchemaField([
      /num_processes/i,
      /进程数/
    ]);

    var baseBatch = batchField ? readInt(batchField.item, 0) : 0;
    var baseGradAccum = gradAccumField ? readInt(gradAccumField.item, 1) : 1;
    var baseEpochs = epochsField ? readInt(epochsField.item, 0) : 0;
    var saveEveryEpochs = saveEveryField ? readInt(saveEveryField.item, 1) : 1;
    var previewEnabled = previewEnableField ? readBool(previewEnableField.item) : false;
    var baseSampleEveryEpochs = sampleEveryField ? readInt(sampleEveryField.item, 1) : 1;
    var trainDataDir = trainDataDirField ? readText(trainDataDirField.item).trim() : "";
    var numProcesses = numProcessesField ? readInt(numProcessesField.item, 1) : 1;
    if (baseGradAccum <= 0) baseGradAccum = 1;
    if (saveEveryEpochs <= 0) saveEveryEpochs = 1;
    if (baseSampleEveryEpochs <= 0) baseSampleEveryEpochs = 1;
    if (numProcesses <= 0) numProcesses = 1;

    requestTrainImageCount(trainDataDir);

    var ratioState = resolvePhaseRatios();
    if (!ratioState.ok) {
      if (statusBlock) {
        updateStatus(statusBlock, "阶段分辨率占比配置错误: " + ratioState.message);
      }
      renderInvalidPreview(previewBlock, ratioState.message);
      return;
    }

    if (baseBatch <= 0 || baseEpochs <= 0) {
      if (statusBlock) {
        updateStatus(statusBlock, "阶段分辨率已启用，但未找到有效的 batch/epoch 输入值，请检查 train_batch_size 与 max_train_epochs。");
      }
      renderInvalidPreview(previewBlock);
      return;
    }

    var preview = buildPhasePreview(
      baseBatch,
      baseGradAccum,
      baseEpochs,
      saveEveryEpochs,
      baseSampleEveryEpochs,
      previewEnabled,
      ratioState.ratios,
      state.trainImageCount,
      numProcesses
    );
    if (statusBlock) {
      if (previewEnabled) {
        updateStatus(
          statusBlock,
          "阶段分辨率已启用：占比总和 " + ratioState.total.toFixed(4) + "%（<=100%）；梯度累加保持与 1024 基准一致，ckpt 与 sample 频率按 1024=x, 768=ceil(1.78x), 512=ceil(4x)（向上取整），并纳入实际 epoch 取整。"
        );
      } else {
        updateStatus(
          statusBlock,
          "阶段分辨率已启用：占比总和 " + ratioState.total.toFixed(4) + "%（<=100%）；梯度累加保持与 1024 基准一致，ckpt 频率按 1024=x, 768=ceil(1.78x), 512=ceil(4x)（向上取整），并已实时预估三阶段 batch 与 epoch（未启用训练预览图时不纳入 sample 频率取整）。"
        );
      }
    }
    renderPreview(previewBlock, preview);
    } finally {
      state.updating = false;
    }
  }

  function scheduleUpdate() {
    if (state.scheduled) return;
    state.scheduled = true;
    setTimeout(function () {
      state.scheduled = false;
      update();
    }, 50);
  }

  var observer = new MutationObserver(function () {
    scheduleUpdate();
  });
  observer.observe(document.documentElement, { childList: true, subtree: true });

  window.addEventListener("load", function () { scheduleUpdate(); });
  setInterval(scheduleUpdate, 1000);
})();
</script>
"""

_HIDE_DEPRECATED_LORA_DOCS_INJECTION = """
<script id="mikazuki-hide-deprecated-lora-docs">
(function () {
  if (window.__MIKAZUKI_HIDE_DEPRECATED_LORA_DOCS__) return;
  window.__MIKAZUKI_HIDE_DEPRECATED_LORA_DOCS__ = true;

  var BLOCKED_PATH_RE = /^\\/lora\\/(basic|flux|sd3|tools|params)(?:\\.(?:html|md))?\\/?$/i;
  var BLOCKED_LINK_RE = /\\/lora\\/(basic|flux|sd3|tools|params)\\.md$/i;
  var BLOCKED_LABEL_RE = /^(新手（SD1\\.5）|Flux|SD3\\.5|工具|参数详解)$/i;
  var REDIRECT_TO = "/lora/master.html";

  function maybeRedirectBlockedPage() {
    var pathname = (window.location && window.location.pathname) ? window.location.pathname : "";
    if (!BLOCKED_PATH_RE.test(pathname)) return false;
    if (pathname !== REDIRECT_TO) {
      window.location.replace(REDIRECT_TO);
    }
    return true;
  }

  function removeBlockedSidebarEntries(root) {
    var scope = root || document;
    var links = scope.querySelectorAll("a.sidebar-item");
    for (var i = 0; i < links.length; i++) {
      var a = links[i];
      var href = (a.getAttribute("href") || "").trim();
      var label = (a.textContent || "").trim();
      if (!BLOCKED_LINK_RE.test(href) && !BLOCKED_LABEL_RE.test(label)) continue;
      var li = a.closest("li");
      if (li && li.parentNode) {
        li.parentNode.removeChild(li);
      }
    }
  }

  function cleanupEmptySidebarChildren() {
    var groups = document.querySelectorAll("ul.sidebar-item-children");
    for (var i = 0; i < groups.length; i++) {
      var ul = groups[i];
      if (ul.querySelector("li")) continue;
      var parentLi = ul.closest("li");
      if (parentLi && parentLi.parentNode) {
        parentLi.parentNode.removeChild(parentLi);
      }
    }
  }

  function tick() {
    if (maybeRedirectBlockedPage()) return;
    removeBlockedSidebarEntries(document);
    cleanupEmptySidebarChildren();
  }

  if (!maybeRedirectBlockedPage()) {
    tick();
    var observer = new MutationObserver(function () { tick(); });
    observer.observe(document.documentElement, { childList: true, subtree: true });
    window.addEventListener("load", tick);
    setInterval(tick, 400);
  }
})();
</script>
"""

_TENSORBOARD_RUNS_DEFAULT_INJECTION = """
<script id="mikazuki-tensorboard-runs-default">
(function () {
  if (window.__MIKAZUKI_TENSORBOARD_RUNS_DEFAULT__) return;
  window.__MIKAZUKI_TENSORBOARD_RUNS_DEFAULT__ = true;

  var TB_PATH_RE = /tensorboard/i;
  var EXCLUDE_LABEL_RE = /^(all|none)$/i;
  var NOISE_LABEL_RE = /(smoothing|ignore outliers|download|tooltip|x-axis|relative|filter|regex|paging|status|sort|direction|ascending|descending|settings|dark mode|light mode|select all|deselect all)/i;
  var DATE_RE = /(20\\d{2})[-_/](\\d{2})[-_/](\\d{2})(?:[ T_:-]?(\\d{2})[:_-]?(\\d{2})[:_-]?(\\d{2}))?/;
  var SUFFIX_RE = /_(\\d{1,6})(?!.*_\\d)/;

  function isTensorboardWrapperPath() {
    var path = (window.location && window.location.pathname) ? String(window.location.pathname) : "";
    return TB_PATH_RE.test(path);
  }

  function getTensorboardIframe() {
    var iframes = document.querySelectorAll("iframe");
    if (!iframes || iframes.length === 0) return null;
    for (var i = 0; i < iframes.length; i++) {
      var f = iframes[i];
      var src = String(f.getAttribute("src") || "");
      if (/tensorboard|6006|\\/proxy\\/tensorboard/i.test(src)) return f;
    }
    return iframes.length === 1 ? iframes[0] : null;
  }

  function parseRunOrderScore(label) {
    if (!label) return Number.NEGATIVE_INFINITY;
    var m = label.match(DATE_RE);
    if (!m) return Number.NEGATIVE_INFINITY;
    var year = parseInt(m[1], 10);
    var month = parseInt(m[2], 10) - 1;
    var day = parseInt(m[3], 10);
    var hh = parseInt(m[4] || "0", 10);
    var mm = parseInt(m[5] || "0", 10);
    var ss = parseInt(m[6] || "0", 10);
    var base = Date.UTC(year, month, day, hh, mm, ss);
    if (!Number.isFinite(base)) return Number.NEGATIVE_INFINITY;
    var suffix = 0;
    var s = label.match(SUFFIX_RE);
    if (s) {
      var parsed = parseInt(s[1], 10);
      suffix = Number.isFinite(parsed) ? parsed : 0;
    }
    // Keep date as primary key and "_n" as tie-breaker.
    return base * 1000000 + suffix;
  }

  function normalizeLabel(text) {
    return String(text || "").replace(/\\s+/g, " ").trim();
  }

  function getRunCandidates(doc) {
    var inputs = doc.querySelectorAll("input[type='checkbox']");
    var candidates = [];
    var seen = new Set();

    for (var i = 0; i < inputs.length; i++) {
      var input = inputs[i];
      if (!input || seen.has(input)) continue;
      seen.add(input);

      var row =
        input.closest("mat-checkbox") ||
        input.closest("paper-checkbox") ||
        input.closest("[role='treeitem']") ||
        input.closest("li") ||
        input.closest("tr") ||
        input.closest("div");
      if (!row) continue;

      var label = normalizeLabel(row.textContent || "");
      if (!label) continue;
      if (EXCLUDE_LABEL_RE.test(label)) continue;
      if (NOISE_LABEL_RE.test(label)) continue;
      if (!DATE_RE.test(label)) continue;

      candidates.push({
        checkbox: input,
        row: row,
        label: label,
        score: parseRunOrderScore(label),
      });
    }
    return candidates;
  }

  function chooseMainGroup(candidates) {
    var groups = new Map();
    for (var i = 0; i < candidates.length; i++) {
      var row = candidates[i].row;
      var parent = row && row.parentElement ? row.parentElement : null;
      if (!parent) continue;
      var count = groups.get(parent) || 0;
      groups.set(parent, count + 1);
    }
    var bestParent = null;
    var bestCount = 0;
    groups.forEach(function (count, parent) {
      if (count > bestCount) {
        bestCount = count;
        bestParent = parent;
      }
    });
    if (!bestParent || bestCount <= 0) return [];
    return candidates.filter(function (c) {
      return c.row && c.row.parentElement === bestParent;
    });
  }

  function setCheckboxChecked(input, wanted) {
    if (!input || input.disabled) return;
    if (!!input.checked === !!wanted) return;
    var clickable =
      input.closest("mat-checkbox") ||
      input.closest("paper-checkbox") ||
      input.closest("label") ||
      input;
    if (clickable && typeof clickable.click === "function") {
      clickable.click();
      return;
    }
    input.checked = !!wanted;
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
  }

  function applyToIframeDocument(iframe) {
    var win = null;
    var doc = null;
    try {
      win = iframe.contentWindow || null;
      doc = iframe.contentDocument || (win ? win.document : null);
    } catch (_e) {
      return false;
    }
    if (!doc || !doc.documentElement) return false;

    if (win && win.__MIKAZUKI_TENSORBOARD_RUNS_APPLIED__) return true;

    var candidates = getRunCandidates(doc);
    if (!candidates || candidates.length === 0) return false;
    var mainGroup = chooseMainGroup(candidates);
    if (!mainGroup || mainGroup.length === 0) mainGroup = candidates;

    mainGroup.sort(function (a, b) {
      if (a.score !== b.score) return b.score - a.score;
      return String(b.label).localeCompare(String(a.label));
    });

    // Reorder to newest -> oldest when rows share the same parent.
    var canReorder = mainGroup.length > 1;
    if (canReorder) {
      var parent = mainGroup[0].row ? mainGroup[0].row.parentElement : null;
      for (var i = 1; i < mainGroup.length; i++) {
        if (!mainGroup[i].row || mainGroup[i].row.parentElement !== parent) {
          canReorder = false;
          break;
        }
      }
      if (canReorder && parent) {
        for (var j = 0; j < mainGroup.length; j++) {
          parent.appendChild(mainGroup[j].row);
        }
      }
    }

    // Default visibility: only latest run checked.
    var latest = mainGroup[0];
    for (var k = 0; k < mainGroup.length; k++) {
      setCheckboxChecked(mainGroup[k].checkbox, mainGroup[k] === latest);
    }

    if (win) {
      win.__MIKAZUKI_TENSORBOARD_RUNS_APPLIED__ = true;
    }
    return true;
  }

  function bindTensorboardIframe(iframe) {
    if (!iframe || iframe.__mikazukiRunsDefaultBound) return;
    iframe.__mikazukiRunsDefaultBound = true;

    var tries = 0;
    var maxTries = 180;
    var timer = null;

    function tryApply() {
      tries += 1;
      var done = applyToIframeDocument(iframe);
      if (done || tries >= maxTries) {
        if (timer) {
          clearInterval(timer);
          timer = null;
        }
      }
    }

    iframe.addEventListener("load", function () {
      tries = 0;
      setTimeout(tryApply, 250);
      setTimeout(tryApply, 1200);
    });

    timer = setInterval(tryApply, 800);
    setTimeout(tryApply, 300);
  }

  function bootstrap() {
    if (!isTensorboardWrapperPath()) return;
    var iframe = getTensorboardIframe();
    if (iframe) bindTensorboardIframe(iframe);
  }

  var observer = new MutationObserver(function () { bootstrap(); });
  observer.observe(document.documentElement, { childList: true, subtree: true });
  window.addEventListener("load", bootstrap);
  setTimeout(bootstrap, 0);
  setInterval(bootstrap, 1500);
})();
</script>
"""

_CTRL_S_SAVE_CONFIG_INJECTION = """
<script id="mikazuki-ctrls-save-config">
(function () {
  if (window.__MIKAZUKI_CTRL_S_SAVE_CONFIG__) return;
  window.__MIKAZUKI_CTRL_S_SAVE_CONFIG__ = true;

  var SAVE_LABEL_RE = /^(保存参数|保存配置|save\\s*params?|save\\s*config)$/i;

  function getSchemaForm() {
    return (
      document.querySelector(".k-form") ||
      document.querySelector(".schema-container form") ||
      document.querySelector("form")
    );
  }

  function normalizeText(text) {
    return String(text || "").replace(/\\s+/g, " ").trim();
  }

  function isElementDisabled(el) {
    if (!el) return true;
    if (el.disabled) return true;
    var ariaDisabled = String(el.getAttribute("aria-disabled") || el.getAttribute("ariadisabled") || "").toLowerCase();
    return ariaDisabled === "true" || ariaDisabled === "1";
  }

  function findSaveConfigButton() {
    var root =
      document.querySelector(".example-container") ||
      document.querySelector("#app") ||
      document.body ||
      document.documentElement;
    if (!root) return null;

    var buttons = root.querySelectorAll("button, [role='button']");
    for (var i = 0; i < buttons.length; i++) {
      var btn = buttons[i];
      if (!btn || isElementDisabled(btn)) continue;
      var label = normalizeText(btn.textContent || btn.innerText || "");
      if (!label) continue;
      if (SAVE_LABEL_RE.test(label)) return btn;
    }
    return null;
  }

  function isSaveHotkey(event) {
    if (!event) return false;
    if (event.isComposing) return false;
    if (event.altKey) return false;
    if (!(event.ctrlKey || event.metaKey)) return false;
    var key = String(event.key || "").toLowerCase();
    return key === "s" || event.code === "KeyS";
  }

  function onKeydown(event) {
    if (!isSaveHotkey(event)) return;

    var btn = findSaveConfigButton();
    if (!btn) return;

    event.preventDefault();
    event.stopPropagation();
    btn.click();
  }

  window.addEventListener("keydown", onKeydown, true);
})();
</script>
"""

_BATCH_SIZE_PROBE_INJECTION = """
<style id="mikazuki-batch-probe-style">
#mikazuki-batch-probe-wrap {
  margin: 6px 0 10px 0;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
#mikazuki-batch-probe-btn {
  width: fit-content;
  min-width: 180px;
  height: 32px;
  padding: 0 12px;
  border: 1px solid #7ea7db;
  border-radius: 6px;
  background: #edf4ff;
  color: #15437a;
  cursor: pointer;
}
#mikazuki-batch-probe-btn[disabled] {
  opacity: 0.7;
  cursor: wait;
}
#mikazuki-batch-probe-status {
  font-size: 12px;
  line-height: 1.4;
  color: #606266;
  white-space: pre-wrap;
  word-break: break-word;
}
#mikazuki-batch-probe-status.error {
  color: #c45656;
}
</style>
<script id="mikazuki-batch-probe">
(function () {
  if (window.__MIKAZUKI_BATCH_PROBE__) return;
  window.__MIKAZUKI_BATCH_PROBE__ = true;

  var state = {
    running: false,
    scheduled: false
  };

  function getSchemaForm() {
    return (
      document.querySelector(".k-form") ||
      document.querySelector(".schema-container form") ||
      document.querySelector("form")
    );
  }

  function getSchemaItems() {
    var form = getSchemaForm();
    if (!form) return [];
    return form.querySelectorAll(".k-schema-item");
  }

  function getItemSearchText(item, title) {
    var titleText = (title && title.textContent ? title.textContent : "").trim();
    var itemText = (item && item.textContent ? item.textContent : "").trim();
    return (titleText + " " + itemText).toLowerCase();
  }

  function findSchemaField(patterns) {
    var items = getSchemaItems();
    if (!patterns || patterns.length === 0) return null;
    for (var i = 0; i < items.length; i++) {
      var item = items[i];
      var title = item.querySelector("h3");
      var searchText = getItemSearchText(item, title);
      for (var j = 0; j < patterns.length; j++) {
        if (patterns[j].test(searchText)) {
          return { item: item, title: title, searchText: searchText };
        }
      }
    }
    return null;
  }

  function findValueElement(item, preferCheckbox) {
    if (!item) return null;
    if (preferCheckbox) {
      var cb = item.querySelector("input[type='checkbox']");
      if (cb) return cb;
    }

    var select = item.querySelector("select");
    if (select) return select;

    var textarea = item.querySelector("textarea");
    if (textarea) return textarea;

    var inputs = item.querySelectorAll("input");
    for (var i = 0; i < inputs.length; i++) {
      var input = inputs[i];
      var type = String(input.getAttribute("type") || "").toLowerCase();
      if (type === "hidden") continue;
      if (type === "checkbox" && !preferCheckbox) continue;
      if (type === "radio" && !input.checked) continue;
      return input;
    }
    return null;
  }

  function dispatchInput(input) {
    if (!input) return;
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
  }

  function parseMaybeNumber(raw) {
    var text = String(raw == null ? "" : raw).trim();
    if (text === "") return "";
    if (!/^-?\\d+(\\.\\d+)?$/.test(text)) return text;
    var n = Number(text);
    return Number.isFinite(n) ? n : text;
  }

  function readFieldValue(item) {
    if (!item) return undefined;
    var checkbox = findValueElement(item, true);
    if (checkbox && String(checkbox.getAttribute("type") || "").toLowerCase() === "checkbox") {
      return !!checkbox.checked;
    }

    var el = findValueElement(item, false);
    if (!el) return undefined;
    var tag = String(el.tagName || "").toLowerCase();
    if (tag === "select" || tag === "textarea") {
      var textValue = String(el.value || "").trim();
      return textValue;
    }

    var type = String(el.getAttribute("type") || "").toLowerCase();
    if (type === "number" || type === "range") {
      return parseMaybeNumber(el.value);
    }
    if (type === "checkbox") return !!el.checked;
    return parseMaybeNumber(el.value);
  }

  function writeNumberField(item, value) {
    var el = findValueElement(item, false);
    if (!el) return false;
    var next = String(value);
    if (String(el.value || "").trim() === next) return false;
    el.value = next;
    dispatchInput(el);
    return true;
  }

  function setStatus(statusEl, message, isError) {
    if (!statusEl) return;
    statusEl.textContent = String(message == null ? "" : message);
    if (isError) statusEl.classList.add("error");
    else statusEl.classList.remove("error");
  }

  function summarizeTrials(trials) {
    if (!Array.isArray(trials) || trials.length === 0) return "";
    var parts = [];
    for (var i = 0; i < trials.length; i++) {
      var t = trials[i] || {};
      parts.push(String(t.batch_size) + ":" + String(t.status || "unknown"));
    }
    return parts.join(", ");
  }

  function setIfFound(config, key, patterns) {
    var field = findSchemaField(patterns);
    if (!field) return;
    var value = readFieldValue(field.item);
    if (value === undefined || value === null) return;
    if (typeof value === "string" && value.trim() === "") return;
    config[key] = value;
  }

  function collectProbeConfig() {
    var cfg = {};
    setIfFound(cfg, "model_train_type", [/model_train_type/i, /训练种类/, /train type/]);
    setIfFound(cfg, "pretrained_model_name_or_path", [/pretrained_model_name_or_path/i, /底模文件路径/]);
    setIfFound(cfg, "train_data_dir", [/train_data_dir/i, /训练数据集路径/]);
    setIfFound(cfg, "reg_data_dir", [/reg_data_dir/i, /正则化数据集路径/]);
    setIfFound(cfg, "resolution", [/(^|[^a-z])resolution([^a-z]|$)/i, /训练图片分辨率/, /训练分辨率/]);

    setIfFound(cfg, "train_batch_size", [/train_batch_size/i, /批量大小/]);
    setIfFound(cfg, "gradient_checkpointing", [/gradient_checkpointing/i, /梯度检查点/]);
    setIfFound(cfg, "gradient_accumulation_steps", [/gradient_accumulation_steps/i, /梯度累加/, /梯度累积/]);

    setIfFound(cfg, "enable_bucket", [/enable_bucket/i, /arb.*桶/, /启用 arb/]);
    setIfFound(cfg, "min_bucket_reso", [/min_bucket_reso/i, /桶最小分辨率/]);
    setIfFound(cfg, "max_bucket_reso", [/max_bucket_reso/i, /桶最大分辨率/]);
    setIfFound(cfg, "bucket_reso_steps", [/bucket_reso_steps/i, /桶分辨率划分单位/]);
    setIfFound(cfg, "bucket_no_upscale", [/bucket_no_upscale/i, /桶不放大图片/]);

    setIfFound(cfg, "network_module", [/network_module/i, /训练网络模块/]);
    setIfFound(cfg, "network_dim", [/network_dim/i, /网络维度/]);
    setIfFound(cfg, "network_alpha", [/network_alpha/i, /网络alpha/]);
    setIfFound(cfg, "network_dropout", [/network_dropout/i, /dropout/]);
    setIfFound(cfg, "network_train_unet_only", [/network_train_unet_only/i, /仅训练\\s*u-?net/]);
    setIfFound(cfg, "network_train_text_encoder_only", [/network_train_text_encoder_only/i, /仅训练文本编码器/]);

    setIfFound(cfg, "optimizer_type", [/optimizer_type/i, /优化器/]);
    setIfFound(cfg, "learning_rate", [/learning_rate/i, /总学习率/]);
    setIfFound(cfg, "unet_lr", [/unet_lr/i, /u-?net\\s*学习率/]);
    setIfFound(cfg, "text_encoder_lr", [/text_encoder_lr/i, /文本编码器学习率/]);

    setIfFound(cfg, "mixed_precision", [/mixed_precision/i, /混合精度/]);
    setIfFound(cfg, "full_fp16", [/full_fp16/i, /完全使用\\s*fp16/]);
    setIfFound(cfg, "full_bf16", [/full_bf16/i, /完全使用\\s*bf16/]);
    setIfFound(cfg, "xformers", [/xformers/i]);
    setIfFound(cfg, "sdpa", [/sdpa/i]);
    setIfFound(cfg, "lowram", [/lowram/i, /低内存模式/]);
    setIfFound(cfg, "cache_latents", [/cache_latents([^_]|$)/i, /缓存图像\\s*latent/]);
    setIfFound(cfg, "cache_latents_to_disk", [/cache_latents_to_disk/i, /缓存图像\\s*latent\\s*到磁盘/]);
    setIfFound(cfg, "cache_text_encoder_outputs", [/cache_text_encoder_outputs([^_]|$)/i, /缓存文本编码器的输出/]);
    setIfFound(cfg, "cache_text_encoder_outputs_to_disk", [/cache_text_encoder_outputs_to_disk/i, /缓存文本编码器的输出到磁盘/]);
    setIfFound(cfg, "persistent_data_loader_workers", [/persistent_data_loader_workers/i, /保留加载训练集的worker/]);
    setIfFound(cfg, "vae", [/^vae\\b/i, /vae 模型文件路径/]);
    setIfFound(cfg, "no_half_vae", [/no_half_vae/i, /不使用半精度\\s*vae/]);

    if (!cfg.model_train_type || String(cfg.model_train_type).trim() === "") {
      cfg.model_train_type = "sdxl-lora";
    }
    return cfg;
  }

  function ensureProbeBlock(batchField, gradField) {
    if (!batchField || !batchField.item || !batchField.item.parentNode) return null;
    var form = getSchemaForm();
    if (!form) return null;

    var wrap = form.querySelector("#mikazuki-batch-probe-wrap");
    if (!wrap) {
      wrap = document.createElement("div");
      wrap.id = "mikazuki-batch-probe-wrap";
      wrap.innerHTML =
        '<button id="mikazuki-batch-probe-btn" type="button">一键检测 Batch Size</button>' +
        '<div id="mikazuki-batch-probe-status">点击后将按当前分辨率和当前训练配置进行短跑探测。</div>';
    }

    if (gradField && gradField.item && gradField.item.parentNode === batchField.item.parentNode) {
      if (gradField.item.previousSibling !== wrap) {
        gradField.item.parentNode.insertBefore(wrap, gradField.item);
      }
    } else if (batchField.item.nextSibling !== wrap) {
      batchField.item.parentNode.insertBefore(wrap, batchField.item.nextSibling);
    }

    return {
      wrap: wrap,
      button: wrap.querySelector("#mikazuki-batch-probe-btn"),
      status: wrap.querySelector("#mikazuki-batch-probe-status")
    };
  }

  async function triggerProbe(button, statusEl) {
    if (!button || state.running) return;
    var cfg = collectProbeConfig();

    if (!cfg.pretrained_model_name_or_path || !cfg.train_data_dir || !cfg.resolution) {
      setStatus(statusEl, "检测失败：请先填写底模路径、训练数据集路径、训练分辨率。", true);
      return;
    }

    state.running = true;
    button.disabled = true;
    setStatus(statusEl, "正在检测，请稍候（会进行真实短训练探测）...", false);
    try {
      var resp = await fetch("/api/probe_batch_size", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(cfg)
      });
      var payload = await resp.json();
      if (!payload || payload.status !== "success") {
        setStatus(statusEl, "检测失败：" + String(payload && payload.message ? payload.message : "未知错误"), true);
        return;
      }

      var data = payload.data || {};
      var recommended = Number(data.recommended_batch_size || 0);
      var maxStable = Number(data.max_stable_batch_size || 0);
      var sharedMemHits = Number(data.shared_memory_hit_count || 0);
      var hasSharedMemHits = !!data.has_shared_memory_hits;
      if (recommended > 0) {
        var currentBatchField = findSchemaField([/train_batch_size/i, /批量大小/]);
        if (currentBatchField && currentBatchField.item) {
          writeNumberField(currentBatchField.item, recommended);
        }
      }
      var est = data.estimated_range || {};
      var estLow = Number(est.low || 0);
      var estHigh = Number(est.high || 0);
      var mem = data.gpu_memory || {};
      var memTotal = Number(mem.total_mib || 0);
      var summary = summarizeTrials(data.trials);
      var msg = "检测完成，推荐 batch_size=" + recommended;
      if (maxStable > 0) {
        msg += "（最大稳定值=" + maxStable + "，已预留安全余量）";
      }
      if (estLow > 0 && estHigh > 0) {
        msg += "；显存预估范围: " + estLow + "~" + estHigh;
      }
      if (memTotal > 0) {
        msg += "；GPU总显存: " + memTotal + "MiB";
      }
      if (hasSharedMemHits || sharedMemHits > 0) {
        msg += "；进程级共享显存命中: " + sharedMemHits + " 次（仅在独显接近满载时触发降档）";
      }
      if (summary) {
        msg += "；测试记录: " + summary;
      }
      setStatus(statusEl, msg, false);
    } catch (e) {
      setStatus(statusEl, "检测失败：网络异常或后端不可用。", true);
    } finally {
      state.running = false;
      button.disabled = false;
    }
  }

  function update() {
    var batchField = findSchemaField([/train_batch_size/i, /批量大小/]);
    if (!batchField) return;
    var gradField = findSchemaField([/gradient_checkpointing/i, /梯度检查点/]);
    var block = ensureProbeBlock(batchField, gradField);
    if (!block || !block.button) return;

    if (!block.button.__mikazukiBound) {
      block.button.__mikazukiBound = true;
      block.button.addEventListener("click", function () {
        triggerProbe(block.button, block.status);
      });
    }
  }

  function scheduleUpdate() {
    if (state.scheduled) return;
    state.scheduled = true;
    setTimeout(function () {
      state.scheduled = false;
      update();
    }, 60);
  }

  var observer = new MutationObserver(function () { scheduleUpdate(); });
  observer.observe(document.documentElement, { childList: true, subtree: true });
  window.addEventListener("load", function () { scheduleUpdate(); });
  setInterval(scheduleUpdate, 1000);
})();
</script>
"""
_patched_frontend_html_cache: dict[str, tuple[int, str]] = {}


def _inject_worker_mode_guard(html_content: str) -> str:
    if 'id="mikazuki-worker-rank-guard"' in html_content:
        return html_content
    if "</body>" in html_content:
        return html_content.replace("</body>", _WORKER_MODE_GUARD_INJECTION + "\n</body>", 1)
    return html_content + _WORKER_MODE_GUARD_INJECTION


def _inject_schema_bootstrap(html_content: str) -> str:
    if 'id="mikazuki-schema-bootstrap"' in html_content:
        return html_content

    module_tag = '<script type="module"'
    if module_tag in html_content:
        return html_content.replace(module_tag, _SCHEMA_BOOTSTRAP_INJECTION + "\n" + module_tag, 1)

    if "</head>" in html_content:
        return html_content.replace("</head>", _SCHEMA_BOOTSTRAP_INJECTION + "\n</head>", 1)

    return _SCHEMA_BOOTSTRAP_INJECTION + "\n" + html_content


def _inject_staged_resolution_preview(html_content: str) -> str:
    if 'id="mikazuki-staged-resolution-preview"' in html_content:
        return html_content
    if "</body>" in html_content:
        return html_content.replace("</body>", _STAGED_RESOLUTION_PREVIEW_INJECTION + "\n</body>", 1)
    return html_content + _STAGED_RESOLUTION_PREVIEW_INJECTION


def _inject_hide_deprecated_lora_docs(html_content: str) -> str:
    if 'id="mikazuki-hide-deprecated-lora-docs"' in html_content:
        return html_content

    module_tag = '<script type="module"'
    if module_tag in html_content:
        return html_content.replace(module_tag, _HIDE_DEPRECATED_LORA_DOCS_INJECTION + "\n" + module_tag, 1)

    if "</head>" in html_content:
        return html_content.replace("</head>", _HIDE_DEPRECATED_LORA_DOCS_INJECTION + "\n</head>", 1)

    return _HIDE_DEPRECATED_LORA_DOCS_INJECTION + "\n" + html_content


def _inject_tensorboard_runs_default(html_content: str) -> str:
    if 'id="mikazuki-tensorboard-runs-default"' in html_content:
        return html_content

    module_tag = '<script type="module"'
    if module_tag in html_content:
        return html_content.replace(module_tag, _TENSORBOARD_RUNS_DEFAULT_INJECTION + "\n" + module_tag, 1)

    if "</head>" in html_content:
        return html_content.replace("</head>", _TENSORBOARD_RUNS_DEFAULT_INJECTION + "\n</head>", 1)

    return _TENSORBOARD_RUNS_DEFAULT_INJECTION + "\n" + html_content


def _inject_ctrl_s_save_config(html_content: str) -> str:
    if 'id="mikazuki-ctrls-save-config"' in html_content:
        return html_content

    module_tag = '<script type="module"'
    if module_tag in html_content:
        return html_content.replace(module_tag, _CTRL_S_SAVE_CONFIG_INJECTION + "\n" + module_tag, 1)

    if "</head>" in html_content:
        return html_content.replace("</head>", _CTRL_S_SAVE_CONFIG_INJECTION + "\n</head>", 1)

    return _CTRL_S_SAVE_CONFIG_INJECTION + "\n" + html_content


def _inject_batch_size_probe(html_content: str) -> str:
    if 'id="mikazuki-batch-probe"' in html_content:
        return html_content

    module_tag = '<script type="module"'
    if module_tag in html_content:
        return html_content.replace(module_tag, _BATCH_SIZE_PROBE_INJECTION + "\n" + module_tag, 1)

    if "</head>" in html_content:
        return html_content.replace("</head>", _BATCH_SIZE_PROBE_INJECTION + "\n</head>", 1)

    return _BATCH_SIZE_PROBE_INJECTION + "\n" + html_content


def _resolve_frontend_html_file(request_path: str) -> Path | None:
    if not FRONTEND_STATIC_DIR:
        return None

    static_root = Path(FRONTEND_STATIC_DIR).resolve()
    raw = (request_path or "").strip().lstrip("/")
    if raw in {"", "."}:
        raw = "index.html"
    if not raw.lower().endswith(".html"):
        return None

    try:
        candidate = (static_root / raw).resolve()
    except OSError:
        return None

    try:
        candidate.relative_to(static_root)
    except ValueError:
        return None

    if not candidate.exists() or not candidate.is_file():
        return None
    return candidate


def _get_patched_frontend_html_content(request_path: str) -> str | None:
    html_file = _resolve_frontend_html_file(request_path)
    if html_file is None:
        return None

    cache_key = str(html_file)
    try:
        mtime_ns = html_file.stat().st_mtime_ns
    except OSError:
        return None

    cached = _patched_frontend_html_cache.get(cache_key)
    if cached and cached[0] == mtime_ns:
        return cached[1]

    try:
        content = html_file.read_text(encoding="utf-8")
    except OSError:
        return None

    content = _inject_schema_bootstrap(content)
    content = _inject_hide_deprecated_lora_docs(content)
    content = _inject_tensorboard_runs_default(content)
    content = _inject_ctrl_s_save_config(content)
    content = _inject_batch_size_probe(content)
    content = _inject_worker_mode_guard(content)
    content = _inject_staged_resolution_preview(content)
    _patched_frontend_html_cache[cache_key] = (mtime_ns, content)
    return content


class SPAStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        patched = _get_patched_frontend_html_content(path)
        if patched is not None:
            return HTMLResponse(content=patched)

        if path in {"", "/", "index.html"}:
            patched = _get_patched_frontend_html_content("index.html")
            if patched is not None:
                return HTMLResponse(content=patched)

        try:
            return await super().get_response(path, scope)
        except HTTPException as ex:
            if ex.status_code == 404:
                patched = _get_patched_frontend_html_content("index.html")
                if patched is not None:
                    return HTMLResponse(content=patched)
                return await super().get_response("index.html", scope)
            else:
                raise ex


def _resolve_lan_ip() -> str | None:
    # Prefer active outbound interface IPv4.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            if ip and not ip.startswith("127."):
                return ip
    except OSError:
        pass

    # Fallback to host-resolved IPv4 addresses.
    try:
        host = socket.gethostname()
        for item in socket.getaddrinfo(host, None, family=socket.AF_INET):
            ip = item[4][0]
            if ip and not ip.startswith("127."):
                return ip
    except OSError:
        pass

    return None


def _is_ipv6_host(host: str) -> bool:
    h = (host or "").strip()
    return h.startswith("[") or ":" in h


def _resolve_browser_host(host: str) -> str:
    if host in ("0.0.0.0", "::", "[::]", "*", ""):
        lan_ip = _resolve_lan_ip()
        if lan_ip:
            return lan_ip
        return "127.0.0.1"
    if _is_ipv6_host(host):
        log.warning(f"IPv6 browser host is disabled, fallback to 127.0.0.1: {host}")
        return "127.0.0.1"
    return host


async def app_startup():
    app_config.load_config()

    await load_schemas()
    await load_presets()
    await asyncio.to_thread(check_torch_gpu)

    if sys.platform == "win32":
        host = _resolve_browser_host(os.environ.get("MIKAZUKI_HOST", "127.0.0.1"))
        port = os.environ.get("MIKAZUKI_PORT", "28000")
        webbrowser.open(f"http://{host}:{port}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await app_startup()
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(proxy_router)


cors_config = os.environ.get("MIKAZUKI_APP_CORS", "")
if cors_config != "":
    if cors_config == "1":
        cors_config = ["http://localhost:8004", "*"]
    else:
        cors_config = cors_config.split(";")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_config,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.middleware("http")
async def add_cache_control_header(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "max-age=0"
    return response

app.include_router(api_router, prefix="/api")
# app.include_router(ipc_router, prefix="/ipc")


@app.get("/")
async def index():
    patched = _get_patched_frontend_html_content("index.html")
    if patched is not None:
        return HTMLResponse(content=patched)
    return PlainTextResponse(
        "Frontend assets are missing (frontend/dist or frontend/index.html). "
        "Run `git clone https://github.com/hanamizuki-ai/lora-gui-dist frontend` "
        "then restart GUI.",
        status_code=503,
    )


@app.get("/favicon.ico", response_class=FileResponse)
async def favicon():
    return FileResponse("assets/favicon.ico")

if FRONTEND_STATIC_DIR and os.path.isdir(FRONTEND_STATIC_DIR):
    app.mount("/", SPAStaticFiles(directory=FRONTEND_STATIC_DIR, html=True), name="static")
else:
    log.warning(
        "frontend static assets not found. GUI static files are unavailable "
        "until frontend assets are restored."
    )
