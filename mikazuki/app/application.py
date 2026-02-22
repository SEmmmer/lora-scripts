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

  function toInt(value) {
    var n = parseInt(value, 10);
    return Number.isFinite(n) ? n : 0;
  }

  function findMachineRankField() {
    var items = document.querySelectorAll(".k-form .k-schema-item");
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
    var form = document.querySelector(".k-form");
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
_patched_index_cache: tuple[int, str] | None = None


def _inject_worker_mode_guard(html_content: str) -> str:
    if 'id="mikazuki-worker-rank-guard"' in html_content:
        return html_content
    if "</body>" in html_content:
        return html_content.replace("</body>", _WORKER_MODE_GUARD_INJECTION + "\n</body>", 1)
    return html_content + _WORKER_MODE_GUARD_INJECTION


def _get_patched_frontend_index_content() -> str | None:
    if FRONTEND_INDEX_FILE is None:
        return None

    index_path = Path(FRONTEND_INDEX_FILE)
    if not index_path.exists() or not index_path.is_file():
        return None

    try:
        mtime_ns = index_path.stat().st_mtime_ns
    except OSError:
        return None

    global _patched_index_cache
    if _patched_index_cache and _patched_index_cache[0] == mtime_ns:
        return _patched_index_cache[1]

    try:
        content = index_path.read_text(encoding="utf-8")
    except OSError:
        return None

    content = _inject_worker_mode_guard(content)
    _patched_index_cache = (mtime_ns, content)
    return content


class SPAStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        if path == "index.html":
            patched = _get_patched_frontend_index_content()
            if patched is not None:
                return HTMLResponse(content=patched)

        try:
            return await super().get_response(path, scope)
        except HTTPException as ex:
            if ex.status_code == 404:
                patched = _get_patched_frontend_index_content()
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

    if sys.platform == "win32" and os.environ.get("MIKAZUKI_DEV", "0") != "1":
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
    patched = _get_patched_frontend_index_content()
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
