const DEFAULT_LANDMARKS = [
  { number: 1, name: "acetabular_outer", label_zh: "髋臼外上缘" },
  { number: 2, name: "triradiate_center", label_zh: "Y 形软骨中心" },
  { number: 3, name: "femoral_head_center", label_zh: "股骨头中心" },
  { number: 4, name: "teardrop_lower", label_zh: "泪滴下缘" },
  { number: 5, name: "femoral_shaft_prox", label_zh: "股骨干轴近端中心" },
  { number: 6, name: "femoral_shaft_dist", label_zh: "股骨干远端中心" },
  { number: 7, name: "femoral_neck_axis_center", label_zh: "股骨颈轴中心" },
  { number: 8, name: "femoral_head_medial", label_zh: "股骨头最内侧缘" },
  { number: 9, name: "femoral_head_lateral", label_zh: "股骨头最外侧缘" },
  { number: 10, name: "obturator_upper", label_zh: "闭孔上缘" },
  { number: 11, name: "femoral_neck_inner_lower", label_zh: "股骨颈内下缘" },
];

const DEFAULT_SETTINGS = {
  dataset_root: "annotation-tool",
  auto_detect: true,
  autosave: true,
  annotator: "default",
};

const App = {
  state: {
    schema: { sides: ["left", "right"], landmarks: DEFAULT_LANDMARKS },
    settings: { ...DEFAULT_SETTINGS },
    image: null,
    annotation: null,
    currentFilename: null,
    transform: { x: 0, y: 0, scale: 1 },
    activeTool: "select",
    activePointKey: "left_acetabular_outer",
    selectedPoint: null,
    selectedConnection: null,
    pendingConnectionStart: null,
    isDragging: false,
    isPanning: false,
    dragStarted: false,
    spaceHeld: false,
    showLabels: true,
    lastMouse: { x: 0, y: 0 },
    contextImagePos: { x: 0, y: 0 },
    history: [],
    historyIndex: -1,
    manifestImages: [],
    progress: null,
    statusFilter: "all",
    importReport: null,
    lastSave: null,
    autosaveTimer: null,
    thumbObserver: null,
    autoDetectPollTimer: null,
    autoDetectStatus: null,
    isNavigating: false,
  },

  init: async () => {
    await App.loadSettings();
    await App.loadSchema();
    App.buildPointList();
    App.bindEvents();
    App.updateSettingsPanel();
    App.resize();
    window.addEventListener("resize", App.resize);
    requestAnimationFrame(App.drawLoop);
    await App.loadManifest();
    App.startAutoDetectPolling();
  },

  loadSettings: async () => {
    try {
      const res = await fetch("/api/annotation/settings");
      if (res.ok) {
        App.state.settings = { ...DEFAULT_SETTINGS, ...(await App.readJsonResponse(res, "settings failed")) };
      }
    } catch (err) {
      console.warn("Settings fallback", err);
    }
  },

  saveSettings: async () => {
    const payload = {
      auto_detect: document.getElementById("autoDetectToggle").checked,
      autosave: document.getElementById("autoSaveToggle").checked,
      annotator: document.getElementById("annotatorInput").value.trim() || "default",
    };
    try {
      const res = await fetch("/api/annotation/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (res.ok) App.state.settings = { ...DEFAULT_SETTINGS, ...(await App.readJsonResponse(res, "settings failed")) };
      App.updateSettingsPanel();
    } catch (err) {
      App.setStatus(`设置保存失败: ${err.message}`);
    }
  },

  updateSettingsPanel: () => {
    document.getElementById("autoDetectToggle").checked = Boolean(App.state.settings.auto_detect);
    document.getElementById("autoSaveToggle").checked = Boolean(App.state.settings.autosave);
    document.getElementById("datasetRootText").textContent = App.state.settings.dataset_root || "annotation-tool";
    App.renderAutoDetectStatus();
    if (!App.state.annotation) {
      document.getElementById("annotatorInput").value = App.state.settings.annotator || "default";
    }
  },

  loadSchema: async () => {
    try {
      const res = await fetch("/api/annotation/schema");
      if (res.ok) App.state.schema = await App.readJsonResponse(res, "schema failed");
    } catch (err) {
      console.warn("Schema fallback", err);
    }
  },

  bindEvents: () => {
    document.getElementById("fileInput").addEventListener("change", App.handleFileSelect);
    document.getElementById("btnFolder").addEventListener("click", () => document.getElementById("folderDialog").showModal());
    document.getElementById("btnPickFolder").addEventListener("click", App.handlePickFolder);
    document.getElementById("btnConfirmFolder").addEventListener("click", App.handleOpenFolder);
    document.getElementById("btnChooseDatasetRoot").addEventListener("click", App.handleChooseDatasetRoot);
    document.getElementById("autoDetectToggle").addEventListener("change", App.saveSettings);
    document.getElementById("autoSaveToggle").addEventListener("change", App.saveSettings);
    document.getElementById("annotatorInput").addEventListener("change", App.saveSettings);
    document.getElementById("btnSave").addEventListener("click", () => App.saveAnnotation());
    document.getElementById("btnAutoDetect").addEventListener("click", App.handleAutoDetectCurrent);
    document.getElementById("btnExportYolo").addEventListener("click", () => {
      window.location.href = "/api/annotation/export-yolo";
    });
    document.getElementById("btnUndo").addEventListener("click", App.undo);
    document.getElementById("btnRedo").addEventListener("click", App.redo);
    document.getElementById("btnFit").addEventListener("click", App.fitToScreen);
    document.getElementById("btnToggleLabels").addEventListener("click", () => {
      App.state.showLabels = !App.state.showLabels;
    });
    document.getElementById("btnDeleteConnection").addEventListener("click", App.deleteSelectedConnection);
    document.querySelectorAll("#progressFilters button").forEach((button) => {
      button.addEventListener("click", () => App.setStatusFilter(button.dataset.statusFilter || "all"));
    });

    document.querySelectorAll(".tool-button").forEach((button) => {
      button.addEventListener("click", () => App.setTool(button.dataset.tool));
    });

    const canvas = document.getElementById("mainCanvas");
    canvas.addEventListener("mousedown", App.handleMouseDown);
    canvas.addEventListener("mousemove", App.handleMouseMove);
    canvas.addEventListener("mouseup", App.handleMouseUp);
    canvas.addEventListener("mouseleave", App.handleMouseUp);
    canvas.addEventListener("wheel", App.handleWheel, { passive: false });
    canvas.addEventListener("contextmenu", App.handleContextMenu);

    document.addEventListener("keydown", App.handleKeyDown);
    document.addEventListener("keyup", App.handleKeyUp);
    document.addEventListener("click", (event) => {
      if (!event.target.closest("#contextMenu")) App.hideContextMenu();
    });
  },

  resize: () => {
    const canvas = document.getElementById("mainCanvas");
    const rect = canvas.parentElement.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.floor(rect.width * ratio));
    canvas.height = Math.max(1, Math.floor(rect.height * ratio));
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${rect.height}px`;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  },

  buildPointList: () => {
    const container = document.getElementById("pointGroups");
    container.innerHTML = "";
    const labels = { left: "图像左侧", right: "图像右侧" };
    App.state.schema.sides.forEach((side) => {
      const group = document.createElement("section");
      group.className = "point-group";
      const title = document.createElement("h3");
      title.textContent = labels[side] || side;
      group.appendChild(title);
      App.state.schema.landmarks.forEach((landmark) => {
        const key = App.keyFor(side, landmark.name);
        const row = document.createElement("div");
        row.className = "point-row missing";
        row.dataset.key = key;
        row.innerHTML = `
          <span class="point-number">#${landmark.number}</span>
          <input type="checkbox" aria-label="visible" />
          <span class="point-name">${landmark.label_zh}</span>
          <span class="source-badge missing">缺失</span>
        `;
        row.addEventListener("click", (event) => {
          if (event.target.tagName !== "INPUT") {
            App.selectPoint(key);
            App.state.activePointKey = key;
          }
        });
        row.querySelector("input").addEventListener("change", (event) => {
          event.stopPropagation();
          App.togglePointVisibility(key, event.target.checked);
        });
        group.appendChild(row);
      });
      container.appendChild(group);
    });
  },

  keyFor: (side, name) => `${side}_${name}`,

  setTool: (tool) => {
    App.state.activeTool = tool;
    if (tool !== "line") App.state.pendingConnectionStart = null;
    document.querySelectorAll(".tool-button").forEach((button) => {
      button.classList.toggle("active", button.dataset.tool === tool);
    });
    document.getElementById("mainCanvas").style.cursor = tool === "point" || tool === "line" ? "crosshair" : "default";
    App.updateSelectedBox();
  },

  setStatus: (message) => {
    document.getElementById("statusText").textContent = message;
  },

  readJsonResponse: async (res, fallbackMessage = "请求失败") => {
    const text = await res.text();
    let payload = {};
    if (text) {
      try {
        payload = JSON.parse(text);
      } catch (err) {
        const snippet = text.slice(0, 220).trim();
        throw new Error(snippet || fallbackMessage);
      }
    }
    if (!res.ok) {
      throw new Error(payload.detail || payload.message || fallbackMessage);
    }
    return payload;
  },

  handlePickFolder: async () => {
    const error = document.getElementById("folderError");
    error.textContent = "";
    try {
      const res = await fetch("/api/annotation/select-folder", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ purpose: "import" }),
      });
      const data = await App.readJsonResponse(res, "folder picker failed");
      document.getElementById("folderPathInput").value = data.path || "";
    } catch (err) {
      error.textContent = `${err.message}；可手动填写路径。`;
    }
  },

  handleChooseDatasetRoot: async () => {
    App.setStatus("请选择保存目录...");
    try {
      const res = await fetch("/api/annotation/select-folder", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ purpose: "dataset" }),
      });
      const data = await App.readJsonResponse(res, "folder picker failed");
      App.state.settings = { ...DEFAULT_SETTINGS, ...(data.settings || {}) };
      App.updateSettingsPanel();
      App.clearWorkspaceView("工作区已切换");
      await App.loadManifest();
      App.startAutoDetectPolling();
      App.setStatus("保存目录已更新");
    } catch (err) {
      App.setStatus(`选择保存目录失败: ${err.message}`);
    }
  },

  handleFileSelect: async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    await App.saveSettings();
    const form = new FormData();
    form.append("file", file);
    App.setStatus("正在载入...");
    try {
      const res = await fetch("/api/annotation/load", { method: "POST", body: form });
      App.applyAnnotation(await App.readJsonResponse(res, "load failed"));
      await App.loadManifest();
    } catch (err) {
      App.setStatus(`载入失败: ${err.message}`);
    } finally {
      event.target.value = "";
    }
  },

  handleOpenFolder: async () => {
    await App.saveSettings();
    const dialog = document.getElementById("folderDialog");
    const path = document.getElementById("folderPathInput").value.trim();
    const error = document.getElementById("folderError");
    error.textContent = "";
    try {
      const res = await fetch("/api/annotation/open-folder", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ folder_path: path, split: "train" }),
      });
      const data = await App.readJsonResponse(res, "import failed");
      App.state.settings = { ...DEFAULT_SETTINGS, ...(data.settings || {}) };
      App.state.autoDetectStatus = data.auto_detect || null;
      App.state.importReport = data.import_report || null;
      App.clearWorkspaceView("文件夹已导入，自动识别将在后台进行");
      App.state.importReport = data.import_report || null;
      dialog.close();
      await App.loadManifest();
      App.startAutoDetectPolling();
      App.renderImportReport();
      App.setStatus(`文件夹已导入：${data.total || 0} 张，后台识别 ${data.queued || 0} 张`);
    } catch (err) {
      error.textContent = err.message;
    }
  },

  handleAutoDetectCurrent: async () => {
    if (!App.state.currentFilename || !App.state.annotation) {
      App.setStatus("请先打开一张图片");
      return;
    }
    const button = document.getElementById("btnAutoDetect");
    button.disabled = true;
    const saved = await App.saveAnnotation({ silent: true, skipManifest: true });
    if (saved === false) {
      button.disabled = false;
      return;
    }
    App.setStatus("正在重新自动识别当前图片...");
    try {
      const res = await fetch("/api/annotation/auto-detect-image", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          filename: App.state.currentFilename,
          preserve_manual: true,
          include_partial: true,
        }),
      });
      const data = await App.readJsonResponse(res, "auto detect image failed");
      const info = data.auto_detect || {};
      App.applyAnnotation(data);
      App.state.lastSave = {
        filename: data.image.filename,
        timeText: new Date().toLocaleTimeString(),
        annotationPath: info.annotation_path || "annotations",
        labelPath: info.label_path || "同名 txt",
      };
      await App.loadManifest();
      App.renderSaveInfo();
      if (info.applied && info.visible_count > 0) {
        App.setStatus(`自动识别完成：识别到 ${info.visible_count} 个点，已保留人工修改点`);
      } else {
        App.setStatus("自动识别仍未找到可用点，请手工标注或把图片发给项目团队排查");
      }
    } catch (err) {
      App.setStatus(`自动识别失败: ${err.message}`);
    } finally {
      button.disabled = false;
    }
  },

  loadManifest: async () => {
    try {
      const res = await fetch("/api/annotation/list");
      if (!res.ok) throw new Error(await res.text());
      const data = await App.readJsonResponse(res, "manifest failed");
      App.state.manifestImages = data.images || [];
      App.state.progress = data.progress || null;
      App.state.autoDetectStatus = data.auto_detect || App.state.autoDetectStatus;
      if (data.settings) {
        App.state.settings = { ...DEFAULT_SETTINGS, ...data.settings };
        App.updateSettingsPanel();
      }
      App.renderThumbnails();
      App.renderAutoDetectStatus();
      App.renderProgressSummary();
    } catch (err) {
      console.warn(err);
    }
  },

  setStatusFilter: (filter) => {
    App.state.statusFilter = filter || "all";
    document.querySelectorAll("#progressFilters button").forEach((button) => {
      button.classList.toggle("active", button.dataset.statusFilter === App.state.statusFilter);
    });
    App.renderThumbnails();
  },

  progressCounts: () => {
    const counts = { total: 0, pending: 0, auto: 0, in_progress: 0, done: 0, needs_review: 0 };
    (App.state.manifestImages || []).forEach((item) => {
      const status = ["pending", "auto", "in_progress", "done"].includes(item.status) ? item.status : "pending";
      counts[status] += 1;
      counts.total += 1;
    });
    counts.needs_review = counts.pending + counts.auto + counts.in_progress;
    return counts;
  },

  renderProgressSummary: () => {
    const counts = App.state.progress?.counts || App.progressCounts();
    const summary = document.getElementById("progressSummary");
    if (!summary) return;
    if (!counts.total) {
      summary.textContent = "未导入文件夹";
    } else {
      summary.innerHTML = `<strong>${counts.done}</strong> / ${counts.total} 已完成；${counts.needs_review} 张仍需处理`;
    }
    const report = App.state.progress?.report_files;
    const reportText = document.getElementById("progressReportText");
    if (reportText && report) {
      reportText.textContent = `已生成 ${report.marker}、${report.html} 和 ${report.csv}`;
    }
    document.querySelectorAll("#progressFilters button").forEach((button) => {
      const filter = button.dataset.statusFilter || "all";
      const count = filter === "all" ? counts.total : counts[filter] || 0;
      button.textContent = `${App.filterLabel(filter)} ${count}`;
      button.classList.toggle("active", filter === App.state.statusFilter);
    });
    App.renderSubmissionCheck(counts);
    App.renderSaveInfo();
    App.renderImportReport();
  },

  renderSubmissionCheck: (counts = null) => {
    const target = document.getElementById("submissionCheck");
    if (!target) return;
    const nextCounts = counts || App.state.progress?.counts || App.progressCounts();
    target.classList.remove("ready", "warning");
    if (!nextCounts.total) {
      target.textContent = "导入文件夹后会显示提交检查";
      return;
    }
    if (nextCounts.needs_review === 0) {
      target.classList.add("ready");
      target.textContent = "提交检查通过：全部图片已完成。保存后将整个文件夹直接发送给项目团队。";
      return;
    }
    target.classList.add("warning");
    target.textContent = `提交前仍需处理 ${nextCounts.needs_review} 张：未标注 ${nextCounts.pending}，待复核 ${nextCounts.auto}，修改中 ${nextCounts.in_progress}。`;
  },

  renderSaveInfo: () => {
    const target = document.getElementById("saveInfoText");
    if (!target) return;
    if (!App.state.currentFilename) {
      target.textContent = "尚未打开当前图片";
      return;
    }
    if (!App.state.lastSave || App.state.lastSave.filename !== App.state.currentFilename) {
      target.textContent = "当前图片修改后请保存；保存会写入 JSON 和同名 YOLO txt";
      return;
    }
    target.textContent = `最近保存 ${App.state.lastSave.timeText}；JSON：${App.state.lastSave.annotationPath}；TXT：${App.state.lastSave.labelPath}`;
  },

  renderImportReport: () => {
    const target = document.getElementById("importReportText");
    if (!target) return;
    const report = App.state.importReport;
    if (!report?.warnings?.length) {
      target.textContent = "命名混乱或目录嵌套时，可先发给项目团队整理";
      return;
    }
    target.textContent = `导入提示：${report.warnings.slice(0, 3).join("；")}`;
  },

  renderThumbnails: () => {
    const strip = document.getElementById("thumbStrip");
    strip.innerHTML = "";
    if (App.state.thumbObserver) {
      App.state.thumbObserver.disconnect();
      App.state.thumbObserver = null;
    }
    const supportsLazyThumbs = "IntersectionObserver" in window;
    if (supportsLazyThumbs) {
      App.state.thumbObserver = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            if (!entry.isIntersecting) return;
            App.loadThumbnail(entry.target);
            App.state.thumbObserver.unobserve(entry.target);
          });
        },
        { root: strip, rootMargin: "360px" },
      );
    }
    const images = App.filteredManifestImages();
    if (!images.length) {
      const empty = document.createElement("div");
      empty.className = "thumb-empty";
      empty.textContent = App.state.manifestImages.length ? "当前筛选没有图片" : "导入文件夹后会显示图片状态";
      strip.appendChild(empty);
      return;
    }
    images.forEach((item, index) => {
      const filename = item.image_path.split("/").pop();
      const thumb = document.createElement("button");
      thumb.type = "button";
      thumb.className = `thumb loading ${filename === App.state.currentFilename ? "active" : ""}`;
      thumb.dataset.filename = filename;
      thumb.title = `${filename} · ${App.statusLabel(item.status)}`;
      thumb.innerHTML = `
        <span class="thumb-status ${item.status || "pending"}"></span>
        <span class="thumb-label">${App.statusShortLabel(item.status)}</span>
      `;
      thumb.addEventListener("click", () => App.loadByName(filename));
      strip.appendChild(thumb);
      if (App.state.thumbObserver) {
        App.state.thumbObserver.observe(thumb);
      } else if (index < 80) {
        App.loadThumbnail(thumb);
      }
    });
  },

  filteredManifestImages: () => {
    const filter = App.state.statusFilter || "all";
    if (filter === "all") return App.state.manifestImages || [];
    return (App.state.manifestImages || []).filter((item) => (item.status || "pending") === filter);
  },

  loadThumbnail: (thumb) => {
    if (!thumb || thumb.dataset.loaded === "1") return;
    const filename = thumb.dataset.filename;
    thumb.style.backgroundImage = `url(/api/annotation/image/${encodeURIComponent(filename)})`;
    thumb.dataset.loaded = "1";
    thumb.classList.remove("loading");
  },

  statusLabel: (status) => {
    if (status === "auto") return "自动初标待复核";
    if (status === "in_progress") return "人工修改中";
    if (status === "done") return "已完成";
    if (status === "queued") return "排队中";
    return "未标注";
  },

  statusShortLabel: (status) => {
    if (status === "auto") return "待复核";
    if (status === "in_progress") return "修改中";
    if (status === "done") return "完成";
    return "未标注";
  },

  filterLabel: (filter) => {
    if (filter === "all") return "全部";
    if (filter === "auto") return "待复核";
    if (filter === "in_progress") return "修改中";
    if (filter === "done") return "已完成";
    return "未标注";
  },

  loadByName: async (filename, options = {}) => {
    if (App.state.annotation && App.state.currentFilename && App.state.currentFilename !== filename) {
      const saved = await App.saveAnnotation({ silent: true, skipManifest: true });
      if (saved === false) return;
    }
    App.setStatus("正在打开缩略图...");
    try {
      const res = await fetch(`/api/annotation/load-by-name?filename=${encodeURIComponent(filename)}`);
      App.applyAnnotation(await App.readJsonResponse(res, "load failed"));
      if (!options.skipManifest) await App.loadManifest();
      App.scrollActiveThumbnailIntoView();
    } catch (err) {
      App.setStatus(`打开失败: ${err.message}`);
    }
  },

  imageFilename: (item) => item?.image_path?.split("/").pop(),

  currentManifestIndex: () =>
    App.state.manifestImages.findIndex((item) => App.imageFilename(item) === App.state.currentFilename),

  loadAdjacentImage: async (delta) => {
    if (App.state.isNavigating) return;
    const images = App.state.manifestImages || [];
    if (!images.length) {
      App.setStatus("当前工作区没有图片");
      return;
    }
    let index = App.currentManifestIndex();
    if (index < 0) {
      index = delta > 0 ? -1 : images.length;
    }
    const nextIndex = index + delta;
    if (nextIndex < 0) {
      App.setStatus("已经是第一张");
      return;
    }
    if (nextIndex >= images.length) {
      App.setStatus("已经是最后一张");
      return;
    }
    const filename = App.imageFilename(images[nextIndex]);
    if (!filename) return;
    App.state.isNavigating = true;
    try {
      await App.loadByName(filename);
    } finally {
      App.state.isNavigating = false;
    }
  },

  scrollActiveThumbnailIntoView: () => {
    const active = document.querySelector(".thumb.active");
    if (!active) return;
    active.scrollIntoView({ block: "nearest", inline: "center" });
  },

  clearWorkspaceView: (message = "未打开图像") => {
    App.state.image = null;
    App.state.annotation = null;
    App.state.currentFilename = null;
    App.state.selectedPoint = null;
    App.state.selectedConnection = null;
    App.state.pendingConnectionStart = null;
    App.state.manifestImages = [];
    App.state.progress = null;
    App.state.importReport = null;
    App.state.lastSave = null;
    App.state.history = [];
    App.state.historyIndex = -1;
    App.state.transform = { x: 0, y: 0, scale: 1 };
    document.getElementById("imageTitle").textContent = "未打开图像";
    document.getElementById("warningList").innerHTML = "";
    document.getElementById("connectionList").innerHTML = "";
    document.getElementById("thumbStrip").innerHTML = "";
    App.renderProgressSummary();
    document.getElementById("selectedConnectionLabel").textContent = "未选中连线";
    document.querySelectorAll(".point-row").forEach((row) => {
      row.classList.add("missing");
      row.classList.remove("selected");
      const checkbox = row.querySelector("input");
      const badge = row.querySelector(".source-badge");
      if (checkbox) checkbox.checked = false;
      if (badge) {
        badge.className = "source-badge missing";
        badge.textContent = "缺失";
      }
    });
    App.updateSummary();
    App.updateSelectedBox();
    App.setStatus(message);
  },

  startAutoDetectPolling: () => {
    clearInterval(App.state.autoDetectPollTimer);
    App.pollAutoDetectStatus();
    App.state.autoDetectPollTimer = setInterval(App.pollAutoDetectStatus, 1600);
  },

  pollAutoDetectStatus: async () => {
    try {
      const res = await fetch("/api/annotation/auto-detect/status");
      if (!res.ok) throw new Error(await res.text());
      const next = await App.readJsonResponse(res, "auto detect status failed");
      const previous = JSON.stringify(App.state.autoDetectStatus || {});
      App.state.autoDetectStatus = next;
      App.renderAutoDetectStatus();
      const changed = previous !== JSON.stringify(next || {});
      if (changed) {
        await App.loadManifest();
        App.maybeRefreshActiveFromBackground();
      }
      if (!next.running && !next.pending && !next.processing) {
        clearInterval(App.state.autoDetectPollTimer);
        App.state.autoDetectPollTimer = null;
      }
    } catch (err) {
      console.warn(err);
    }
  },

  renderAutoDetectStatus: () => {
    const target = document.getElementById("autoDetectQueueText");
    if (!target) return;
    const status = App.state.autoDetectStatus;
    if (!status || !status.total) {
      target.textContent = "未排队";
      return;
    }
    const finished = (status.done || 0) + (status.skipped || 0) + (status.failed || 0);
    const processing = status.processing ? `，正在处理 ${status.processing}` : "";
    target.textContent = `${finished}/${status.total} 完成，待处理 ${status.pending || 0}${processing}`;
  },

  maybeRefreshActiveFromBackground: () => {
    if (!App.state.currentFilename || !App.state.annotation) return;
    const source = App.state.annotation.auto_initialization?.source;
    if (source !== "queued") return;
    const manualVisible = Object.values(App.state.annotation.keypoints || {}).some(
      (point) => App.pointIsVisible(point) && point.source === "manual",
    );
    if (manualVisible) return;
    const item = App.state.manifestImages.find((entry) => entry.image_path.split("/").pop() === App.state.currentFilename);
    if (!item || item.status === "pending") return;
    App.loadByName(App.state.currentFilename, { skipManifest: true });
  },

  applyAnnotation: (annotation) => {
    annotation.connections = Array.isArray(annotation.connections) ? annotation.connections : [];
    App.state.annotation = annotation;
    App.state.currentFilename = annotation.image.filename;
    App.state.selectedPoint = null;
    App.state.selectedConnection = null;
    App.state.pendingConnectionStart = null;
    App.state.activePointKey = "left_acetabular_outer";
    App.state.lastSave = null;
    document.getElementById("annotatorInput").value = annotation.annotator?.user_id || App.state.settings.annotator || "default";
    document.getElementById("imageTitle").textContent = annotation.image.filename;
    App.loadImage(annotation.image_url || `/api/annotation/image/${encodeURIComponent(annotation.image.filename)}`);
    App.resetHistory();
    App.refreshAll();
    App.renderWarnings();
    App.renderSaveInfo();
  },

  loadImage: (src) => {
    const image = new Image();
    image.onload = () => {
      App.state.image = image;
      App.fitToScreen();
      App.resetHistory();
      App.setStatus("已载入");
    };
    image.src = src;
  },

  saveAnnotation: async (options = {}) => {
    if (!App.state.annotation) return true;
    App.state.annotation.annotator = App.state.annotation.annotator || {};
    App.state.annotation.annotator.user_id = document.getElementById("annotatorInput").value.trim() || "default";
    App.state.settings.annotator = App.state.annotation.annotator.user_id;
    if (App.state.image) {
      App.state.annotation.image.width = App.state.image.naturalWidth || App.state.image.width;
      App.state.annotation.image.height = App.state.image.naturalHeight || App.state.image.height;
    }
    if (!options.silent) App.setStatus("正在保存...");
    try {
      const res = await fetch("/api/annotation/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(App.state.annotation),
      });
      const payload = await App.readJsonResponse(res, "save failed");
      App.state.lastSave = {
        filename: App.state.annotation.image.filename,
        timeText: new Date().toLocaleTimeString(),
        annotationPath: payload.annotation_path || "annotations",
        labelPath: payload.label_path || "同名 txt",
      };
      App.setStatus(options.silent ? "已自动保存" : "已保存");
      if (!options.skipManifest) await App.loadManifest();
      App.renderSaveInfo();
      return true;
    } catch (err) {
      App.setStatus(`保存失败: ${err.message}`);
      return false;
    }
  },

  scheduleAutosave: () => {
    if (!App.state.settings.autosave || !App.state.annotation) return;
    clearTimeout(App.state.autosaveTimer);
    App.state.autosaveTimer = setTimeout(() => App.saveAnnotation({ silent: true }), 900);
  },

  resetHistory: () => {
    if (!App.state.annotation) {
      App.state.history = [];
      App.state.historyIndex = -1;
      return;
    }
    App.state.history = [JSON.stringify(App.state.annotation)];
    App.state.historyIndex = 0;
  },

  pushHistory: () => {
    if (!App.state.annotation) return;
    const snapshot = JSON.stringify(App.state.annotation);
    if (App.state.history[App.state.historyIndex] === snapshot) return;
    App.state.history = App.state.history.slice(0, App.state.historyIndex + 1);
    App.state.history.push(snapshot);
    App.state.historyIndex = App.state.history.length - 1;
  },

  restoreHistory: () => {
    App.state.annotation = JSON.parse(App.state.history[App.state.historyIndex]);
    App.refreshAll();
  },

  undo: () => {
    if (App.state.historyIndex <= 0) return;
    App.state.historyIndex -= 1;
    App.restoreHistory();
    App.scheduleAutosave();
  },

  redo: () => {
    if (App.state.historyIndex >= App.state.history.length - 1) return;
    App.state.historyIndex += 1;
    App.restoreHistory();
    App.scheduleAutosave();
  },

  refreshAll: () => {
    App.refreshPointList();
    App.renderConnectionList();
    App.updateSummary();
    App.updateSelectedBox();
  },

  selectPoint: (key) => {
    App.state.selectedPoint = key;
    App.state.selectedConnection = null;
    document.querySelectorAll(".point-row").forEach((row) => row.classList.toggle("selected", row.dataset.key === key));
    App.renderConnectionList();
    App.updateSelectedBox();
  },

  selectConnection: (id) => {
    App.state.selectedConnection = id;
    App.state.selectedPoint = null;
    document.querySelectorAll(".point-row").forEach((row) => row.classList.remove("selected"));
    App.renderConnectionList();
    App.updateSelectedBox();
  },

  pointIsVisible: (point) => Boolean(point && point.visible && point.visibility > 0 && point.x !== null && point.y !== null),

  togglePointVisibility: (key, visible) => {
    if (!App.state.annotation?.keypoints?.[key]) return;
    App.pushHistory();
    const point = App.state.annotation.keypoints[key];
    if (visible && (point.x === null || point.y === null)) {
      const center = App.imageCenter();
      point.x = center.x;
      point.y = center.y;
    }
    point.visible = visible;
    point.visibility = visible ? 2 : 0;
    point.source = "manual";
    point.confidence = visible ? 1 : 0;
    point.updated_at = new Date().toISOString();
    App.refreshAll();
    App.pushHistory();
    App.scheduleAutosave();
  },

  imageCenter: () => {
    if (!App.state.image) return { x: 0, y: 0 };
    return { x: App.state.image.width / 2, y: App.state.image.height / 2 };
  },

  placePoint: (key, imagePos) => {
    if (!App.state.annotation?.keypoints?.[key]) return;
    App.pushHistory();
    const point = App.state.annotation.keypoints[key];
    point.x = App.clamp(imagePos.x, 0, App.state.image.width);
    point.y = App.clamp(imagePos.y, 0, App.state.image.height);
    point.visible = true;
    point.visibility = 2;
    point.source = "manual";
    point.confidence = 1;
    point.updated_at = new Date().toISOString();
    point.annotator = document.getElementById("annotatorInput").value.trim() || "default";
    App.selectPoint(key);
    App.refreshAll();
    App.pushHistory();
    App.scheduleAutosave();
  },

  markMissing: (key) => {
    if (!App.state.annotation?.keypoints?.[key]) return;
    App.pushHistory();
    const point = App.state.annotation.keypoints[key];
    point.x = null;
    point.y = null;
    point.visible = false;
    point.visibility = 0;
    point.source = "manual";
    point.confidence = 0;
    point.updated_at = new Date().toISOString();
    App.refreshAll();
    App.pushHistory();
    App.scheduleAutosave();
  },

  addConnection: (pointA, pointB) => {
    if (!pointA || !pointB || pointA === pointB) return;
    App.pushHistory();
    const existing = App.state.annotation.connections.find(
      (item) =>
        (item.point_a === pointA && item.point_b === pointB) ||
        (item.point_a === pointB && item.point_b === pointA),
    );
    if (existing) {
      existing.visible = true;
      existing.updated_at = new Date().toISOString();
      App.selectConnection(existing.id);
    } else {
      const id = `manual_${pointA}__${pointB}_${Date.now()}`;
      App.state.annotation.connections.push({
        id,
        point_a: pointA,
        point_b: pointB,
        label: "人工连线",
        source: "manual",
        visible: true,
        updated_at: new Date().toISOString(),
        annotator: document.getElementById("annotatorInput").value.trim() || "default",
      });
      App.selectConnection(id);
    }
    App.state.pendingConnectionStart = null;
    App.refreshAll();
    App.pushHistory();
    App.scheduleAutosave();
  },

  deleteSelectedConnection: () => {
    if (!App.state.annotation || !App.state.selectedConnection) return;
    App.pushHistory();
    const index = App.state.annotation.connections.findIndex((item) => item.id === App.state.selectedConnection);
    if (index < 0) return;
    const connection = App.state.annotation.connections[index];
    if (connection.source === "default") {
      connection.visible = false;
      connection.updated_at = new Date().toISOString();
    } else {
      App.state.annotation.connections.splice(index, 1);
    }
    App.state.selectedConnection = null;
    App.refreshAll();
    App.pushHistory();
    App.scheduleAutosave();
  },

  handleMouseDown: (event) => {
    if (event.button === 2 || !App.state.annotation || !App.state.image) return;
    const pos = App.mousePos(event);
    const imagePos = App.toImageCoords(pos.x, pos.y);
    App.hideContextMenu();
    if (event.button === 1 || App.state.spaceHeld) {
      App.state.isPanning = true;
      App.state.lastMouse = pos;
      return;
    }
    if (App.state.activeTool === "point") {
      App.placePoint(App.state.activePointKey, imagePos);
      return;
    }
    const hitPoint = App.hitTestPoint(pos.x, pos.y);
    if (App.state.activeTool === "line") {
      if (!hitPoint) {
        App.setStatus("连线工具需要点击两个已显示的点");
        return;
      }
      if (!App.state.pendingConnectionStart) {
        App.state.pendingConnectionStart = hitPoint;
        App.selectPoint(hitPoint);
        App.setStatus("请选择第二个点完成连线");
      } else {
        App.addConnection(App.state.pendingConnectionStart, hitPoint);
        App.setStatus("连线已添加");
      }
      return;
    }
    if (hitPoint) {
      App.selectPoint(hitPoint);
      App.state.activePointKey = hitPoint;
      App.state.isDragging = true;
      App.state.dragStarted = false;
      App.state.lastMouse = pos;
      return;
    }
    const hitConnection = App.hitTestConnection(pos.x, pos.y);
    if (hitConnection) {
      App.selectConnection(hitConnection);
      return;
    }
    App.state.isPanning = true;
    App.state.lastMouse = pos;
  },

  handleMouseMove: (event) => {
    const pos = App.mousePos(event);
    if (App.state.isDragging && App.state.selectedPoint) {
      const point = App.state.annotation.keypoints[App.state.selectedPoint];
      if (!App.state.dragStarted) {
        App.pushHistory();
        App.state.dragStarted = true;
      }
      point.x = App.clamp((point.x || 0) + (pos.x - App.state.lastMouse.x) / App.state.transform.scale, 0, App.state.image.width);
      point.y = App.clamp((point.y || 0) + (pos.y - App.state.lastMouse.y) / App.state.transform.scale, 0, App.state.image.height);
      point.visible = true;
      point.visibility = 2;
      point.source = "manual";
      point.confidence = 1;
      point.updated_at = new Date().toISOString();
      App.state.lastMouse = pos;
      App.refreshPointList();
      App.updateSummary();
    } else if (App.state.isPanning) {
      App.state.transform.x += pos.x - App.state.lastMouse.x;
      App.state.transform.y += pos.y - App.state.lastMouse.y;
      App.state.lastMouse = pos;
    }
  },

  handleMouseUp: () => {
    if (App.state.isDragging && App.state.dragStarted) {
      App.pushHistory();
      App.scheduleAutosave();
    }
    App.state.isDragging = false;
    App.state.isPanning = false;
    App.state.dragStarted = false;
  },

  handleWheel: (event) => {
    if (!App.state.image) return;
    event.preventDefault();
    const pos = App.mousePos(event);
    const oldScale = App.state.transform.scale;
    const factor = event.deltaY > 0 ? 0.9 : 1.1;
    const newScale = App.clamp(oldScale * factor, 0.05, 12);
    App.state.transform.x = pos.x - (pos.x - App.state.transform.x) * (newScale / oldScale);
    App.state.transform.y = pos.y - (pos.y - App.state.transform.y) * (newScale / oldScale);
    App.state.transform.scale = newScale;
  },

  handleContextMenu: (event) => {
    event.preventDefault();
    if (!App.state.image || !App.state.annotation) return;
    const pos = App.mousePos(event);
    App.state.contextImagePos = App.toImageCoords(pos.x, pos.y);
    const side = App.state.contextImagePos.x > App.state.image.width / 2 ? "left" : "right";
    App.showContextMenu(pos.x, pos.y, side);
  },

  showContextMenu: (x, y, side) => {
    const menu = document.getElementById("contextMenu");
    const items = document.getElementById("contextItems");
    const sideText = side === "left" ? "图像左侧" : "图像右侧";
    document.getElementById("contextTitle").textContent = sideText;
    items.innerHTML = "";
    App.state.schema.landmarks.forEach((landmark) => {
      const key = App.keyFor(side, landmark.name);
      const point = App.state.annotation.keypoints[key];
      const item = document.createElement("div");
      item.className = "context-item";
      item.innerHTML = `<span>#${landmark.number} ${landmark.label_zh}</span><small>${App.pointIsVisible(point) ? "已标" : ""}</small>`;
      item.addEventListener("click", (event) => {
        event.stopPropagation();
        App.placePoint(key, App.state.contextImagePos);
        App.hideContextMenu();
      });
      items.appendChild(item);
    });
    const wrap = document.getElementById("canvasWrap").getBoundingClientRect();
    menu.style.left = `${App.clamp(x, 8, wrap.width - 230)}px`;
    menu.style.top = `${App.clamp(y, 8, wrap.height - 360)}px`;
    menu.style.display = "block";
  },

  hideContextMenu: () => {
    document.getElementById("contextMenu").style.display = "none";
  },

  handleKeyDown: (event) => {
    const tag = event.target.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || event.target.isContentEditable) return;
    if (!event.ctrlKey && !event.metaKey && !event.altKey && event.key === "ArrowLeft") {
      event.preventDefault();
      App.loadAdjacentImage(-1);
      return;
    }
    if (!event.ctrlKey && !event.metaKey && !event.altKey && event.key === "ArrowRight") {
      event.preventDefault();
      App.loadAdjacentImage(1);
      return;
    }
    if (event.code === "Space") {
      event.preventDefault();
      App.state.spaceHeld = true;
      document.getElementById("mainCanvas").style.cursor = "grab";
    }
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
      event.preventDefault();
      App.saveAnnotation();
    }
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
      event.preventDefault();
      App.undo();
    }
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "y") {
      event.preventDefault();
      App.redo();
    }
    if (event.key === "Delete") {
      if (App.state.selectedConnection) App.deleteSelectedConnection();
      else if (App.state.selectedPoint) App.markMissing(App.state.selectedPoint);
    }
    if (event.key === "Escape") {
      App.state.pendingConnectionStart = null;
      App.setStatus("已取消当前连线");
    }
    if (event.key === "Tab") {
      event.preventDefault();
      App.selectNextPoint();
    }
    if (event.key.toLowerCase() === "v") App.setTool("select");
    if (event.key.toLowerCase() === "p") App.setTool("point");
    if (event.key.toLowerCase() === "l") App.setTool("line");
    if (event.key.toLowerCase() === "h") App.state.showLabels = !App.state.showLabels;
  },

  handleKeyUp: (event) => {
    if (event.code === "Space") {
      App.state.spaceHeld = false;
      document.getElementById("mainCanvas").style.cursor =
        App.state.activeTool === "point" || App.state.activeTool === "line" ? "crosshair" : "default";
    }
  },

  selectNextPoint: () => {
    const keys = App.allKeys();
    if (!keys.length) return;
    const index = keys.indexOf(App.state.selectedPoint);
    const next = keys[(index + 1 + keys.length) % keys.length];
    App.selectPoint(next);
    App.state.activePointKey = next;
  },

  allKeys: () => {
    const keys = [];
    App.state.schema.sides.forEach((side) => {
      App.state.schema.landmarks.forEach((landmark) => keys.push(App.keyFor(side, landmark.name)));
    });
    return keys;
  },

  mousePos: (event) => {
    const rect = document.getElementById("mainCanvas").getBoundingClientRect();
    return { x: event.clientX - rect.left, y: event.clientY - rect.top };
  },

  toImageCoords: (x, y) => ({
    x: (x - App.state.transform.x) / App.state.transform.scale,
    y: (y - App.state.transform.y) / App.state.transform.scale,
  }),

  toViewportCoords: (x, y) => ({
    x: x * App.state.transform.scale + App.state.transform.x,
    y: y * App.state.transform.scale + App.state.transform.y,
  }),

  hitTestPoint: (x, y) => {
    let best = null;
    let bestDistance = Infinity;
    for (const [key, point] of Object.entries(App.state.annotation.keypoints || {})) {
      if (!App.pointIsVisible(point)) continue;
      const view = App.toViewportCoords(point.x, point.y);
      const distance = Math.hypot(view.x - x, view.y - y);
      if (distance < 12 && distance < bestDistance) {
        best = key;
        bestDistance = distance;
      }
    }
    return best;
  },

  hitTestConnection: (x, y) => {
    let best = null;
    let bestDistance = Infinity;
    for (const connection of App.visibleConnections()) {
      const p1 = App.state.annotation.keypoints[connection.point_a];
      const p2 = App.state.annotation.keypoints[connection.point_b];
      if (!App.pointIsVisible(p1) || !App.pointIsVisible(p2)) continue;
      const a = App.toViewportCoords(p1.x, p1.y);
      const b = App.toViewportCoords(p2.x, p2.y);
      const distance = App.distanceToSegment(x, y, a.x, a.y, b.x, b.y);
      if (distance < 8 && distance < bestDistance) {
        best = connection.id;
        bestDistance = distance;
      }
    }
    return best;
  },

  distanceToSegment: (px, py, ax, ay, bx, by) => {
    const dx = bx - ax;
    const dy = by - ay;
    if (dx === 0 && dy === 0) return Math.hypot(px - ax, py - ay);
    const t = App.clamp(((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy), 0, 1);
    return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
  },

  fitToScreen: () => {
    if (!App.state.image) return;
    const canvas = document.getElementById("mainCanvas");
    const rect = canvas.getBoundingClientRect();
    const scale = Math.min(rect.width / App.state.image.width, rect.height / App.state.image.height) * 0.92;
    App.state.transform = {
      x: (rect.width - App.state.image.width * scale) / 2,
      y: (rect.height - App.state.image.height * scale) / 2,
      scale,
    };
  },

  drawLoop: () => {
    App.draw();
    requestAnimationFrame(App.drawLoop);
  },

  draw: () => {
    const canvas = document.getElementById("mainCanvas");
    const rect = canvas.getBoundingClientRect();
    const ctx = canvas.getContext("2d");
    ctx.save();
    ctx.setTransform(window.devicePixelRatio || 1, 0, 0, window.devicePixelRatio || 1, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);
    ctx.fillStyle = "#1b1d1f";
    ctx.fillRect(0, 0, rect.width, rect.height);
    if (!App.state.image) {
      ctx.fillStyle = "#c7d0d5";
      ctx.font = "15px sans-serif";
      ctx.fillText("打开图像后开始标注", 24, 34);
      ctx.restore();
      return;
    }

    ctx.translate(App.state.transform.x, App.state.transform.y);
    ctx.scale(App.state.transform.scale, App.state.transform.scale);
    ctx.drawImage(App.state.image, 0, 0);
    App.drawConnections(ctx);
    App.drawPoints(ctx);
    ctx.restore();
  },

  visibleConnections: () => (App.state.annotation?.connections || []).filter((item) => item.visible !== false),

  drawConnections: (ctx) => {
    if (!App.state.annotation) return;
    const inv = 1 / App.state.transform.scale;
    App.visibleConnections().forEach((connection) => {
      const p1 = App.state.annotation.keypoints[connection.point_a];
      const p2 = App.state.annotation.keypoints[connection.point_b];
      if (!App.pointIsVisible(p1) || !App.pointIsVisible(p2)) return;
      const selected = connection.id === App.state.selectedConnection;
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.strokeStyle = selected ? "#ffffff" : App.connectionColor(connection);
      ctx.lineWidth = (selected ? 3 : 1.7) * inv;
      if (connection.source === "manual") ctx.setLineDash([7 * inv, 4 * inv]);
      ctx.stroke();
      ctx.setLineDash([]);
    });
  },

  drawPoints: (ctx) => {
    if (!App.state.annotation) return;
    const inv = 1 / App.state.transform.scale;
    App.allKeys().forEach((key) => {
      const point = App.state.annotation.keypoints[key];
      if (!App.pointIsVisible(point)) return;
      const color = point.side === "left" ? "#d9480f" : "#1c7ed6";
      const pending = key === App.state.pendingConnectionStart;
      ctx.beginPath();
      ctx.arc(point.x, point.y, (pending ? 7 : 5.2) * inv, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.lineWidth = (pending || key === App.state.selectedPoint ? 3 : 2) * inv;
      ctx.strokeStyle = pending || key === App.state.selectedPoint ? "#ffffff" : App.sourceColor(point.source);
      ctx.stroke();
      if (App.state.showLabels) {
        ctx.font = `${12 * inv}px sans-serif`;
        ctx.lineWidth = 3 * inv;
        const text = `#${point.number}`;
        ctx.strokeStyle = "rgba(0,0,0,0.75)";
        ctx.strokeText(text, point.x + 8 * inv, point.y - 7 * inv);
        ctx.fillStyle = "#ffffff";
        ctx.fillText(text, point.x + 8 * inv, point.y - 7 * inv);
      }
    });
  },

  connectionColor: (connection) => {
    if (connection.source === "manual") return "#ffd43b";
    if (connection.label && connection.label.includes("图像左")) return "#63e6be";
    return "#74c0fc";
  },

  sourceColor: (source) => {
    if (source === "retuve") return "#6741d9";
    if (source === "manual") return "#2f9e44";
    if (App.isEstimatedSource(source)) return "#e67700";
    return "#adb5bd";
  },

  refreshPointList: () => {
    document.querySelectorAll(".point-row").forEach((row) => {
      const key = row.dataset.key;
      const point = App.state.annotation?.keypoints?.[key];
      const visible = App.pointIsVisible(point);
      row.classList.toggle("missing", !visible);
      row.classList.toggle("selected", key === App.state.selectedPoint);
      const checkbox = row.querySelector("input");
      checkbox.checked = visible;
      const badge = row.querySelector(".source-badge");
      const source = visible ? point.source || "manual" : "missing";
      badge.className = `source-badge ${App.sourceClass(source)}`;
      badge.textContent = App.sourceLabel(source);
    });
  },

  renderConnectionList: () => {
    const list = document.getElementById("connectionList");
    list.innerHTML = "";
    if (!App.state.annotation) return;
    App.visibleConnections().forEach((connection) => {
      const row = document.createElement("div");
      row.className = `connection-row ${connection.id === App.state.selectedConnection ? "selected" : ""}`;
      row.innerHTML = `<span>${App.connectionLabel(connection)}</span><small>${connection.source === "manual" ? "人工" : "默认"}</small>`;
      row.addEventListener("click", () => App.selectConnection(connection.id));
      list.appendChild(row);
    });
  },

  sourceLabel: (source) => {
    if (source === "retuve") return "retuve";
    if (source === "manual") return "人工";
    if (source === "imported_label") return "导入";
    if (source === "pose11_side") return "模型";
    if (App.isEstimatedSource(source)) return "估计";
    return "缺失";
  },

  sourceClass: (source) => {
    if (App.isEstimatedSource(source)) return "estimated";
    return source || "missing";
  },

  isEstimatedSource: (source) =>
    source === "estimated" ||
    source === "hippelvis_like_mask" ||
    source === "imported_label" ||
    source === "pose11_side",

  updateSummary: () => {
    const counts = { model: 0, estimated: 0, manual: 0, missing: 0 };
    if (!App.state.annotation) {
      counts.missing = 22;
    } else {
      Object.values(App.state.annotation.keypoints).forEach((point) => {
        if (!App.pointIsVisible(point)) counts.missing += 1;
        else if (point.source === "retuve" || point.source === "pose11_side") counts.model += 1;
        else if (point.source === "manual") counts.manual += 1;
        else counts.estimated += 1;
      });
    }
    const complete = 22 - counts.missing;
    document.getElementById("completionText").textContent = `${complete} / 22`;
    document.getElementById("countModel").textContent = counts.model;
    document.getElementById("countEstimated").textContent = counts.estimated;
    document.getElementById("countManual").textContent = counts.manual;
    document.getElementById("countMissing").textContent = counts.missing;
  },

  updateSelectedBox: () => {
    const label = document.getElementById("selectedLabel");
    const coords = document.getElementById("selectedCoords");
    const connectionLabel = document.getElementById("selectedConnectionLabel");
    if (!App.state.annotation || !App.state.selectedPoint) {
      label.textContent = "-";
      coords.textContent = App.state.activeTool === "line" && App.state.pendingConnectionStart ? "请选择第二个点" : "-";
    } else {
      const point = App.state.annotation.keypoints[App.state.selectedPoint];
      label.textContent = point.label;
      coords.textContent = App.pointIsVisible(point) ? `${point.x.toFixed(1)}, ${point.y.toFixed(1)} · ${App.sourceLabel(point.source)}` : "缺失";
    }
    const selectedConnection = App.state.annotation?.connections?.find((item) => item.id === App.state.selectedConnection);
    connectionLabel.textContent = selectedConnection ? App.connectionLabel(selectedConnection) : "未选中连线";
  },

  connectionLabel: (connection) => {
    const a = App.state.annotation?.keypoints?.[connection.point_a];
    const b = App.state.annotation?.keypoints?.[connection.point_b];
    if (!a || !b) return connection.label || "连线";
    const sideA = a.side === "left" ? "左" : "右";
    const sideB = b.side === "left" ? "左" : "右";
    return `${sideA}#${a.number} - ${sideB}#${b.number}`;
  },

  renderWarnings: () => {
    const container = document.getElementById("warningList");
    container.innerHTML = "";
    const warnings = App.state.annotation?.auto_initialization?.warnings || [];
    warnings.forEach((warning) => {
      const item = document.createElement("div");
      item.className = "warning-item";
      item.textContent = warning;
      container.appendChild(item);
    });
  },

  clamp: (value, min, max) => Math.max(min, Math.min(max, value)),
};

window.addEventListener("load", App.init);
